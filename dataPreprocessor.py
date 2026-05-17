"""
DataPreprocessor — ysaForB2b

6-adımlı preprocessing pipeline:
  1. Session düzeyinde özellik çıkarımı
  2. Kategorik event alanlarının integer encoding'i
  3. tags alanının multi-hot encoding'i
  4. message_normalized alanının offline BERT embedding'i
  5. Dizilerin sabit uzunluğa (MAX_SESSION_LENGTH = 30) getirilmesi
  6. Tüm artifact'ların diske kaydedilmesi

Girdi: tagged_events_example.json (veya aynı şemadaki JSONL/JSON dosyası)
Çıktı: processed/ ve vocab/ klasörlerine JSON artifact'lar
"""

import gc
import json
import logging
import math
from pathlib import Path

import numpy as np
from typing import Dict, List, Literal, Optional, Tuple


_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sabit sözlükler — şemadan türetilmiş, deterministik
# ---------------------------------------------------------------------------

_DAY_OF_WEEK: Dict[str, int] = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3,
    "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7,
}

_TIME_OF_DAY: Dict[str, int] = {
    "night": 1, "morning": 2, "afternoon": 3, "evening": 4,
}

# urgency_level sıralı encode — PAD=0 rezerve
_URGENCY: Dict[str, int] = {
    "PAD": 0, "low": 1, "medium": 2, "high": 3,
}

# integer encode uygulanacak olay alanları
_CATEGORICAL_FIELDS = [
    "event_type",
    "event_category",
    "action_type",
    "message_template",
    "outcome",
    "sentiment",
    "actor_role",
    "platform",
    "user_journey_stage",
]


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def _build_vocab(values: List[str]) -> Dict[str, int]:
    """PAD=0 rezerve, geri kalanlar alfabetik sırayla 1'den başlar."""
    vocab: Dict[str, int] = {"PAD": 0}
    for idx, token in enumerate(sorted(set(values)), start=1):
        vocab[token] = idx
    return vocab


def _log1p_normalize(values: List[float]) -> List[float]:
    """
    Sağa çarpık dağılımlar için log1p + 0-1 ölçekleme.
    duration_minutes ve event_count gibi alanlarda kullanılır.
    """
    transformed = [math.log1p(v) for v in values]
    min_v = min(transformed)
    max_v = max(transformed)
    span = max_v - min_v if max_v != min_v else 1.0
    return [(v - min_v) / span for v in transformed]


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """
    tagged_events_example.json şemasındaki oturum verilerini
    sinir ağına hazır sabit boyutlu tensör formatına dönüştürür.

    Kullanım:
        preprocessor = DataPreprocessor("tagged_events_example.json")
        output = preprocessor.fit_transform()
        # veya diske kaydetmek için:
        preprocessor.save_artifacts("output/")
    """

    # Her oturumun sabit uzunlukta temsil edileceği pencere boyutu.
    # Bu değer padding, truncation ve tensör boyutlarının tek kaynağıdır;
    # değiştirmek tüm pipeline'ı otomatik olarak etkiler.
    MAX_SESSION_LENGTH: int = 30

    def __init__(
        self,
        data_path: str,
        truncation: Literal["pre", "post"] = "post",
        padding: Literal["pre", "post"] = "post",
        time_shift: bool = True,
        time_shift_mode: Literal["last", "all"] = "last",
        label_field: str = "event_type",
        min_session_length: int = 2,
        bert_model_name: str = "bert-base-multilingual-cased",
        bert_pooling: Literal["cls", "mean"] = "cls",
        bert_batch_size: int = 32,
    ) -> None:
        """
        Parametreler:
            data_path       : JSON veya JSONL formatındaki veri dosyasının yolu.
            truncation      : "post" → diziyi baştan keser (ilk 30 olay korunur).
                              "pre"  → diziyi sondan keser (son 30 olay korunur).
            padding         : "post" → sona 0 ekler; "pre" → başa 0 ekler.
            time_shift      : True ise X dizileri son eleman hariç tutulur,
                              y ise son event'in label_field ID'si olur.
            time_shift_mode : "last" → her oturumdan tek ornek (son event).
                              "all"  → her prefix -> bir sonraki event ornegi.
            label_field     : hedef etiketin alinacagi kategorik alan.
            min_session_length : time shifting için minimum event sayisi.
            bert_model_name : HuggingFace model kimliği.
            bert_pooling    : "cls" → [CLS] token; "mean" → attention-weighted ortalama.
            bert_batch_size : BERT inference'ta kaç metin birlikte işlenir.
        """
        self.data_path       = Path(data_path)
        self.truncation      = truncation
        self.padding         = padding
        self.time_shift      = time_shift
        self.time_shift_mode = time_shift_mode
        self.label_field     = label_field
        self.min_session_length = min_session_length
        self.bert_model_name = bert_model_name
        self.bert_pooling    = bert_pooling
        self.bert_batch_size = bert_batch_size

        # load_data() çağrıldığında doldurulur; tüm adımların ortak giriş kaynağıdır
        self.raw_sessions: List[dict] = []

        # Adım 2: fit sırasında inşa edilen kategorik sözlükler {alan_adı: {token: int}}
        # PAD=0 garantisi sayesinde sıfır-padding'li zaman adımları maskeleme dışında kalabilir
        self.vocabs: Dict[str, Dict[str, int]] = {}

        # Adım 3: tag multi-hot matrisi için sıralı referans listesi ve hızlı lookup
        self.tag_vocab: List[str] = []
        self.tag2idx: Dict[str, int] = {}

        # Adım 4: benzersiz metin → 768-boyutlu vektör önbelleği
        # Aynı message_normalized birden fazla event'te geçse de yalnızca bir kez encode edilir
        self.message_embeddings: Dict[str, List[float]] = {}

        # BERT model bileşenleri — ilk embedding isteğinde _lazy_load_bert() tarafından yüklenir
        # Başlangıçta None bırakmak, BERT gerektirmeyen pipeline'larda gereksiz yükü önler
        self._tokenizer  = None
        self._bert_model = None
        self._device: str = "cpu"

    # -----------------------------------------------------------------------
    # Veri yükleme
    # -----------------------------------------------------------------------

    def load_data(self) -> List[dict]:
        """
        JSON array veya JSONL formatını otomatik algılar ve yükler.

        JSON array  : [ {...}, {...} ]
        JSONL       : Her satırda bağımsız bir JSON nesnesi
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {self.data_path}")

        raw = self.data_path.read_text(encoding="utf-8").strip()

        # İlk karakter '{' ise her satır bağımsız JSON nesnesi → JSONL formatı
        # Aksi hâlde tüm içerik tek bir JSON dizisi olarak parse edilir
        if raw.startswith("{"):  # JSONL
            self.raw_sessions = [
                json.loads(line) for line in raw.splitlines() if line.strip()
            ]
        else:  # JSON array
            self.raw_sessions = json.loads(raw)

        # En üst seviye liste değilse pipeline ilerleyemez; erken hata ver
        if not isinstance(self.raw_sessions, list):
            raise ValueError("Veri en üst seviyede JSON dizisi olmalıdır.")

        return self.raw_sessions

    # -----------------------------------------------------------------------
    # ADIM 1 — Session düzeyinde özellik çıkarımı
    # -----------------------------------------------------------------------

    def extract_session_features(self, sessions: Optional[List[dict]] = None) -> List[Dict]:
        """
        Her oturum için sessionStartTime ve summary bloklarından
        sabit boyutlu sayısal bir özellik sözlüğü üretir.

        Döndürülen liste raw_sessions ile birebir aynı sıradadır.

        Özellik grupları:
          - Zaman : hour_of_day, month, day_of_week (ordinal), time_of_day (ordinal)
          - Platform : web/mobile/email/system one-hot
          - Oturum metrikleri : event_count ve duration_minutes (log1p normalize)
          - Boolean sinyaller : has_failure, has_purchase, has_bid, has_rfx, has_order
        """
        features = []
        use_sessions = sessions if sessions is not None else self.raw_sessions
        for session in use_sessions:
            # sessionStartTime → zaman ve platform özellikleri
            st = session.get("sessionStartTime", {})
            # summary → oturum düzey istatistikler ve davranış sinyalleri
            sm = session.get("summary", {})

            features.append({
                # --- Zaman özellikleri ---
                # hour_of_day ham tutulur; model gece/gündüz örüntüsünü öğrenir
                "hour_of_day":         st.get("hour_of_day", 0) / 24.0,
                "month":               (st.get("month", 1) - 1) / 11.0,
                # Haftanın günü ve gün dilimi ordinal: bilinmeyen değer → 0 (PAD semantiği)
                "day_of_week_id":      _DAY_OF_WEEK.get(st.get("day_of_week", ""), 0),
                "time_of_day_id":      _TIME_OF_DAY.get(st.get("time_of_day", ""), 0),
                # Hafta sonu flag'i: B2B'de hafta sonu aktivitesi anomali sinyali olabilir
                "is_weekend":          int(bool(st.get("is_weekend", False))),
                # --- Platform one-hot (bilinmeyen platform → tüm sütunlar 0) ---
                # Embedding katmanı yerine one-hot: platform az kategorili ve sırasız
                "platform_web":        int(st.get("platform") == "web"),
                "platform_mobile":     int(st.get("platform") == "mobile"),
                "platform_email":      int(st.get("platform") == "email"),
                "platform_system":     int(st.get("platform") == "system"),
                # --- Ham metrik değerler — normalizasyon aşağıda toplu uygulanır ---
                "event_count":         sm.get("event_count", 0),
                "duration_minutes":    sm.get("duration_minutes", 0.0),
                # --- Boolean davranış/dönüşüm sinyalleri ---
                # has_failure: oturumda hata/başarısız işlem yaşandı mı?
                "has_failure":         int(bool(sm.get("has_failure", False))),
                # has_purchase / has_bid / has_rfx / has_order: dönüşüm funnel katmanları
                "has_purchase":        int(bool(sm.get("has_purchase", False))),
                "has_bid":             int(bool(sm.get("has_bid", False))),
                "has_rfx":             int(bool(sm.get("has_rfx", False))),
                "has_order":           int(bool(sm.get("has_order", False))),
            })

        # event_count ve duration_minutes sağa çarpık dağılım gösterir (çok sayıda
        # kısa oturum, az sayıda çok uzun oturum). log1p bu asimetriyi düzeltir,
        # ardından 0-1 ölçekleme farklı birimleri karşılaştırılabilir kılar.
        norm_ec  = _log1p_normalize([float(f["event_count"])      for f in features])
        norm_dur = _log1p_normalize([float(f["duration_minutes"]) for f in features])

        # Normalize değerleri orijinal dict'e ekle; ham değerler hata ayıklama için korunur
        for i, feat in enumerate(features):
            feat["event_count_norm"]      = norm_ec[i]
            feat["duration_minutes_norm"] = norm_dur[i]

        return features

    # -----------------------------------------------------------------------
    # ADIM 2 — Kategorik alanların integer encoding'i
    # -----------------------------------------------------------------------

    def build_categorical_vocabs(self) -> Dict[str, Dict[str, int]]:
        """
        _CATEGORICAL_FIELDS listesindeki her alan için PAD=0 kuralıyla
        deterministik integer sözlük inşa eder.

        Neden deterministik:
          Sözlükler her çalıştırmada aynı alfabetik sırayı korur.
          Bu, train/val/test bölünmesinden sonra token çakışmasını önler.

        urgency_level sıralı bir alan olduğu için _URGENCY sabitiyle
        ayrıca işlenir (low < medium < high sırası korunur).
        """
        # Her alan için tüm veri setindeki değerleri topla; vocab sonradan bu listeden üretilir
        all_values: Dict[str, List[str]] = {field: [] for field in _CATEGORICAL_FIELDS}

        for session in self.raw_sessions:
            for event in session.get("sequentialEvents", []):
                for field in _CATEGORICAL_FIELDS:
                    val = event.get(field)
                    # None değerleri vocab'a dahil etme; encode sırasında PAD (0) atanır
                    if val is not None:
                        all_values[field].append(str(val))

        # _build_vocab: tekil + alfabetik sıralama → her çalıştırmada aynı token-ID eşleşmesi
        self.vocabs = {
            field: _build_vocab(vals) for field, vals in all_values.items()
        }
        # urgency_level sıralı (ordinal) encode: low=1 < medium=2 < high=3
        # Alfabetik sıra bu anlamsal sırayı yansıtmadığı için sabit dict kullanılır
        self.vocabs["urgency_level"] = _URGENCY

        return self.vocabs

    def encode_categorical_sequences(
        self,
        sessions: Optional[List[dict]] = None,
    ) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
        """
        Her session icin her kategorik alan adina ait padded integer array uretir.
        Cikti: {alan_adi: np.ndarray(N, T, int32)}
        """
        if not self.vocabs:
            self.build_categorical_vocabs()

        all_fields = list(self.vocabs.keys())
        use_sessions = sessions if sessions is not None else self.raw_sessions
        N = len(use_sessions)
        T = self.MAX_SESSION_LENGTH
        session_ids: List[str] = []
        user_ids: List[str] = []

        # Dogrudan numpy array olustur — Python list overhead yok
        arrays: Dict[str, np.ndarray] = {
            f: np.zeros((N, T), dtype=np.int32) for f in all_fields
        }

        for i, session in enumerate(use_sessions):
            session_ids.append(session.get("sessionId", ""))
            user_ids.append(session.get("userId", ""))
            events = session.get("sequentialEvents", [])
            # Truncation
            if len(events) > T:
                events = events[-T:] if self.truncation == "pre" else events[:T]
            n = len(events)
            for field in all_fields:
                vocab = self.vocabs[field]
                row = np.array(
                    [vocab.get(str(e.get(field)) if e.get(field) is not None else "PAD", 0)
                     for e in events],
                    dtype=np.int32,
                )
                if self.padding == "pre":
                    arrays[field][i, T - n:] = row
                else:
                    arrays[field][i, :n] = row

        return arrays, session_ids, user_ids

    def _build_time_shifted_samples(
        self,
    ) -> Tuple[List[dict], List[int]]:
        """
        Her oturumda son event'i hedef (y) olarak ayirir ve
        girdiler (X) icin kalan event dizisini dondurur.

        Not: PAD/None etiketler atlanir, kisa oturumlar filtrelenir.
        """
        if not self.vocabs:
            self.build_categorical_vocabs()

        if self.label_field not in self.vocabs:
            raise ValueError(f"Label field not in vocabs: {self.label_field}")

        label_vocab = self.vocabs[self.label_field]
        shifted_sessions: List[dict] = []
        labels: List[int] = []

        for session in self.raw_sessions:
            events = session.get("sequentialEvents", [])
            if len(events) < self.min_session_length:
                continue

            if self.time_shift_mode == "last":
                candidate_pairs = [(events[:-1], events[-1])]
            elif self.time_shift_mode == "all":
                candidate_pairs = [(events[:i], events[i]) for i in range(1, len(events))]
            else:
                raise ValueError(f"Unknown time_shift_mode: {self.time_shift_mode}")

            for prefix, target_event in candidate_pairs:
                if not prefix:
                    continue

                raw_label = target_event.get(self.label_field)
                label_token = str(raw_label) if raw_label is not None else "PAD"
                label_id = label_vocab.get(label_token, 0)
                # PAD/unknown etiketler egitimde kullanilmaz
                if label_id == 0:
                    continue

                new_session = dict(session)
                new_session["sequentialEvents"] = list(prefix)

                shifted_sessions.append(new_session)
                labels.append(label_id)

        return shifted_sessions, labels

    # -----------------------------------------------------------------------
    # ADIM 3 — tags multi-hot encoding
    # -----------------------------------------------------------------------

    def build_tag_vocab(self, sessions: Optional[List[dict]] = None) -> List[str]:
        """
        Tüm veri setindeki benzersiz tag'leri alfabetik sıralar.

        Neden multi-hot:
          tags alanı birden fazla etiket içerebilir (örn. ["bid", "submitted",
          "has_buyer"]). Integer encode multi-label durumunu temsil edemez;
          her tag bağımsız bir 0/1 feature olmalıdır.
        """
        tags_seen: set = set()
        use_sessions = sessions if sessions is not None else self.raw_sessions
        for session in use_sessions:
            for event in session.get("sequentialEvents", []):
                for tag in event.get("tags", []):
                    tags_seen.add(tag)
        # Alfabetik sıralama vocab boyutunu ve indeks atamasını deterministik yapar
        self.tag_vocab = sorted(tags_seen)
        # tag2idx: encode sırasında O(1) lookup için ters eşleme
        self.tag2idx   = {tag: idx for idx, tag in enumerate(self.tag_vocab)}
        return self.tag_vocab

    def encode_and_pad_tags_to_file(
        self, sessions: Optional[List[dict]], out_path: Path
    ) -> int:
        """
        Tag dizilerini encode eder, padding uygular ve dogrudan
        memory-mapped .npy dosyasina yazar. Python list hic olusturulmaz.
        Cikti: (N, T, tag_vocab_size) uint8 dosyasi.
        Dondurur: tag_vocab_size.
        """
        if not self.tag_vocab:
            self.build_tag_vocab(sessions)

        use_sessions = sessions if sessions is not None else self.raw_sessions
        N        = len(use_sessions)
        T        = self.MAX_SESSION_LENGTH
        tag_size = len(self.tag_vocab)

        mmap = np.lib.format.open_memmap(
            str(out_path), mode="w+", dtype=np.uint8, shape=(N, T, tag_size)
        )

        for i, session in enumerate(use_sessions):
            events = session.get("sequentialEvents", [])
            if len(events) > T:
                events = events[-T:] if self.truncation == "pre" else events[:T]
            n = len(events)
            offset = T - n if self.padding == "pre" else 0
            for j, event in enumerate(events):
                for tag in event.get("tags", []):
                    idx = self.tag2idx.get(tag)
                    if idx is not None:
                        mmap[i, offset + j, idx] = 1

        del mmap  # flush & close
        return tag_size

    # -----------------------------------------------------------------------
    # ADIM 4 — Offline BERT embedding (message_normalized)
    # -----------------------------------------------------------------------

    def _lazy_load_bert(self) -> None:
        """BERT modelini yalnızca ilk ihtiyaç duyulduğunda belleğe yükler."""
        # Yükleme daha önce yapıldıysa tekrar yükleme
        if self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer, logging as hf_logging
        except Exception as exc:
            raise ImportError(
                "BERT embedding için 'torch' ve 'transformers' gerekli.\n"
                "pip install torch transformers"
            ) from exc

        _prev_hf_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        _from_pretrained_kwargs = dict(local_files_only=True)
        try:
            self._tokenizer  = AutoTokenizer.from_pretrained(self.bert_model_name, **_from_pretrained_kwargs)
            self._bert_model = AutoModel.from_pretrained(self.bert_model_name, **_from_pretrained_kwargs)
        except Exception:
            # Cache'de yoksa internetten indir, sonraki çalıştırmalar cache'den gelir
            _from_pretrained_kwargs = {}
            self._tokenizer  = AutoTokenizer.from_pretrained(self.bert_model_name)
            self._bert_model = AutoModel.from_pretrained(self.bert_model_name)
        hf_logging.set_verbosity(_prev_hf_verbosity)
        # eval() modu dropout ve batch norm'u kapatır; inference için gerekli
        self._bert_model.eval()

        import torch
        # CUDA → Apple MPS → CPU sırasıyla en hızlı backend'i seç
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        self._bert_model.to(self._device)

    def extract_message_embeddings_offline(
        self, sessions: Optional[List[dict]] = None
    ) -> Dict[str, List[float]]:
        """
        Benzersiz message_normalized metinlerini bir kez işler ve önbelleğe alır.

        Neden offline:
          Eğitim sırasında her adımda BERT çalıştırmak çok maliyetlidir.
          Bu yöntemle her benzersiz metin yalnızca bir kez işlenir, sonuç
          dosyaya kaydedilir ve eğitim lookup ile çalışır.

        Neden BERT:
          message_normalized "Tedarikçi teklif verdi." ile
          "COMPANY_X firmasının talebine teklif verildi." arasındaki
          anlamsal farkı sadece integer ID yakalayamaz; BERT bu farkı
          768 boyutlu vektör uzayında kodlar.
        """
        import hashlib
        use_sessions = sessions if sessions is not None else self.raw_sessions

        unique_texts = sorted({
            event.get("message_normalized", "")
            for session in use_sessions
            for event in session.get("sequentialEvents", [])
            if event.get("message_normalized")
        })

        if not unique_texts:
            return self.message_embeddings

        # Disk cache: .npz (numpy binary) — JSON'a göre 5-10x küçük, parse yükü sıfır
        _cache_key  = hashlib.md5("\n".join(unique_texts).encode()).hexdigest()
        _cache_path = Path(self.data_path).parent / f".bert_emb_cache_{_cache_key}.npz"
        if _cache_path.exists():
            _LOGGER.info("BERT embedding cache bulundu, diskten yukleniyor: %s", _cache_path)
            _npz = np.load(str(_cache_path), allow_pickle=False)
            # texts dizisi ve vectors matrisi olarak saklanir
            _texts   = _npz["texts"].tolist()
            _vectors = _npz["vectors"]          # (N, emb_dim) float32
            self.message_embeddings = {t: _vectors[i] for i, t in enumerate(_texts)}
            return self.message_embeddings

        self._lazy_load_bert()
        import torch

        total_batches = (len(unique_texts) + self.bert_batch_size - 1) // self.bert_batch_size
        # Python list yerine dogrudan numpy array biriktir → Python object overhead yok
        all_vecs: List[np.ndarray] = []

        with torch.no_grad():
            for batch_idx, i in enumerate(range(0, len(unique_texts), self.bert_batch_size)):
                batch  = unique_texts[i : i + self.bert_batch_size]
                tokens = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64,
                )
                tokens  = {k: v.to(self._device) for k, v in tokens.items()}
                outputs = self._bert_model(**tokens)

                if self.bert_pooling == "mean":
                    last = outputs.last_hidden_state
                    mask = tokens["attention_mask"].unsqueeze(-1).expand(last.size()).float()
                    vecs = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                else:
                    vecs = outputs.last_hidden_state[:, 0, :]

                all_vecs.append(vecs.detach().cpu().to(torch.float32).numpy())

                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                    _LOGGER.info(
                        "BERT encoding: %d/%d batch islendi (%d/%d metin)",
                        batch_idx + 1, total_batches,
                        min(i + self.bert_batch_size, len(unique_texts)), len(unique_texts),
                    )

        vectors_arr = np.concatenate(all_vecs, axis=0)  # (N, emb_dim) float32
        del all_vecs

        # Cache'e yaz: texts string dizisi + vectors float32 matrisi
        _LOGGER.info("BERT embedding cache diske yaziliyor: %s", _cache_path)
        np.savez_compressed(
            str(_cache_path),
            texts=np.array(unique_texts, dtype=str),
            vectors=vectors_arr,
        )

        self.message_embeddings = {t: vectors_arr[i] for i, t in enumerate(unique_texts)}
        return self.message_embeddings

    def build_message_embedding_sequences(
        self, sessions: Optional[List[dict]] = None
    ) -> np.ndarray:
        """
        Her oturum icin message_normalized embedding'lerini
        siralanmis sekilde toplar (padding oncesi).
        Sonucu dogrudan numpy array olarak dondurur — Python list overhead'i yok.
        """
        if not self.message_embeddings:
            self.extract_message_embeddings_offline(sessions)

        emb_dim  = next(iter(self.message_embeddings.values())).shape[0] if self.message_embeddings else 768
        zero_vec = np.zeros(emb_dim, dtype=np.float32)

        use_sessions = sessions if sessions is not None else self.raw_sessions
        result: List[List[np.ndarray]] = []
        for session in use_sessions:
            seq = [
                self.message_embeddings.get(
                    event.get("message_normalized", ""), zero_vec
                )
                for event in session.get("sequentialEvents", [])
            ]
            result.append(seq)

        return result

    # -----------------------------------------------------------------------
    # ADIM 5 — Padding & Truncation (MAX_SESSION_LENGTH = 30)
    # -----------------------------------------------------------------------
    #
    # Truncation (kırpma):
    #   "post" → diziyi ilk MAX_SESSION_LENGTH olayına kırpar (baştan alır).
    #   "pre"  → diziyi son  MAX_SESSION_LENGTH olayına kırpar (sondan alır).
    #
    # Padding (doldurma):
    #   "post" → kısa dizinin sonuna 0 / sıfır vektör ekler.
    #   "pre"  → kısa dizinin başına 0 / sıfır vektör ekler.
    #
    # Tüm 3 veri türü (integer, multi-hot, embedding) aynı kuralı uygular.

    # _pad_truncate_int ve _pad_truncate_multihot artik kullanilmiyor.
    # encode_categorical_sequences dogrudan padded numpy array uretiyor.
    # encode_and_pad_tags_to_file dogrudan memmap'e yaziyor.

    def _pad_truncate_embeddings_to_file(self, sequences, out_path: Path) -> int:
        """BERT embedding dizilerini MAX_SESSION_LENGTH'e getirir ve
        dogrudan diske memory-mapped .npy olarak yazar.
        Bellekte hicbir zaman tam array tutulmaz.
        Dondurur: emb_dim (metadata icin)."""
        T       = self.MAX_SESSION_LENGTH
        emb_dim = sequences[0][0].shape[0] if sequences and len(sequences[0]) > 0 else 768
        N       = len(sequences)

        # np.lib.format.open_memmap: duzgun .npy header + disk-backed array
        mmap = np.lib.format.open_memmap(
            str(out_path), mode="w+", dtype=np.float32, shape=(N, T, emb_dim)
        )
        for i, seq in enumerate(sequences):
            if not seq:
                continue
            vecs = np.stack(seq, axis=0)   # (len_seq, emb_dim)
            n = len(vecs)
            if n > T:
                vecs = vecs[-T:] if self.truncation == "pre" else vecs[:T]
                n = T
            if self.padding == "pre":
                mmap[i, T - n:] = vecs
            else:
                mmap[i, :n]     = vecs
        del mmap  # flush & close
        return emb_dim

    # -----------------------------------------------------------------------
    # ADIM 6 — Artifact kaydetme
    # -----------------------------------------------------------------------

    def save_artifacts(self, output_dir: str) -> Dict[str, str]:
        """
        Pipeline'ı adım adım çalıştırır ve her artifact'ı üretildikten hemen
        sonra diske yazar. Her aşama bittiğinde büyük ara veriler bellekten
        atılır — düşük RAM'li makinelerde (örn. 8 GB Mac) OOM'u önler.

        Çıktı dosyaları:
          output_dir/
            vocab/
              all_vocabs.json
              tag_vocab.json
            processed/
              session_features.json
              event_sequences.json
              tag_sequences.json
              message_embedding_sequences.json
              labels.json (time_shift=True ise)
              metadata.json
        """
        out = Path(output_dir)
        (out / "vocab").mkdir(parents=True, exist_ok=True)
        (out / "processed").mkdir(parents=True, exist_ok=True)

        def _dump(path: Path, obj) -> None:
            with path.open("w", encoding="utf-8") as _f:
                json.dump(obj, _f, ensure_ascii=False)

        paths: Dict[str, Path] = {
            "all_vocabs":                   out / "vocab"     / "all_vocabs.json",
            "tag_vocab":                    out / "vocab"     / "tag_vocab.json",
            "session_features":             out / "processed" / "session_features.json",
            "event_sequences":              out / "processed" / "event_sequences.npz",
            "tag_sequences":                out / "processed" / "tag_sequences.npy",
            "message_embedding_sequences":  out / "processed" / "message_embedding_sequences.npy",
            "user_ids":                     out / "processed" / "user_ids.npy",
            "metadata":                     out / "processed" / "metadata.json",
        }

        # 0) Veri yükle ve vocabs'ı çıkar (küçük yapılar, bellekte tut)
        if not self.raw_sessions:
            self.load_data()
        if not self.vocabs:
            self.build_categorical_vocabs()
        self.build_tag_vocab()

        _dump(paths["all_vocabs"], self.vocabs)
        _dump(paths["tag_vocab"],  self.tag_vocab)
        _LOGGER.info("Vocab dosyalari yazildi.")

        # 1) Time shift (etkin ise) → büyük raw_sessions'ı küçült
        labels: Optional[List[int]] = None
        if self.time_shift:
            sessions_for_x, labels = self._build_time_shifted_samples()
            if not sessions_for_x:
                raise ValueError("Time shifting sonrasi gecerli oturum kalmadi.")
            # Ham veriye artık ihtiyacımız yok — bellekten at
            self.raw_sessions = []
            gc.collect()
        else:
            sessions_for_x = self.raw_sessions

        num_sessions = len(sessions_for_x)
        _LOGGER.info("Pipeline'a giren ornek sayisi: %d", num_sessions)

        # 2) Session features → yaz → boşalt
        session_features = self.extract_session_features(sessions_for_x)
        session_feature_dim = len(session_features[0]) if session_features else 0
        _dump(paths["session_features"], session_features)
        del session_features
        gc.collect()
        _LOGGER.info("session_features yazildi.")

        # 3) Kategorik diziler + user_ids → padded numpy array → .npz / .npy
        cat_arrays, _, user_ids_raw = self.encode_categorical_sequences(sessions_for_x)
        np.savez_compressed(str(paths["event_sequences"]), **cat_arrays)
        del cat_arrays
        gc.collect()
        _LOGGER.info("event_sequences yazildi.")

        # userId vocab: PAD=0, geri kalanlar alfabetik sira
        user_vocab: Dict[str, int] = {"PAD": 0}
        for uid in sorted(set(user_ids_raw)):
            if uid and uid not in user_vocab:
                user_vocab[uid] = len(user_vocab)
        encoded_user_ids = np.array(
            [user_vocab.get(uid, 0) for uid in user_ids_raw], dtype=np.int32
        )
        np.save(str(paths["user_ids"]), encoded_user_ids)
        del encoded_user_ids, user_ids_raw
        gc.collect()
        _LOGGER.info("user_ids yazildi. Benzersiz kullanici: %d", len(user_vocab))

        # 4) Tag dizileri → dogrudan memmap'e (N, T, tag_vocab) uint8
        # Hic Python list olusturulmaz; RAM'de sadece bir satir tutuluyor
        tag_vocab_size = self.encode_and_pad_tags_to_file(
            sessions_for_x, paths["tag_sequences"]
        )
        gc.collect()
        _LOGGER.info("tag_sequences yazildi.")

        # 5) BERT embedding sıraları → padding → doğrudan diske memmap yaz
        raw_emb_seqs = self.build_message_embedding_sequences(sessions_for_x)
        emb_dim = self._pad_truncate_embeddings_to_file(
            raw_emb_seqs, paths["message_embedding_sequences"]
        )
        del raw_emb_seqs
        # BERT modelini de bellekten at (sonraki adımlar gerekmiyor)
        self._bert_model = None
        self._tokenizer = None
        gc.collect()
        _LOGGER.info("message_embedding_sequences yazildi.")

        # 6) Labels (time_shift varsa)
        label_num_classes = None
        if labels is not None:
            labels_path = out / "processed" / "labels.json"
            paths["labels"] = labels_path
            _dump(labels_path, labels)
            label_num_classes = len(self.vocabs[self.label_field])

        # 7) Metadata
        _dump(paths["metadata"], {
            "maxSessionLength":  self.MAX_SESSION_LENGTH,
            "numSessions":       num_sessions,
            "truncation":        self.truncation,
            "padding":           self.padding,
            "timeShifted":       self.time_shift,
            "timeShiftMode":     self.time_shift_mode,
            "labelName":         self.label_field if labels is not None else None,
            "labelNumClasses":   label_num_classes,
            "minSessionLength":  self.min_session_length,
            "bertModel":         self.bert_model_name,
            "bertPooling":       self.bert_pooling,
            "tagVocabSize":      len(self.tag_vocab),
            "categoricalVocabs": {k: len(v) for k, v in self.vocabs.items()},
            "embeddingDimension": emb_dim,
            "sessionFeatureDim": session_feature_dim,
            "userVocabSize":     len(user_vocab),
        })

        return {name: str(path) for name, path in paths.items()}

    # -----------------------------------------------------------------------
    # Ana pipeline — fit_transform
    # -----------------------------------------------------------------------

    def fit_transform(self) -> Dict[str, object]:
        """
        Tüm 6 adımı sırayla çalıştırır ve sonuçları tek sözlükte döndürür.

        Döndürülen anahtarlar:
          sessionIds                 : oturum kimlikleri
          userIds                    : kullanıcı kimlikleri
          sessionFeatures            : oturum düzey sayısal özellikler (normalize edilmiş)
          vocabs                     : kategorik sözlükler
          tag_vocab                  : tag listesi
          eventSequences             : padding'li integer diziler {alan_adı: matris}
          tagSequences               : padding'li multi-hot matrisler
          messageEmbeddingSequences  : padding'li BERT vektör dizileri
          embeddingDimension         : BERT vektör boyutu (genellikle 768)
          numSessions                : toplam oturum sayısı
          maxSessionLength           : 30
        """
        if not self.raw_sessions:
            self.load_data()

        # Time shifting: X dizilerini son eleman haric tut, y'yi ayir
        labels: Optional[List[int]] = None
        sessions_for_x = self.raw_sessions
        if self.time_shift:
            if not self.vocabs:
                self.build_categorical_vocabs()
            sessions_for_x, labels = self._build_time_shifted_samples()

            if not sessions_for_x:
                raise ValueError("Time shifting sonrasi gecerli oturum kalmadi.")

        # Adım 1 — sessionStartTime + summary → sabit boyutlu sayısal özellik sözlüğü
        session_features = self.extract_session_features(sessions_for_x)

        # Adım 2 — kategorik olay alanları → PAD=0'lı integer diziler, ardından 30'a getir
        cat_sequences, session_ids, user_ids = self.encode_categorical_sequences(sessions_for_x)
        padded_cat: Dict[str, List[List[int]]] = {
            field: self._pad_truncate_int(seqs)
            for field, seqs in cat_sequences.items()
        }

        # Adım 3 — tags listesi → multi-hot binary matris, ardından 30'a getir
        raw_tag_seqs    = self.encode_tag_sequences(sessions_for_x)
        padded_tag_seqs = self._pad_truncate_multihot(raw_tag_seqs)

        # Adım 4 — benzersiz message_normalized metinleri BERT ile encode et (önbellekli)
        # Adım 5 — embedding dizilerini 30'a getir (truncation/padding)
        raw_emb_seqs    = self.build_message_embedding_sequences(sessions_for_x)
        padded_emb_seqs = self._pad_truncate_embeddings(raw_emb_seqs)

        emb_dim = len(padded_emb_seqs[0][0]) if padded_emb_seqs else 768

        output: Dict[str, object] = {
            "sessionIds":                 session_ids,
            "userIds":                    user_ids,
            "sessionFeatures":            session_features,
            "vocabs":                     self.vocabs,
            "tag_vocab":                  self.tag_vocab,
            "eventSequences":             padded_cat,
            "tagSequences":               padded_tag_seqs,
            "messageEmbeddingSequences":  padded_emb_seqs,
            "embeddingDimension":         emb_dim,
            "numSessions":                len(session_ids),
            "maxSessionLength":           self.MAX_SESSION_LENGTH,
        }

        if labels is not None:
            output["labels"] = labels
            output["labelName"] = self.label_field
            output["labelNumClasses"] = len(self.vocabs[self.label_field])

        return output


# ---------------------------------------------------------------------------
# Örnek kullanım
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    preprocessor = DataPreprocessor(
        data_path="tagged_events_example.json",
        truncation="post",   # 30'dan uzun dizileri başından keser (ilk 30 olay korunur)
        padding="post",      # 30'dan kısa dizilerin sonuna 0 eklenir
        bert_model_name="bert-base-multilingual-cased",
        bert_pooling="cls",
        bert_batch_size=32,
    )

    output = preprocessor.fit_transform()

    print(f"Oturum sayısı          : {output['numSessions']}")
    print(f"Max oturum uzunluğu    : {output['maxSessionLength']}")
    print(f"Tag vocab boyutu       : {len(output['tag_vocab'])}")
    print(f"Kategorik alan sayısı  : {len(output['vocabs'])}")
    print(f"Embedding boyutu       : {output['embeddingDimension']}")
    print(f"\nİlk oturum event_type  : {output['eventSequences']['event_type'][0]}")
    print(f"İlk oturum tag[0]      : {output['tagSequences'][0][0]}")
    print(f"Session feature keys   : {list(output['sessionFeatures'][0].keys())}")
