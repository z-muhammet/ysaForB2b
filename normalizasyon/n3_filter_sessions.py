#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  NLP VERİ HAZIRLAMA PİPELINE  —  3. Normalizasyon              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  n1_events_tagged.jsonl      ← Seviye 1                         ║
║    Ham event'ler, KVKK anonimizasyonu, NLP tag'leri             ║
║    Script: tag_and_anonymize.py                                  ║
║                                                                  ║
║  n2_sessions_clean.jsonl     ← Seviye 2                         ║
║    Login bazlı session gruplandırması                            ║
║    Otomatik eventler temizlendi, consecutive dedup yapıldı       ║
║    KVKK açıkları kapatıldı (Silver Plus, Register Other)        ║
║    Script: build_sessions.py + clean_sessions.py                 ║
║                                                                  ║
║  n3_sessions_model_ready.jsonl  ← Seviye 3  (bu script)        ║
║    Tek eventli session'lar silindi                               ║
║    2-eventli session'larda pasif/anlamsız combolar silindi       ║
║    NLP / ML modeline doğrudan verilebilir                        ║
║    Script: n3_filter_sessions.py  (bu dosya)                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json
import os
import sys
from collections import Counter

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Dosya isimleri ────────────────────────────────────────────────────────────
INPUT_FILE  = "sessions_clean.jsonl"     # n2 çıktısı
OUTPUT_FILE = "n3_sessions_model_ready.jsonl"

# n3 tamamlandıktan sonra silinecek ara dosya
INTERMEDIATE_INPUT = "sessions_clean.jsonl"


# ── Pasif template'ler: bu ikili kombinasyonlar 2-event session'da anlamsız ──
# Sadece bu template'lerden oluşan 2-event session'lar silinir.
# (3+ eventli session'lara bu kural UYGULANMAZ)
PASSIVE_TEMPLATES = {
    "RFX_VIEWED",
    "PAYMENT_INSUFFICIENT_PACKAGE_WARNING",
    "PAYMENT_UPGRADE_CLICKED",
}


def is_meaningful_2event(session: dict) -> bool:
    """
    2-event session'ın anlamlı olup olmadığını belirler.
    Her iki event de pasif ise False döner (silinir).
    En az biri gerçek kullanıcı aksiyonu ise True döner (korunur).

    Korunan örnekler:
      INSUFFICIENT → UNSUBSCRIBE     (churn sinyali, çok değerli)
      RFX_VIEWED   → BID_SUBMITTED   (gördü, teklif verdi)
      RESET_REQUESTED → RESET_LINK   (şifre sıfırlama akışı)
      BID_REVISED  → RFX_VIEWED      (bid aktivitesi)
      CART_ADDED   → PAYMENT_ATTEMPT (ödeme hunisi)

    Silinen örnekler:
      INSUFFICIENT → UPGRADE_CLICKED  (tıklayıp çıktı, işlem yok)
      INSUFFICIENT → RFX_VIEWED       (uyarı görüp gezmeye devam)
      RFX_VIEWED   → INSUFFICIENT     (gezdi, uyarı aldı, gitti)
    """
    events = session.get("sequentialEvents", [])
    templates = [ev.get("message_template", "") for ev in events]
    return not all(t in PASSIVE_TEMPLATES for t in templates)


def main():
    print(f"─ Okuyorum: {INPUT_FILE} ─────────────────────────────")

    sessions = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))

    total_in = len(sessions)
    print(f"  Giriş session : {total_in:,}")

    # ── Filtreleme ────────────────────────────────────────────────────────────
    removed_single   = 0  # tek eventli
    removed_passive  = 0  # 2-eventli pasif
    kept = []

    for s in sessions:
        ec = s["summary"]["event_count"]

        # Kural 1: Tek eventli → direkt sil
        if ec == 1:
            removed_single += 1
            continue

        # Kural 2: 2-eventli → pasif ikili mi kontrol et
        if ec == 2 and not is_meaningful_2event(s):
            removed_passive += 1
            continue

        # 3+ eventli → hepsini koru
        kept.append(s)

    # ── Yaz ──────────────────────────────────────────────────────────────────
    print(f"\n─ Yazıyorum: {OUTPUT_FILE} ──────────────────────────────")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for s in kept:
            fout.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Girdi dosyasını sil ─────────────────────────────────────────────────────
    if os.path.exists(INPUT_FILE):
        os.remove(INPUT_FILE)
        print(f"  Silindi: {INPUT_FILE}")

    # ── İstatistik ────────────────────────────────────────────────────────────
    ec_dist = Counter(s["summary"]["event_count"] for s in kept)
    total_events = sum(s["summary"]["event_count"] for s in kept)
    avg_events   = total_events / len(kept) if kept else 0

    purchase_count = sum(1 for s in kept if s["summary"].get("has_purchase"))
    bid_count      = sum(1 for s in kept if s["summary"].get("has_bid"))
    order_count    = sum(1 for s in kept if s["summary"].get("has_order"))
    rfx_count      = sum(1 for s in kept if s["summary"].get("has_rfx"))
    failure_count  = sum(1 for s in kept if s["summary"].get("has_failure"))

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  3. Normalizasyon Raporu                                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Giriş session (n2)           : {total_in:>8,}                        ║
║                                                                  ║
║  [SİLİNDİ] Tek-eventli        : {removed_single:>8,}  (anlamsız, seq yok)   ║
║  [SİLİNDİ] 2-event pasif      : {removed_passive:>8,}  (her ikisi pasif)    ║
║                                                                  ║
║  Çıkış session (n3)           : {len(kept):>8,}                        ║
║  Toplam event                 : {total_events:>8,}                        ║
║  Ort. event / session         : {avg_events:>10.1f}                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Session içerik dağılımı:                                        ║
║    Satın alma içeren           : {purchase_count:>8,}                        ║
║    Bid içeren                  : {bid_count:>8,}                        ║
║    Sipariş içeren              : {order_count:>8,}                        ║
║    RFX içeren                  : {rfx_count:>8,}                        ║
║    Hata içeren                 : {failure_count:>8,}                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Event sayısı dağılımı (ilk 10):""")

    for k in sorted(ec_dist.keys())[:10]:
        bar = "█" * min(int(ec_dist[k] / max(ec_dist.values()) * 20), 20)
        print(f"║    {k:>3} event: {ec_dist[k]:>6,}  {bar}")

    print(f"""╠══════════════════════════════════════════════════════════════════╣
║  Dosya Hiyerarşisi:                                              ║
║    n1_events_tagged.jsonl        ← Seviye 1 (ham tag + anon)    ║
║    n2_sessions_clean.jsonl       ← Seviye 2 (session + temizlik)║
║    n3_sessions_model_ready.jsonl ← Seviye 3 (ML'e hazır)        ║
╚══════════════════════════════════════════════════════════════════╝""")


if __name__ == "__main__":
    main()
