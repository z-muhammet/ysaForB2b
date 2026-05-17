#!/usr/bin/env python3
"""
Session Builder
---------------
Girdi : tagged_events.jsonl
Cikti : sessions.jsonl

Ne yapar:
  1. Anlamsiz login fail eventlerini tamamen siler.
  2. Kullanici basarili giris yaptiginda yeni session baslatir.
  3. Bir sonraki basarili girise kadar olan tum eventleri o session'a atar.
  4. Session icinde hic event yoksa (sadece giris yapip ciktiysa) o session yazilmaz.
"""

import json
import os
import sys
import hashlib
from collections import defaultdict
from datetime import datetime

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

INPUT_FILE  = "tagged_events.jsonl"
OUTPUT_FILE = "sessions.jsonl"

# ── Tamamen silinecek template'ler ────────────────────────────────────────────
REMOVE_TEMPLATES = {
    "LOGIN_FAIL_WRONG_CREDENTIALS",
    "LOGIN_FAIL_RECAPTCHA",
    "LOGIN_FAIL_SYSTEM_ERROR",
}

# ── Yeni session baslatan template'ler ───────────────────────────────────────
SESSION_STARTERS = {
    "LOGIN_SUCCESS_WEB",
    "LOGIN_SUCCESS_MOBILE",
}

# ── Sequential event'te tutulacak alanlar ────────────────────────────────────
SEQ_FIELDS = [
    "event_type",
    "event_category",
    "action_type",
    "message_template",
    "message_normalized",
    "tags",
    "named_entities",
    "outcome",
    "sentiment",
    "actor_role",
    "platform",
    "urgency_level",
    "user_journey_stage",
    "membership_tier",
    "error_type",
    "is_automated",
    "created_date",
    "hour_of_day",
    "day_of_week",
    "time_of_day",
    "is_weekend",
]


# ── Yardimci fonksiyonlar ─────────────────────────────────────────────────────

def make_session_id(user_id: str, timestamp: str) -> str:
    """Deterministik session ID — ayni giris her zaman ayni ID'yi uretir."""
    raw = f"{user_id}:{timestamp}"
    return "sess_" + hashlib.sha256(raw.encode()).hexdigest()[:8].upper()


def make_session_start_time(timestamp_str: str, platform: str) -> dict:
    """
    NLP motorunun anlayacagi formatta session baslangic bilgisi.
    Zaman ozellikleri + platform tek bir objede toplanir.
    """
    try:
        dt   = datetime.fromisoformat(timestamp_str)
        hour = dt.hour
        tod  = (
            "night"     if hour < 6  else
            "morning"   if hour < 12 else
            "afternoon" if hour < 18 else
            "evening"
        )
        return {
            "timestamp"  : timestamp_str,
            "year"       : dt.year,
            "month"      : dt.month,
            "day_of_week": dt.strftime("%A"),
            "hour_of_day": hour,
            "time_of_day": tod,
            "is_weekend" : dt.weekday() >= 5,
            "platform"   : platform,
        }
    except Exception:
        return {"timestamp": timestamp_str, "platform": platform}


def to_seq_event(record: dict, seq_num: int) -> dict:
    """Bir tagli kaydi temiz sequential event formatina donusturur."""
    item = {"seq": seq_num}
    for field in SEQ_FIELDS:
        val = record.get(field)
        # Null / bos degerleri yazma — gereksiz gurultu azalt
        if val is None or val == [] or val == {}:
            continue
        item[field] = val
    return item


# ── Ozet istatistik hesaplama ─────────────────────────────────────────────────

def session_summary(events: list) -> dict:
    """Session icindeki eventlerin ozet istatistiklerini cikarir."""
    categories   = list({e.get("event_category") for e in events if e.get("event_category")})
    action_types = list({e.get("action_type")     for e in events if e.get("action_type")})
    all_tags     = []
    for e in events:
        all_tags.extend(e.get("tags") or [])
    unique_tags  = list(dict.fromkeys(all_tags))  # siralama korunarak unique

    outcomes     = [e.get("outcome") for e in events if e.get("outcome")]
    has_failure  = "failure" in outcomes
    has_purchase = any("membership_purchase" in (e.get("tags") or []) for e in events)
    has_bid      = any(e.get("event_type") == "bid"   for e in events)
    has_rfx      = any(e.get("event_type") == "rfx"   for e in events)
    has_order    = any(e.get("event_type") == "order"  for e in events)

    # Session sonu zaman damgasi
    last_ts = events[-1].get("created_date") if events else None

    result = {
        "event_count"     : len(events),
        "categories"      : categories,
        "action_types"    : action_types,
        "unique_tags"     : unique_tags,
        "has_failure"     : has_failure,
        "has_purchase"    : has_purchase,
        "has_bid"         : has_bid,
        "has_rfx"         : has_rfx,
        "has_order"       : has_order,
        "session_end_time": last_ts,
    }

    # Session suresi (dakika) — hesaplanabiliyorsa ekle
    if last_ts:
        try:
            start_ts_str = events[0].get("created_date")
            if start_ts_str:
                delta = datetime.fromisoformat(last_ts) - datetime.fromisoformat(start_ts_str)
                result["duration_minutes"] = round(delta.total_seconds() / 60, 1)
        except Exception:
            pass

    return result


# ── Ana islem akisi ───────────────────────────────────────────────────────────

def main():
    # 1. Oku ve filtrele ───────────────────────────────────────────────────────
    print("Okuyorum ve anlamsiz eventler filtreleniyor...")
    records       = []
    removed_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r        = json.loads(line)
                template = r.get("message_template", "")
                if template in REMOVE_TEMPLATES:
                    removed_count += 1
                    continue
                records.append(r)
            except Exception:
                pass

    total_in = len(records) + removed_count
    print(f"  Toplam kayit (giris)     : {total_in:>10,}")
    print(f"  Silinen login fail kayit : {removed_count:>10,}")
    print(f"  Isleme alınan kayit      : {len(records):>10,}")

    # 2. Kullanici bazinda grupla, tarihe gore sirala ─────────────────────────
    print("\nKullanici bazinda gruplaniyor ve sirаlaniyor...")
    by_user: dict = defaultdict(list)
    for r in records:
        by_user[r["user_id"]].append(r)

    for uid in by_user:
        by_user[uid].sort(key=lambda x: x.get("created_date", ""))

    print(f"  Unique kullanici         : {len(by_user):>10,}")

    # 3. Session'lari olustur ─────────────────────────────────────────────────
    print("\nSession'lar olusturuluyor...")
    sessions          = []
    skipped_empty     = 0   # Giris yapip hic islem yapilmayan session
    skipped_no_login  = 0   # Login oncesi eventler (session yok)

    for uid, events in by_user.items():
        login_event     = None   # aktif session'in giris eventi
        session_events  = []     # giris sonrasi toplanan eventler

        for ev in events:
            template = ev.get("message_template", "")

            if template in SESSION_STARTERS:
                # Onceki session'i kapat
                if login_event is not None:
                    if session_events:
                        sessions.append(_build_session(uid, login_event, session_events))
                    else:
                        skipped_empty += 1

                # Yeni session baslat
                login_event    = ev
                session_events = []

            elif login_event is not None:
                # Aktif session'a event ekle
                session_events.append(ev)

            else:
                # Henuz login olmamis kullanicinin eventleri — session yok
                skipped_no_login += 1

        # Dosya sonunda son acik session'i kapat
        if login_event is not None:
            if session_events:
                sessions.append(_build_session(uid, login_event, session_events))
            else:
                skipped_empty += 1

    # 4. Yaz ─────────────────────────────────────────────────────────────────
    print(f"\nYaziliyor: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for s in sessions:
            fout.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Girdi dosyasini sil ─────────────────────────────────────────────────────
    if os.path.exists(INPUT_FILE):
        os.remove(INPUT_FILE)
        print(f"  Silindi: {INPUT_FILE}")

    # 5. Rapor ────────────────────────────────────────────────────────────────
    event_counts = [s["summary"]["event_count"] for s in sessions]
    total_events = sum(event_counts)
    avg_events   = total_events / len(sessions) if sessions else 0
    max_events   = max(event_counts) if event_counts else 0

    print(f"\n── Sonuc ──────────────────────────────────────────")
    print(f"  Yazilan session          : {len(sessions):>10,}")
    print(f"  Atlanan (bos session)    : {skipped_empty:>10,}  (giris yapip islem yok)")
    print(f"  Atlanan (login oncesi)   : {skipped_no_login:>10,}  (session atanamayan event)")
    print(f"  Toplam sekansli event    : {total_events:>10,}")
    print(f"  Ort. event / session     : {avg_events:>12.1f}")
    print(f"  Max event / session      : {max_events:>10,}")
    print(f"  Cikti dosyasi            : {OUTPUT_FILE}")

    # Ornek bir session yazdir
    if sessions:
        print(f"\n── Ornek Session ──────────────────────────────────")
        sample = next((s for s in sessions if s["summary"]["event_count"] >= 3), sessions[0])
        print(json.dumps(sample, ensure_ascii=False, indent=2))


def _build_session(uid: str, login_event: dict, session_events: list) -> dict:
    """Session nesnesini olusturur."""
    platform = login_event.get("platform", "web")
    start_ts = login_event.get("created_date", "")
    summary  = session_summary(session_events)

    return {
        "userId"          : uid,
        "sessionId"       : make_session_id(uid, start_ts),
        "sessionStartTime": make_session_start_time(start_ts, platform),
        "summary"         : summary,
        "sequentialEvents": [
            to_seq_event(ev, i + 1)
            for i, ev in enumerate(session_events)
        ],
    }


if __name__ == "__main__":
    main()
