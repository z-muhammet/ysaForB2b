#!/usr/bin/env python3
"""
Session Cleaner
---------------
Girdi : sessions.jsonl
Cikti : sessions_clean.jsonl

Yapilan temizlikler:
  [KVKK-1] MEMBERSHIP_OTHER  — Silver Plus tier firma isimleri anonymize edilmemisti, duzeltildi
  [KVKK-2] REGISTER_OTHER    — Tedarikci onay/dogrulama mesajlarindaki firma isimleri anonymize edildi
  [NLP-3]  is_automated=True eventleri session'lardan cikarıldi
           (MEMBERSHIP_EXPIRY_NOTIFICATION, SYSTEM_*, SURVEY_*, REMIND_RFX_SENT)
  [NLP-4]  View/passive eventlerdeki arka arkaya tekrarlar (consecutive duplicate) kaldirildi
           BID, ORDER, CART, PURCHASE gibi gercek kullanici aksiyonlarina dokunulmadi
  [NLP-5]  Sadece automated eventlerden olusan session'lar silindi
           (Kullanici etkilesimi olmayan, tamamen sistem uretimi)
"""

import json
import os
import re
import sys
import hashlib
from datetime import datetime

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INPUT_FILE  = "sessions.jsonl"
OUTPUT_FILE = "sessions_clean.jsonl"
KVKK_MAP    = "entity_map.json"       # Var olan haritaya ek girisleri yazacagiz

# ── Deterministik anonymization (tag_and_anonymize.py ile ayni mantik) ────────
_entity_cache: dict = {}

def _anon(entity_type: str, raw_name: str) -> str:
    name = raw_name.strip()
    if not name:
        return f"{entity_type}_UNKNOWN"
    key = f"{entity_type}:{name.upper()}"
    if key not in _entity_cache:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8].upper()
        _entity_cache[key] = f"{entity_type}_{h}"
    return _entity_cache[key]

def anon_company(name: str) -> str:
    return _anon("COMPANY", name)


# ── Var olan entity_map.json'u yukle (tutarlilik icin) ────────────────────────
def load_existing_map() -> dict:
    try:
        with open(KVKK_MAP, "r", encoding="utf-8") as f:
            existing = json.load(f)
        # Reverse map'i cache'e yukle: anon_id → original
        # Biz cache'i key=(TYPE:NAME) → anon_id olarak tutuyoruz, reverse lazim degil
        # Sadece bilgi amacli yukluyoruz
        return existing
    except FileNotFoundError:
        return {}


# ── [KVKK-1] MEMBERSHIP_OTHER duzeltici ──────────────────────────────────────
# Pattern: "EMAIL_XXXXXXXX kullanıcısı [FIRMA ADI] firmasının paketini Silver Plus olarak değiştirdi."
RE_MEM_OTHER = re.compile(
    r'^(EMAIL_[A-F0-9]+(?:\.\w+)?)\s+kullanıcısı\s+(.+?)\s+firmasının paketini\s+(.+?)\s+olarak değiştirdi',
    re.IGNORECASE
)

def fix_membership_other(event: dict) -> dict:
    norm = event.get("message_normalized", "")
    mo = RE_MEM_OTHER.match(norm)
    if not mo:
        return event  # Dokunamadik, oldugu gibi birak

    email_anon   = mo.group(1)            # Zaten anonymize
    company_raw  = mo.group(2).strip()
    tier         = mo.group(3).strip()
    company_anon = anon_company(company_raw)

    new_norm = (
        f"{email_anon} kullanıcısı {company_anon} firmasının paketini "
        f"{tier} olarak değiştirdi."
    )

    event = dict(event)
    event["message_normalized"] = new_norm
    event["message_template"]   = "MEMBERSHIP_PACKAGE_CHANGED_BY_ADMIN"
    event["action_type"]        = "package_changed_by_admin"
    event["membership_tier"]    = tier

    entities = dict(event.get("named_entities") or {})
    entities["company"]      = company_anon
    entities["admin_email"]  = email_anon
    event["named_entities"]  = entities

    # Tag guncelle
    tags = list(event.get("tags") or [])
    if "membership" not in tags:
        tags.append("membership")
    tier_tag = f"tier_{tier.lower().replace(' ', '_')}"
    if tier_tag not in tags:
        tags.append(tier_tag)
    if "admin_action" not in tags:
        tags.append("admin_action")
    event["tags"] = tags

    return event


# ── [KVKK-2] REGISTER_OTHER duzeltici ────────────────────────────────────────
# Pattern A: "[FIRMA] tedarikçi doğrulama yaptı."
# Pattern B: "[FIRMA] tedarikçinin başvurusunu onayladı."
RE_REG_VERIFY  = re.compile(r'^(.+?)\s+tedarikçi doğrulama yaptı', re.IGNORECASE)
RE_REG_APPROVE = re.compile(r'^(.+?)\s+tedarikçinin başvurusunu onayladı', re.IGNORECASE)

def fix_register_other(event: dict) -> dict:
    norm = event.get("message_normalized", "")

    mo = RE_REG_VERIFY.match(norm)
    if mo:
        company_raw  = mo.group(1).strip()
        company_anon = anon_company(company_raw)
        event = dict(event)
        event["message_normalized"] = f"{company_anon} tedarikçi doğrulama yaptı."
        event["message_template"]   = "REGISTER_SUPPLIER_VERIFIED"
        event["action_type"]        = "supplier_verified"
        event["named_entities"]     = {"company": company_anon}
        tags = list(event.get("tags") or [])
        if "supplier_verification" not in tags:
            tags.append("supplier_verification")
        event["tags"] = tags
        return event

    mo = RE_REG_APPROVE.match(norm)
    if mo:
        company_raw  = mo.group(1).strip()
        company_anon = anon_company(company_raw)
        event = dict(event)
        event["message_normalized"] = f"{company_anon} tedarikçinin başvurusu onaylandı."
        event["message_template"]   = "REGISTER_SUPPLIER_APPROVED"
        event["action_type"]        = "supplier_registration_approved"
        event["named_entities"]     = {"company": company_anon}
        return event

    return event  # Dokunamadigimiz kalan kayitlar


# ── [NLP-3] Automated event filter ───────────────────────────────────────────
# Bu template'ler kullanicinin yapmadigi, sistemin urettigi olaylar
AUTOMATED_TEMPLATES = {
    "MEMBERSHIP_EXPIRY_NOTIFICATION",
    "SYSTEM_NEVER_LOGIN",
    "SYSTEM_EMAIL_UNSUBSCRIBE",
    "SURVEY_SUPPLIER_EVALUATION_SENT",
    "REMIND_RFX_SENT",
}


# ── [NLP-4] Consecutive duplicate filter ─────────────────────────────────────
# Sadece su "pasif/view" template'lerde arka arkaya tekrar temizlenir.
# BID, ORDER, CART, PURCHASE ve diger gercek aksiyonlara DOKUNULMAZ.
DEDUP_TEMPLATES = {
    "RFX_VIEWED",
    "PAYMENT_INSUFFICIENT_WARNING",        # dogru isim
    "PAYMENT_INSUFFICIENT_PACKAGE_WARNING", # tag_and_anonymize'daki isim
    "PAYMENT_UPGRADE_CLICKED",
    "PAYMENT_INFO",
    "PAYMENT_ATTEMPT",
    "RESET_LINK_CLICKED",
    "RESET_REQUESTED",
    "RFX_OTHER",
    "MEMBERSHIP_OTHER",                    # duzeltildikten sonra kalan fallback'ler
    "LOGIN_UNKNOWN",
}

# Asla dokunulmayacak template'ler (referans olarak dokumante edildi)
_PROTECTED_TEMPLATES = {
    "BID_SUBMITTED", "BID_SUBMITTED_TO_BUYER", "BID_SUBMITTED_GENERIC",
    "BID_REVISED", "BID_DELETED",
    "ORDER_CREATED",
    "PAYMENT_CART_ADDED",
    "MEMBERSHIP_PURCHASED", "PAYMENT_DIRECT_PURCHASE", "PAYMENT_MEMBERSHIP_PURCHASED",
    "PASSWORD_CHANGED_VIA_LINK", "PASSWORD_CHANGED_PROFILE",
    "RFX_CREATED",
    "REGISTER_APPROVED", "REGISTER_SUPPLIER_VERIFIED", "REGISTER_SUPPLIER_APPROVED",
    "MEMBERSHIP_PACKAGE_CHANGED_BY_ADMIN",
    "UNSUBSCRIBE_EMAIL",
}


def deduplicate_consecutive(events: list) -> list:
    """
    DEDUP_TEMPLATES icin arka arkaya gelen ayni template'i tek kayda indirir.
    Korunan template'lere hic dokunmaz.
    """
    if not events:
        return events

    result = [events[0]]
    for ev in events[1:]:
        tmpl = ev.get("message_template", "")
        prev = result[-1].get("message_template", "")
        if tmpl == prev and tmpl in DEDUP_TEMPLATES:
            continue  # tekrar — atla
        result.append(ev)

    return result


# ── Session summary yeniden hesapla ──────────────────────────────────────────
def recompute_summary(events: list, original_summary: dict) -> dict:
    if not events:
        return {}

    categories   = list({e.get("event_category") for e in events if e.get("event_category")})
    action_types = list({e.get("action_type")     for e in events if e.get("action_type")})
    all_tags     = []
    for e in events:
        all_tags.extend(e.get("tags") or [])
    unique_tags = list(dict.fromkeys(all_tags))

    outcomes     = [e.get("outcome") for e in events if e.get("outcome")]
    last_ts      = events[-1].get("created_date") if events else None

    result = {
        "event_count"     : len(events),
        "categories"      : categories,
        "action_types"    : action_types,
        "unique_tags"     : unique_tags,
        "has_failure"     : "failure" in outcomes,
        "has_purchase"    : any("membership_purchase" in (e.get("tags") or []) for e in events),
        "has_bid"         : any(e.get("event_type") == "bid"   for e in events),
        "has_rfx"         : any(e.get("event_type") == "rfx"   for e in events),
        "has_order"       : any(e.get("event_type") == "order"  for e in events),
        "session_end_time": last_ts,
    }

    if last_ts and events:
        try:
            start = events[0].get("created_date", "")
            if start:
                delta = datetime.fromisoformat(last_ts) - datetime.fromisoformat(start)
                result["duration_minutes"] = round(delta.total_seconds() / 60, 1)
        except Exception:
            if "duration_minutes" in original_summary:
                result["duration_minutes"] = original_summary["duration_minutes"]

    return result


# ── seq numaralarini sifirla ──────────────────────────────────────────────────
def reindex_seq(events: list) -> list:
    return [{**ev, "seq": i + 1} for i, ev in enumerate(events)]


# ── Ana islem ─────────────────────────────────────────────────────────────────
def main():
    load_existing_map()  # Var olan entity_map'i cache'e al

    stats = {
        "in_sessions"        : 0,
        "kvkk1_fixed"        : 0,
        "kvkk2_fixed"        : 0,
        "automated_removed"  : 0,
        "consecutive_removed": 0,
        "auto_only_dropped"  : 0,
        "empty_after_clean"  : 0,
        "out_sessions"       : 0,
        "out_events"         : 0,
    }

    print(f"Okuyorum: {INPUT_FILE}")
    output_sessions = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            session = json.loads(line)
            stats["in_sessions"] += 1
            events = session.get("sequentialEvents", [])

            # ── [KVKK-1 & KVKK-2] Firma ismi duzeltme ────────────────────────
            fixed_events = []
            for ev in events:
                tmpl = ev.get("message_template", "")
                if tmpl == "MEMBERSHIP_OTHER":
                    ev = fix_membership_other(ev)
                    stats["kvkk1_fixed"] += 1
                elif tmpl == "REGISTER_OTHER":
                    ev = fix_register_other(ev)
                    stats["kvkk2_fixed"] += 1
                fixed_events.append(ev)

            # ── [NLP-3] Automated eventleri cikar ────────────────────────────
            before_auto = len(fixed_events)
            user_events = [
                ev for ev in fixed_events
                if ev.get("message_template") not in AUTOMATED_TEMPLATES
            ]
            stats["automated_removed"] += before_auto - len(user_events)

            # ── [NLP-5] Sadece automated icerikli session kontrolu ─────────────
            # Eger TUM eventler automated ise (hepsi silindi ve liste bosti)
            if not user_events:
                # Orijinal eventlerin tumu automated miydi?
                if all(ev.get("message_template") in AUTOMATED_TEMPLATES for ev in events):
                    stats["auto_only_dropped"] += 1
                else:
                    stats["empty_after_clean"] += 1
                continue

            # ── [NLP-4] Consecutive duplicate temizle ─────────────────────────
            before_dedup = len(user_events)
            deduped = deduplicate_consecutive(user_events)
            stats["consecutive_removed"] += before_dedup - len(deduped)

            # ── Temizleme sonrasi bos session kontrolu ────────────────────────
            if not deduped:
                stats["empty_after_clean"] += 1
                continue

            # ── Seq numaralari sifirla ve summary yeniden hesapla ─────────────
            final_events = reindex_seq(deduped)
            new_summary  = recompute_summary(final_events, session.get("summary", {}))

            clean_session = {
                "userId"          : session["userId"],
                "sessionId"       : session["sessionId"],
                "sessionStartTime": session["sessionStartTime"],
                "summary"         : new_summary,
                "sequentialEvents": final_events,
            }
            output_sessions.append(clean_session)
            stats["out_sessions"] += 1
            stats["out_events"]   += len(final_events)

    # ── Yaz ──────────────────────────────────────────────────────────────────
    print(f"Yaziliyor: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for s in output_sessions:
            fout.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ── Yeni entity'leri entity_map.json'a ekle ───────────────────────────────
    try:
        with open(KVKK_MAP, "r", encoding="utf-8") as f:
            existing_map = json.load(f)
    except FileNotFoundError:
        existing_map = {}

    new_entries = 0
    for anon_id, original in _entity_cache.items():
        # cache key = "COMPANY:NAME", value = "COMPANY_XXXXXXXX"
        # reverse: anon_id value → original name
        pass

    # _reverse_map yerine cache'den reverse map olustur
    reverse = {v: k.split(":", 1)[1] for k, v in _entity_cache.items()}
    for anon_id, orig in reverse.items():
        if anon_id not in existing_map:
            existing_map[anon_id] = orig
            new_entries += 1

    with open(KVKK_MAP, "w", encoding="utf-8") as f:
        json.dump(existing_map, f, ensure_ascii=False, indent=2)

    # ── Girdi dosyasini sil ───────────────────────────────────────────────────
    if os.path.exists(INPUT_FILE):
        os.remove(INPUT_FILE)
        print(f"  Silindi: {INPUT_FILE}")

    # ── Rapor ─────────────────────────────────────────────────────────────────
    avg_ev = stats["out_events"] / stats["out_sessions"] if stats["out_sessions"] else 0

    print(f"""
── Temizlik Raporu ─────────────────────────────────────────
  Giris session           : {stats["in_sessions"]:>10,}

  [KVKK-1] Membership Other duzeltilen  : {stats["kvkk1_fixed"]:>7,}  event
  [KVKK-2] Register Other duzeltilen    : {stats["kvkk2_fixed"]:>7,}  event
  [NLP-3]  Automated event kaldirildi   : {stats["automated_removed"]:>7,}  event
  [NLP-4]  Consecutive duplicate kaldır : {stats["consecutive_removed"]:>7,}  event
  [NLP-5]  Auto-only session silindi    : {stats["auto_only_dropped"]:>7,}  session
           Temizlik sonrasi bos session  : {stats["empty_after_clean"]:>7,}  session

  Cikis session           : {stats["out_sessions"]:>10,}
  Cikis event (toplam)    : {stats["out_events"]:>10,}
  Ort. event / session    : {avg_ev:>12.1f}
  Entity map yeni ek      : {new_entries:>10,}  (entity_map.json guncellendi)
  Cikti                   : {OUTPUT_FILE}
────────────────────────────────────────────────────────────""")

    # Ornek temizlenmis session goster
    if output_sessions:
        sample = next(
            (s for s in output_sessions if s["summary"]["event_count"] >= 3),
            output_sessions[0]
        )
        print("\n── Ornek Temiz Session ─────────────────────────────────")
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
