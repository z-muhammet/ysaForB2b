#!/usr/bin/env python3
"""
KVKK-uyumlu anonimizasyon + NLP tagging scripti.
Girdi : processed_user_events.json  (JSONL formatı)
Çıktı : tagged_events.jsonl          (anonimleştirilmiş + taglenmiş)
        entity_map.json              (orijinal → anon ID haritası, güvenli saklayın)

Anonimizasyon: SHA-256 tabanlı deterministik — aynı firma her zaman aynı COMPANY_XXXXXXXX ID'sini alır.
"""

import json
import re
import hashlib
import sys
import os
from datetime import datetime

# Windows terminal encoding fix
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Konfigürasyon ─────────────────────────────────────────────────────────────
INPUT_FILE  = "processed_user_events.json"
OUTPUT_FILE = "tagged_events.jsonl"
MAP_FILE    = "entity_map.json"  # Gizli tutun! (KVKK)
CHUNK_SIZE  = 5000

# ── Deterministik Anonimizasyon ───────────────────────────────────────────────
_entity_cache: dict = {}          # (type, normalized_name) → anon_id
_reverse_map:  dict = {}          # anon_id → original_name  (entity_map.json için)

def _anon(entity_type: str, raw_name: str) -> str:
    """Aynı isim her zaman aynı ENTITY_TYPE_XXXXXXXX döner."""
    name = raw_name.strip()
    if not name:
        return f"{entity_type}_UNKNOWN"
    key = f"{entity_type}:{name.upper()}"
    if key not in _entity_cache:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8].upper()
        anon_id = f"{entity_type}_{h}"
        _entity_cache[key] = anon_id
        _reverse_map[anon_id] = raw_name  # haritaya kaydet
    return _entity_cache[key]

def anon_company(name: str) -> str: return _anon("COMPANY", name)
def anon_email(email: str)  -> str: return _anon("EMAIL",   email)
def anon_rfx(name: str)     -> str: return _anon("RFX",     name)   # RFX adı da potansiyel PII

# ── Mesaj Parserleri ──────────────────────────────────────────────────────────

def parse_login(msg: str) -> dict:
    m = msg.strip()
    base = {"event_category": "auth", "actor_role": "user", "is_automated": False}
    if "Mobile" in m:
        return {**base, "action_type": "login_success", "outcome": "success",
                "platform": "mobile", "sentiment": "positive", "error_type": None,
                "message_template": "LOGIN_SUCCESS_MOBILE",
                "message_normalized": "Kullanıcı mobil uygulamadan başarıyla giriş yaptı.",
                "tags": ["login", "success", "mobile", "authentication"]}
    if "successfuly" in m or "successfully" in m:
        return {**base, "action_type": "login_success", "outcome": "success",
                "platform": "web", "sentiment": "positive", "error_type": None,
                "message_template": "LOGIN_SUCCESS_WEB",
                "message_normalized": "Kullanıcı web üzerinden başarıyla giriş yaptı.",
                "tags": ["login", "success", "web", "authentication"]}
    if "şifre hatalı" in m.lower() or "kullanıcı adı" in m.lower():
        return {**base, "action_type": "login_fail", "outcome": "failure",
                "platform": "web", "sentiment": "negative", "error_type": "wrong_credentials",
                "message_template": "LOGIN_FAIL_WRONG_CREDENTIALS",
                "message_normalized": "Kullanıcı adı veya şifre hatalı girildi.",
                "tags": ["login", "failure", "wrong_credentials", "authentication"]}
    if "recaptcha" in m.lower():
        return {**base, "action_type": "login_fail", "outcome": "failure",
                "platform": "web", "sentiment": "negative", "error_type": "recaptcha",
                "message_template": "LOGIN_FAIL_RECAPTCHA",
                "message_normalized": "Giriş denemesi recaptcha hatası ile engellendi.",
                "tags": ["login", "failure", "recaptcha", "authentication", "bot_protection"]}
    if "error" in m.lower() or "hata" in m.lower():
        return {**base, "action_type": "login_fail", "outcome": "failure",
                "platform": "web", "sentiment": "negative", "error_type": "system_error",
                "message_template": "LOGIN_FAIL_SYSTEM_ERROR",
                "message_normalized": "Giriş denemesi sistem hatası ile başarısız oldu.",
                "tags": ["login", "failure", "system_error", "authentication"]}
    return {**base, "action_type": "login_unknown", "outcome": "unknown",
            "platform": "unknown", "sentiment": "neutral", "error_type": None,
            "message_template": "LOGIN_UNKNOWN",
            "message_normalized": m,
            "tags": ["login", "unknown"]}


def parse_reset_password(msg: str) -> dict:
    m = msg.strip()
    base = {"event_category": "auth", "actor_role": "user", "is_automated": False}
    if "linkine tıkladı" in m.lower():
        return {**base, "action_type": "reset_link_clicked", "outcome": "neutral",
                "platform": "email", "sentiment": "neutral", "error_type": None,
                "message_template": "RESET_LINK_CLICKED",
                "message_normalized": "Şifre sıfırlama bağlantısına tıklandı.",
                "tags": ["reset_password", "link_clicked", "email"]}
    if "talebi" in m.lower() or ("sıfırlama" in m.lower() and "tıkla" not in m.lower() and "değiştir" not in m.lower() and "hata" not in m.lower() and "karşılan" not in m.lower()):
        return {**base, "action_type": "reset_requested", "outcome": "neutral",
                "platform": "web", "sentiment": "neutral", "error_type": None,
                "message_template": "RESET_REQUESTED",
                "message_normalized": "Şifre sıfırlama talebinde bulunuldu.",
                "tags": ["reset_password", "request", "web"]}
    if "profilden" in m.lower() and "değiştirildi" in m.lower():
        return {**base, "action_type": "password_changed_profile", "outcome": "success",
                "platform": "web", "sentiment": "positive", "error_type": None,
                "message_template": "PASSWORD_CHANGED_PROFILE",
                "message_normalized": "Profil sayfasından şifre başarıyla değiştirildi.",
                "tags": ["reset_password", "success", "profile", "password_change"]}
    if "bağlantısı ile şifre değiştirdi" in m.lower() or "linki ile şifre değiştirdi" in m.lower():
        return {**base, "action_type": "password_changed_link", "outcome": "success",
                "platform": "email", "sentiment": "positive", "error_type": None,
                "message_template": "PASSWORD_CHANGED_VIA_LINK",
                "message_normalized": "Şifre sıfırlama bağlantısı aracılığıyla şifre başarıyla değiştirildi.",
                "tags": ["reset_password", "success", "email_link", "password_change"]}
    if "minimum" in m.lower() or "gereksini" in m.lower():
        return {**base, "action_type": "password_change_fail", "outcome": "failure",
                "platform": "web", "sentiment": "negative", "error_type": "weak_password",
                "message_template": "PASSWORD_FAIL_WEAK",
                "message_normalized": "Şifre minimum güvenlik gereksinimlerini karşılamadı.",
                "tags": ["reset_password", "failure", "weak_password", "validation"]}
    if "mevcut şifre" in m.lower() or "hatalı girdi" in m.lower():
        return {**base, "action_type": "password_change_fail", "outcome": "failure",
                "platform": "web", "sentiment": "negative", "error_type": "wrong_current_password",
                "message_template": "PASSWORD_FAIL_WRONG_CURRENT",
                "message_normalized": "Mevcut şifre hatalı girildi.",
                "tags": ["reset_password", "failure", "wrong_current_password"]}
    if "recaptcha" in m.lower():
        return {**base, "action_type": "reset_fail", "outcome": "failure",
                "platform": "web", "sentiment": "negative", "error_type": "recaptcha",
                "message_template": "RESET_FAIL_RECAPTCHA",
                "message_normalized": "Şifre sıfırlama denemesi recaptcha hatası ile engellendi.",
                "tags": ["reset_password", "failure", "recaptcha", "bot_protection"]}
    return {**base, "action_type": "reset_password_other", "outcome": "unknown",
            "platform": "web", "sentiment": "neutral", "error_type": None,
            "message_template": "RESET_PASSWORD_OTHER",
            "message_normalized": m,
            "tags": ["reset_password"]}


# Bid: "[SUPPLIER] tarafından [BUYER] firmasının talebine teklif verildi"
#       "[SUPPLIER] tarafından bir teklif verildi"
#       "Teklif Verildi / Revize Edildi / Silindi"
RE_BID_TO_BUYER   = re.compile(r'^(.+?)\s+tarafından\s+(.+?)\s+firmasının talebine teklif verildi$', re.I)
RE_BID_GENERIC    = re.compile(r'^(.+?)\s+tarafından bir teklif verildi$', re.I)

def parse_bid(msg: str) -> dict:
    m = msg.strip()
    base = {"event_category": "procurement", "is_automated": False}

    if m in ("Teklif Verildi",):
        return {**base, "action_type": "bid_submitted", "outcome": "success",
                "actor_role": "supplier", "platform": "web", "sentiment": "positive",
                "error_type": None, "named_entities": {},
                "message_template": "BID_SUBMITTED",
                "message_normalized": "Tedarikçi teklif verdi.",
                "tags": ["bid", "submitted", "procurement", "supplier_action"]}
    if m in ("Teklif Revize Edildi",):
        return {**base, "action_type": "bid_revised", "outcome": "success",
                "actor_role": "supplier", "platform": "web", "sentiment": "positive",
                "error_type": None, "named_entities": {},
                "message_template": "BID_REVISED",
                "message_normalized": "Tedarikçi teklifini revize etti.",
                "tags": ["bid", "revised", "procurement", "supplier_action"]}
    if m in ("Teklif Silindi",):
        return {**base, "action_type": "bid_deleted", "outcome": "neutral",
                "actor_role": "supplier", "platform": "web", "sentiment": "negative",
                "error_type": None, "named_entities": {},
                "message_template": "BID_DELETED",
                "message_normalized": "Teklif silindi.",
                "tags": ["bid", "deleted", "procurement", "supplier_action"]}

    mo = RE_BID_TO_BUYER.match(m)
    if mo:
        supplier_id = anon_company(mo.group(1))
        buyer_id    = anon_company(mo.group(2))
        return {**base, "action_type": "bid_submitted", "outcome": "success",
                "actor_role": "supplier", "platform": "web", "sentiment": "positive",
                "error_type": None,
                "named_entities": {"supplier_company": supplier_id, "buyer_company": buyer_id},
                "message_template": "BID_SUBMITTED_TO_BUYER",
                "message_normalized": f"{supplier_id} tarafından {buyer_id} firmasının talebine teklif verildi.",
                "tags": ["bid", "submitted", "procurement", "supplier_action", "has_buyer"]}

    mo = RE_BID_GENERIC.match(m)
    if mo:
        supplier_id = anon_company(mo.group(1))
        return {**base, "action_type": "bid_submitted", "outcome": "success",
                "actor_role": "supplier", "platform": "web", "sentiment": "positive",
                "error_type": None,
                "named_entities": {"supplier_company": supplier_id},
                "message_template": "BID_SUBMITTED_GENERIC",
                "message_normalized": f"{supplier_id} tarafından teklif verildi.",
                "tags": ["bid", "submitted", "procurement", "supplier_action"]}

    # Eski format: "X tarafından bir teklif verildi" (bazı eski kayıtlar)
    return {**base, "action_type": "bid_other", "outcome": "unknown",
            "actor_role": "supplier", "platform": "web", "sentiment": "neutral",
            "error_type": None, "named_entities": {},
            "message_template": "BID_OTHER",
            "message_normalized": _generic_anon(m),
            "tags": ["bid", "procurement"]}


# RFX: "Teklif Talebi Görüntülendi"
#       "[COMPANY] firması tarafından [RFX_NAME] adında bir teklif talebi oluşturulmuştur"
RE_RFX_CREATE = re.compile(
    r'^(.+?)\s+firması tarafından\s+(.+?)\s+adında bir teklif talebi oluşturulmuştur', re.I
)

def parse_rfx(msg: str) -> dict:
    m = msg.strip()
    base = {"event_category": "procurement", "is_automated": False}

    if "görüntülendi" in m.lower():
        return {**base, "action_type": "rfx_viewed", "outcome": "neutral",
                "actor_role": "supplier", "platform": "web", "sentiment": "neutral",
                "error_type": None, "named_entities": {},
                "message_template": "RFX_VIEWED",
                "message_normalized": "Teklif talebi görüntülendi.",
                "tags": ["rfx", "viewed", "procurement", "supplier_action"]}

    mo = RE_RFX_CREATE.match(m)
    if mo:
        company_id = anon_company(mo.group(1))
        rfx_id_val = anon_rfx(mo.group(2))
        return {**base, "action_type": "rfx_created", "outcome": "success",
                "actor_role": "buyer", "platform": "web", "sentiment": "positive",
                "error_type": None,
                "named_entities": {"buyer_company": company_id, "rfx_name": rfx_id_val},
                "message_template": "RFX_CREATED",
                "message_normalized": f"{company_id} firması tarafından {rfx_id_val} adında teklif talebi oluşturuldu.",
                "tags": ["rfx", "created", "procurement", "buyer_action"]}

    return {**base, "action_type": "rfx_other", "outcome": "neutral",
            "actor_role": "buyer", "platform": "web", "sentiment": "neutral",
            "error_type": None, "named_entities": {},
            "message_template": "RFX_OTHER",
            "message_normalized": _generic_anon(m),
            "tags": ["rfx", "procurement"]}


# Payment
RE_PAY_COMPANY_BUY  = re.compile(
    r'^(.+?)\s+isimli firma\s+(.+?)\s+tipinde yeni bir üyelik satın aldı', re.I
)
RE_PAY_CART         = re.compile(r'^(Gold|Premium|Silver Plus|Silver|Platin)\s+Paket Sepete Eklendi$', re.I)
RE_PAY_BUY_DIRECT   = re.compile(r'^(Gold|Premium|Silver Plus|Silver|Platin)\s+Paket Satın Aldı$', re.I)
RE_PAY_CART2        = re.compile(r'^(gold|premium|silver plus|silver|platin)\s+paketini sepete ekledi', re.I)

def _parse_payment_list(raw: list) -> tuple:
    """['Gold', 'X isimli firma Gold tipinde...'] formatını çöz"""
    if len(raw) >= 2:
        tier = str(raw[0]).strip()
        text = str(raw[1]).strip()
        mo = RE_PAY_COMPANY_BUY.match(text)
        if mo:
            return anon_company(mo.group(1)), tier
    return None, None

def parse_payment(msg) -> dict:
    base = {"event_category": "payment", "is_automated": False}

    # List formatı
    if isinstance(msg, list):
        company_id, tier = _parse_payment_list(msg)
        if company_id:
            return {**base, "action_type": "membership_purchased", "outcome": "success",
                    "actor_role": "buyer", "platform": "web", "sentiment": "positive",
                    "error_type": None, "membership_tier": tier,
                    "named_entities": {"buyer_company": company_id},
                    "message_template": "PAYMENT_MEMBERSHIP_PURCHASED",
                    "message_normalized": f"{company_id} firması {tier} tipinde üyelik satın aldı.",
                    "tags": ["payment", "success", "membership_purchase", f"tier_{tier.lower().replace(' ','_')}", "buyer_action"]}

    m = str(msg).strip()

    # List-string formatı: "['Gold', '...']"
    if m.startswith("['") or m.startswith('["'):
        try:
            parsed = eval(m)  # güvenli: sadece liste literal
            if isinstance(parsed, list):
                company_id, tier = _parse_payment_list(parsed)
                if company_id:
                    return {**base, "action_type": "membership_purchased", "outcome": "success",
                            "actor_role": "buyer", "platform": "web", "sentiment": "positive",
                            "error_type": None, "membership_tier": tier,
                            "named_entities": {"buyer_company": company_id},
                            "message_template": "PAYMENT_MEMBERSHIP_PURCHASED",
                            "message_normalized": f"{company_id} firması {tier} tipinde üyelik satın aldı.",
                            "tags": ["payment", "success", "membership_purchase", f"tier_{tier.lower().replace(' ','_')}", "buyer_action"]}
        except Exception:
            pass

    # "X isimli firma Y tipinde yeni bir üyelik satın aldı" (düz string)
    mo = RE_PAY_COMPANY_BUY.match(m)
    if mo:
        company_id = anon_company(mo.group(1))
        tier = mo.group(2).strip()
        return {**base, "action_type": "membership_purchased", "outcome": "success",
                "actor_role": "buyer", "platform": "web", "sentiment": "positive",
                "error_type": None, "membership_tier": tier,
                "named_entities": {"buyer_company": company_id},
                "message_template": "PAYMENT_MEMBERSHIP_PURCHASED",
                "message_normalized": f"{company_id} firması {tier} tipinde üyelik satın aldı.",
                "tags": ["payment", "success", "membership_purchase", f"tier_{tier.lower().replace(' ','_')}", "buyer_action"]}

    if m == "Yetersiz Paket Uyarısı":
        return {**base, "action_type": "package_insufficient_warning", "outcome": "warning",
                "actor_role": "user", "platform": "web", "sentiment": "negative",
                "error_type": None, "membership_tier": None, "named_entities": {},
                "message_template": "PAYMENT_INSUFFICIENT_PACKAGE_WARNING",
                "message_normalized": "Yetersiz paket uyarısı gösterildi.",
                "tags": ["payment", "warning", "insufficient_package", "upsell_trigger"]}

    if m == "Odeme Denemesi":
        return {**base, "action_type": "payment_attempt", "outcome": "attempt",
                "actor_role": "buyer", "platform": "web", "sentiment": "neutral",
                "error_type": None, "membership_tier": None, "named_entities": {},
                "message_template": "PAYMENT_ATTEMPT",
                "message_normalized": "Ödeme denemesi yapıldı.",
                "tags": ["payment", "attempt", "checkout"]}

    if "yeni paket satın al" in m.lower():
        return {**base, "action_type": "upgrade_button_clicked", "outcome": "neutral",
                "actor_role": "buyer", "platform": "web", "sentiment": "positive",
                "error_type": None, "membership_tier": None, "named_entities": {},
                "message_template": "PAYMENT_UPGRADE_CLICKED",
                "message_normalized": "Yeni paket satın al butonuna tıklandı.",
                "tags": ["payment", "upgrade_intent", "click", "funnel"]}

    mo = RE_PAY_CART.match(m)
    if mo:
        tier = mo.group(1)
        return {**base, "action_type": "package_added_to_cart", "outcome": "neutral",
                "actor_role": "buyer", "platform": "web", "sentiment": "positive",
                "error_type": None, "membership_tier": tier, "named_entities": {},
                "message_template": "PAYMENT_CART_ADDED",
                "message_normalized": f"{tier} paketi sepete eklendi.",
                "tags": ["payment", "cart", "checkout_funnel", f"tier_{tier.lower().replace(' ','_')}"]}

    mo = RE_PAY_CART2.match(m)
    if mo:
        tier = mo.group(1).capitalize()
        return {**base, "action_type": "package_added_to_cart", "outcome": "neutral",
                "actor_role": "buyer", "platform": "web", "sentiment": "positive",
                "error_type": None, "membership_tier": tier, "named_entities": {},
                "message_template": "PAYMENT_CART_ADDED",
                "message_normalized": f"{tier} paketi sepete eklendi.",
                "tags": ["payment", "cart", "checkout_funnel"]}

    mo = RE_PAY_BUY_DIRECT.match(m)
    if mo:
        tier = mo.group(1)
        return {**base, "action_type": "membership_purchased", "outcome": "success",
                "actor_role": "buyer", "platform": "web", "sentiment": "positive",
                "error_type": None, "membership_tier": tier, "named_entities": {},
                "message_template": "PAYMENT_DIRECT_PURCHASE",
                "message_normalized": f"{tier} paketi satın alındı.",
                "tags": ["payment", "success", "membership_purchase", f"tier_{tier.lower().replace(' ','_')}"]}

    if "hata" in m.lower() or "başarısız" in m.lower():
        return {**base, "action_type": "payment_failed", "outcome": "failure",
                "actor_role": "buyer", "platform": "web", "sentiment": "negative",
                "error_type": "payment_error", "membership_tier": None, "named_entities": {},
                "message_template": "PAYMENT_FAILED",
                "message_normalized": "Ödeme işlemi başarısız oldu.",
                "tags": ["payment", "failure", "payment_error"]}

    if "odeme bilgisi" in m.lower() or "ödeme bilgisi" in m.lower():
        return {**base, "action_type": "payment_info", "outcome": "neutral",
                "actor_role": "buyer", "platform": "web", "sentiment": "neutral",
                "error_type": None, "membership_tier": None, "named_entities": {},
                "message_template": "PAYMENT_INFO",
                "message_normalized": "Ödeme bilgisi kaydedildi.",
                "tags": ["payment", "info", "checkout"]}

    return {**base, "action_type": "payment_other", "outcome": "unknown",
            "actor_role": "buyer", "platform": "web", "sentiment": "neutral",
            "error_type": None, "membership_tier": None, "named_entities": {},
            "message_template": "PAYMENT_OTHER",
            "message_normalized": _generic_anon(m),
            "tags": ["payment"]}


# Membership
RE_MEM_EXPIRY = re.compile(
    r'^(.+?)\s+firmasına sistem tarafından paket bitimine\s+(.+?)\s+kaldığına dair bilgilendirme emaili gönderilmiştir',
    re.I | re.DOTALL
)
RE_MEM_CHANGE = re.compile(
    r'^(\S+@\S+)\s+kullanıcısı\s+(.+?)\s+firmasının paketini\s+(\w+)\s+olarak değiştirdi',
    re.I
)

def parse_membership(msg: str) -> dict:
    m = msg.strip()
    base = {"event_category": "membership", "is_automated": False}

    mo = RE_MEM_EXPIRY.match(m)
    if mo:
        company_id = anon_company(mo.group(1))
        period = mo.group(2).strip()
        return {**base, "action_type": "package_expiry_notification", "outcome": "neutral",
                "actor_role": "system", "platform": "email", "sentiment": "negative",
                "is_automated": True, "error_type": None,
                "named_entities": {"company": company_id},
                "message_template": "MEMBERSHIP_EXPIRY_NOTIFICATION",
                "message_normalized": f"{company_id} firmasına paket bitimine {period} kaldığına dair bildirim emaili gönderildi.",
                "tags": ["membership", "expiry_warning", "email_notification", "retention", "automated"]}

    mo = RE_MEM_CHANGE.match(m)
    if mo:
        email_id   = anon_email(mo.group(1))
        company_id = anon_company(mo.group(2))
        tier       = mo.group(3)
        return {**base, "action_type": "package_changed_by_admin", "outcome": "success",
                "actor_role": "admin", "platform": "web", "sentiment": "positive",
                "error_type": None, "membership_tier": tier,
                "named_entities": {"admin_email": email_id, "company": company_id},
                "message_template": "MEMBERSHIP_PACKAGE_CHANGED_BY_ADMIN",
                "message_normalized": f"{email_id} kullanıcısı {company_id} firmasının paketini {tier} olarak değiştirdi.",
                "tags": ["membership", "package_upgrade", "admin_action", f"tier_{tier.lower()}"]}

    return {**base, "action_type": "membership_other", "outcome": "neutral",
            "actor_role": "system", "platform": "web", "sentiment": "neutral",
            "is_automated": True, "error_type": None, "named_entities": {},
            "message_template": "MEMBERSHIP_OTHER",
            "message_normalized": _generic_anon(m),
            "tags": ["membership"]}


# Order: "[COMPANY] firması sipariş oluşturdu"
RE_ORDER = re.compile(r'^(.+?)\s+firması sipariş oluşturdu$', re.I)

def parse_order(msg: str) -> dict:
    m = msg.strip()
    base = {"event_category": "procurement", "is_automated": False}
    mo = RE_ORDER.match(m)
    if mo:
        company_id = anon_company(mo.group(1))
        return {**base, "action_type": "order_created", "outcome": "success",
                "actor_role": "buyer", "platform": "web", "sentiment": "positive",
                "error_type": None, "named_entities": {"buyer_company": company_id},
                "message_template": "ORDER_CREATED",
                "message_normalized": f"{company_id} firması sipariş oluşturdu.",
                "tags": ["order", "created", "procurement", "buyer_action", "conversion"]}
    return {**base, "action_type": "order_other", "outcome": "neutral",
            "actor_role": "buyer", "platform": "web", "sentiment": "neutral",
            "error_type": None, "named_entities": {},
            "message_template": "ORDER_OTHER",
            "message_normalized": _generic_anon(m),
            "tags": ["order", "procurement"]}


# Register: "[COMPANY] müşterisinin başvurusunu onayladı."
RE_REGISTER = re.compile(r'^(.+?)\s+müşterisinin başvurusunu onayladı', re.I)

def parse_register(msg: str) -> dict:
    m = msg.strip()
    base = {"event_category": "auth", "is_automated": False}
    mo = RE_REGISTER.match(m)
    if mo:
        company_id = anon_company(mo.group(1))
        return {**base, "action_type": "registration_approved", "outcome": "success",
                "actor_role": "admin", "platform": "web", "sentiment": "positive",
                "error_type": None, "named_entities": {"company": company_id},
                "message_template": "REGISTER_APPROVED",
                "message_normalized": f"{company_id} müşterisinin başvurusu onaylandı.",
                "tags": ["register", "approved", "onboarding", "admin_action", "acquisition"]}
    return {**base, "action_type": "register_other", "outcome": "neutral",
            "actor_role": "admin", "platform": "web", "sentiment": "neutral",
            "error_type": None, "named_entities": {},
            "message_template": "REGISTER_OTHER",
            "message_normalized": _generic_anon(m),
            "tags": ["register"]}


# Remind-RFX: "[COMPANY] firması tarafından [N] numaralı [RFX_NAME] teklif talebi için hatırlatma emaili gönderilmiştir"
RE_REMIND = re.compile(
    r'^(.+?)\s+firması tarafından\s+(\d+)\s+numaralı\s+(.+?)\s+teklif talebi için hatırlatma email',
    re.I
)

def parse_remind_rfx(msg: str) -> dict:
    m = msg.strip()
    base = {"event_category": "procurement", "is_automated": True}
    mo = RE_REMIND.match(m)
    if mo:
        company_id = anon_company(mo.group(1))
        rfx_num    = mo.group(2)          # RFX numarası — sayısal, kişisel değil, tutuyoruz
        rfx_name   = anon_rfx(mo.group(3))
        return {**base, "action_type": "rfx_reminder_sent", "outcome": "neutral",
                "actor_role": "system", "platform": "email", "sentiment": "neutral",
                "error_type": None,
                "named_entities": {"buyer_company": company_id, "rfx_id": rfx_num, "rfx_name": rfx_name},
                "message_template": "REMIND_RFX_SENT",
                "message_normalized": f"{company_id} firması tarafından {rfx_num} numaralı {rfx_name} teklif talebi için hatırlatma emaili gönderildi.",
                "tags": ["remind_rfx", "email_notification", "procurement", "automated", "engagement"]}
    return {**base, "action_type": "remind_rfx_other", "outcome": "neutral",
            "actor_role": "system", "platform": "email", "sentiment": "neutral",
            "error_type": None, "named_entities": {},
            "message_template": "REMIND_RFX_OTHER",
            "message_normalized": _generic_anon(m),
            "tags": ["remind_rfx", "procurement"]}


def parse_survey(msg: str) -> dict:
    return {
        "event_category": "engagement", "action_type": "supplier_survey_sent",
        "outcome": "neutral", "actor_role": "system", "platform": "email",
        "sentiment": "neutral", "is_automated": True, "error_type": None,
        "named_entities": {},
        "message_template": "SURVEY_SUPPLIER_EVALUATION_SENT",
        "message_normalized": "Tedarikçi değerlendirme anketi gönderildi.",
        "tags": ["survey", "supplier_evaluation", "email", "automated", "quality_control"]
    }


def parse_unsubscribe(msg: str) -> dict:
    return {
        "event_category": "engagement", "action_type": "email_unsubscribed",
        "outcome": "neutral", "actor_role": "user", "platform": "email",
        "sentiment": "negative", "is_automated": False, "error_type": None,
        "named_entities": {},
        "message_template": "UNSUBSCRIBE_EMAIL",
        "message_normalized": "Kullanıcı email bildirimlerinden çıkış yaptı.",
        "tags": ["unsubscribe", "email", "churn_signal", "engagement_loss"]
    }


def parse_system(msg: str) -> dict:
    m = msg.strip()
    base = {"event_category": "system", "is_automated": True, "named_entities": {}}
    if m == "NeverLogin":
        return {**base, "action_type": "user_never_logged_in", "outcome": "neutral",
                "actor_role": "system", "platform": "system", "sentiment": "negative",
                "error_type": None,
                "message_template": "SYSTEM_NEVER_LOGIN",
                "message_normalized": "Kullanıcı hiç giriş yapmamış.",
                "tags": ["system", "never_login", "inactive_user", "churn_risk"]}
    if m == "EmailUnsubscribe":
        return {**base, "action_type": "email_unsubscribed_system", "outcome": "neutral",
                "actor_role": "system", "platform": "email", "sentiment": "negative",
                "error_type": None,
                "message_template": "SYSTEM_EMAIL_UNSUBSCRIBE",
                "message_normalized": "Kullanıcı sistem tarafından email listesinden çıkarıldı.",
                "tags": ["system", "email_unsubscribe", "automated", "churn_signal"]}
    return {**base, "action_type": "system_event", "outcome": "neutral",
            "actor_role": "system", "platform": "system", "sentiment": "neutral",
            "error_type": None,
            "message_template": "SYSTEM_OTHER",
            "message_normalized": m,
            "tags": ["system"]}


# ── Genel fallback: basit firma ismi tespiti ──────────────────────────────────
_RE_FIRMA = re.compile(r'\b([A-ZÇĞİÖŞÜa-zçğışöüA-Z][^\n]{3,60}?)\s+firma', re.UNICODE)
_RE_EMAIL = re.compile(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}')

def _generic_anon(text: str) -> str:
    """Email adreslerini anonimleştir (genel fallback)."""
    def repl_email(mo):
        return anon_email(mo.group(0))
    return _RE_EMAIL.sub(repl_email, text)


# ── Zaman Özellikleri ─────────────────────────────────────────────────────────
def extract_time_features(created_date: str) -> dict:
    try:
        dt = datetime.fromisoformat(created_date)
        hour = dt.hour
        return {
            "hour_of_day"   : hour,
            "day_of_week"   : dt.strftime("%A"),           # Monday, Tuesday...
            "is_weekend"    : dt.weekday() >= 5,
            "time_of_day"   : (
                "night"     if hour < 6  else
                "morning"   if hour < 12 else
                "afternoon" if hour < 18 else
                "evening"
            ),
            "month"         : dt.month,
            "year"          : dt.year,
        }
    except Exception:
        return {}


# ── Ana dispatch fonksiyonu ───────────────────────────────────────────────────
PARSERS = {
    "login"          : parse_login,
    "reset-password" : parse_reset_password,
    "bid"            : parse_bid,
    "rfx"            : parse_rfx,
    "payment"        : parse_payment,
    "membership"     : parse_membership,
    "order"          : parse_order,
    "register"       : parse_register,
    "remind-rfx"     : parse_remind_rfx,
    "survey"         : lambda m: parse_survey(m),
    "unsubscribe"    : lambda m: parse_unsubscribe(m),
    "system"         : parse_system,
}

def tag_record(record: dict) -> dict:
    event_type = record.get("event_type", "")
    raw_msg    = record.get("message", "")

    parser = PARSERS.get(event_type)
    if parser:
        nlp = parser(raw_msg)
    else:
        nlp = {
            "event_category": "unknown", "action_type": "unknown",
            "outcome": "unknown", "actor_role": "unknown",
            "platform": "unknown", "sentiment": "neutral",
            "is_automated": False, "error_type": None,
            "named_entities": {}, "message_template": "UNKNOWN",
            "message_normalized": str(raw_msg),
            "tags": ["unknown"]
        }

    time_features = extract_time_features(record.get("created_date", ""))

    # user_id zaten hash'li görünüyor (717e632b) — dokunmuyoruz
    result = {
        "user_id"            : record.get("user_id"),
        "event_type"         : event_type,
        "created_date"       : record.get("created_date"),

        # NLP Tags
        "event_category"     : nlp.get("event_category"),
        "action_type"        : nlp.get("action_type"),
        "outcome"            : nlp.get("outcome"),
        "actor_role"         : nlp.get("actor_role"),
        "platform"           : nlp.get("platform"),
        "sentiment"          : nlp.get("sentiment"),
        "is_automated"       : nlp.get("is_automated"),
        "error_type"         : nlp.get("error_type"),
        "membership_tier"    : nlp.get("membership_tier"),
        "urgency_level"      : _urgency(nlp.get("tags", []), nlp.get("outcome")),
        "user_journey_stage" : _journey_stage(event_type, nlp.get("action_type", ""), nlp.get("outcome", "")),

        # Metin
        "message_normalized" : nlp.get("message_normalized"),
        "message_template"   : nlp.get("message_template"),
        "tags"               : nlp.get("tags", []),
        "named_entities"     : nlp.get("named_entities", {}),

        # Zaman
        **time_features,
    }
    return result


def _urgency(tags: list, outcome: str) -> str:
    if "expiry_warning" in tags or "insufficient_package" in tags:
        return "high"
    if outcome == "failure" or "churn" in " ".join(tags):
        return "medium"
    return "low"


def _journey_stage(event_type: str, action_type: str, outcome: str) -> str:
    if event_type == "register":                              return "acquisition"
    if event_type in ("login", "reset-password"):            return "activation"
    if event_type == "payment" and outcome == "success":     return "revenue"
    if event_type == "payment":                              return "revenue_funnel"
    if event_type in ("rfx", "bid", "order"):                return "retention"
    if event_type == "membership":                           return "retention"
    if event_type in ("unsubscribe", "system"):              return "churn"
    if event_type in ("survey", "remind-rfx"):               return "engagement"
    return "retention"


# ── Ana işlem döngüsü ─────────────────────────────────────────────────────────
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"HATA: {INPUT_FILE} bulunamadı.", file=sys.stderr)
        sys.exit(1)

    total = 0
    errors = 0

    print(f"İşleniyor: {INPUT_FILE}")
    print(f"Çıktı    : {OUTPUT_FILE}")
    print(f"Entity haritası: {MAP_FILE}  ← KVKK: güvenli ortamda saklayın!\n")

    with (
        open(INPUT_FILE,  "r", encoding="utf-8") as fin,
        open(OUTPUT_FILE, "w", encoding="utf-8") as fout,
    ):
        chunk = []
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                tagged = tag_record(record)
                chunk.append(tagged)
                total += 1
            except Exception as e:
                errors += 1
                continue

            if len(chunk) >= CHUNK_SIZE:
                for r in chunk:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                chunk.clear()
                print(f"  {total:>8,} kayıt işlendi...", end="\r")

        for r in chunk:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Entity haritasını kaydet
    with open(MAP_FILE, "w", encoding="utf-8") as fmap:
        json.dump(_reverse_map, fmap, ensure_ascii=False, indent=2)

    print(f"\n✓ Tamamlandı:")
    print(f"  Toplam kayıt  : {total:,}")
    print(f"  Hata          : {errors:,}")
    print(f"  Unique entity : {len(_reverse_map):,}")
    print(f"  Çıktı dosyası : {OUTPUT_FILE}")
    print(f"  Entity haritası: {MAP_FILE}")

    # İstatistik özeti
    print("\n─ Tag dağılımı (örnek 50K üzerinden) ─")
    _print_stats()


def _print_stats():
    from collections import Counter
    action_counts  = Counter()
    outcome_counts = Counter()
    journey_counts = Counter()
    n = 0
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if n >= 50000:
                break
            try:
                r = json.loads(line)
                action_counts[r.get("action_type",  "?")] += 1
                outcome_counts[r.get("outcome",     "?")] += 1
                journey_counts[r.get("user_journey_stage", "?")] += 1
                n += 1
            except Exception:
                pass
    print(f"\naction_type (top 15):")
    for k, v in action_counts.most_common(15):
        print(f"  {k:<45} {v:>6,}")
    print(f"\noutcome:")
    for k, v in outcome_counts.most_common():
        print(f"  {k:<20} {v:>6,}")
    print(f"\nuser_journey_stage:")
    for k, v in journey_counts.most_common():
        print(f"  {k:<25} {v:>6,}")


if __name__ == "__main__":
    main()
