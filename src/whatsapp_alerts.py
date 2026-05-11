"""WhatsApp Cloud API client (Meta official).

Uses Meta's WhatsApp Business Cloud API:
    https://developers.facebook.com/docs/whatsapp/cloud-api

Why not pywhatkit / Twilio / whatsapp-web.js?
    * pywhatkit pops a real Chrome tab for every message -> unusable in practice
    * Twilio is reliable but costs money per message
    * Cloud API is FREE up to 1000 conversations/month, fully headless, officially supported

LIMITATIONS — read these:
    * Cloud API DOES NOT support sending into WhatsApp groups (Meta restriction).
      To reach multiple people, loop through a recipient list (a "broadcast list").
    * Freeform text only works within 24h of the recipient's last inbound message.
      Outside that window you must use a pre-approved template (e.g. `hello_world`).
    * The Meta test phone number can only message up to 5 verified recipients;
      a production phone number requires business verification.

SETUP:
    1. https://developers.facebook.com/ -> create app -> add "WhatsApp" product
    2. In the "API Setup" tab note down:
         - Phone number ID (the bot's sending number ID, not the recipient)
         - Temporary access token (24h) — or generate a permanent one via a System User
    3. Add each recipient WhatsApp number under "To" — Meta texts them an OTP
    4. Have each recipient text "hi" to the bot's WhatsApp number once
       (this opens the 24h freeform conversation window)
    5. Add credentials to .env:
         WHATSAPP_PHONE_NUMBER_ID=1234567890
         WHATSAPP_ACCESS_TOKEN=EAAG...
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

import requests

log = logging.getLogger(__name__)

API_VERSION = os.environ.get("WHATSAPP_API_VERSION", "v23.0")
API_BASE = f"https://graph.facebook.com/{API_VERSION}"
HTTP_TIMEOUT = 15
TEXT_LIMIT = 4096  # WhatsApp body length limit


class WhatsAppError(RuntimeError):
    pass


def _credentials() -> tuple[str, str]:
    phone_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
    token    = os.environ.get("WHATSAPP_ACCESS_TOKEN")
    if not phone_id or not token:
        raise WhatsAppError(
            "Missing WhatsApp Cloud API credentials. Add to your .env:\n"
            "  WHATSAPP_PHONE_NUMBER_ID=...\n"
            "  WHATSAPP_ACCESS_TOKEN=...\n"
            "(See src/whatsapp_alerts.py docstring for setup steps.)"
        )
    return phone_id, token


def _normalize(phone: str) -> str:
    """Cloud API wants the number with country code but NO leading '+'."""
    p = (phone or "").strip().replace(" ", "").replace("-", "").lstrip("+")
    if not p.isdigit() or len(p) < 8:
        raise WhatsAppError(f"Invalid phone number: {phone!r} (expected E.164, e.g. +923001234567)")
    return p


def _post(payload: dict) -> dict:
    phone_id, token = _credentials()
    url = f"{API_BASE}/{phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=HTTP_TIMEOUT)
    except requests.RequestException as e:
        raise WhatsAppError(f"Network error contacting Cloud API: {e}") from e

    if r.status_code >= 400:
        try:
            err = r.json().get("error", {})
        except ValueError:
            err = {"message": r.text}
        code = err.get("code")
        msg  = err.get("message", "Unknown error")
        hint = _diagnose(code, msg)
        raise WhatsAppError(f"Cloud API error: {msg} (code={code}, http={r.status_code}){hint}")

    return r.json()


def _diagnose(code, msg: str) -> str:
    """Tack on an actionable hint for common Cloud API errors."""
    m = (msg or "").lower()
    if code == 131047 or ("re-engagement" in m) or ("24" in m and "hour" in m):
        return ("\nHint: 24-hour window expired. Have the recipient text the bot's "
                "WhatsApp number once to reopen the conversation, or send a template "
                "via send_template().")
    if code == 131030:
        return ("\nHint: Recipient is not on the allowed list. Add their number in the "
                "Meta dashboard under API Setup -> To, and confirm the OTP they receive.")
    if code in (190, 102, 463):
        return "\nHint: Access token expired or invalid. Generate a new one in the Meta dashboard."
    if code == 100:
        return "\nHint: Bad parameter — check WHATSAPP_PHONE_NUMBER_ID is the bot's number ID (not the recipient)."
    return ""


# ---------- public API ----------

def send_message(phone: str, message: str) -> dict:
    """Send a freeform WhatsApp text message via Cloud API.

    Only works within 24h of the recipient's last inbound message. Outside
    that window, this raises WhatsAppError with a clear hint.
    """
    return _post({
        "messaging_product": "whatsapp",
        "to":   _normalize(phone),
        "type": "text",
        "text": {"body": message[:TEXT_LIMIT]},
    })


def send_template(phone: str, template_name: str = "hello_world", language: str = "en_US") -> dict:
    """Send a pre-approved template. Works even outside the 24h freeform window.

    `hello_world` is built-in. Custom templates must be approved in Meta Business Manager.
    """
    return _post({
        "messaging_product": "whatsapp",
        "to":       _normalize(phone),
        "type":     "template",
        "template": {"name": template_name, "language": {"code": language}},
    })


def send_to_many(phones: Iterable[str], message: str) -> tuple[int, list[str]]:
    """Fan out the same message to a list of recipients. Returns (sent_count, errors)."""
    sent = 0
    errors: list[str] = []
    for p in phones:
        try:
            send_message(p, message)
            sent += 1
        except WhatsAppError as e:
            errors.append(f"{p}: {e}")
            log.warning("WhatsApp send to %s failed: %s", p, e)
    return sent, errors


def credentials_ok() -> bool:
    """Cheap check used by the UI to decide whether to show a 'credentials missing' banner."""
    return bool(os.environ.get("WHATSAPP_PHONE_NUMBER_ID") and os.environ.get("WHATSAPP_ACCESS_TOKEN"))
