"""PSX → WhatsApp alert worker.

Run this in a separate terminal alongside `streamlit run medibot.py`. It polls
the PSX market-watch page at a fixed cadence, evaluates the rules saved in
`data/watchlist.json` (which the Streamlit UI edits), and fires WhatsApp
messages via pywhatkit when a rule triggers.

Usage:
    python alert_worker.py
    python alert_worker.py --interval 120        # poll every 2 minutes
    python alert_worker.py --off-hours           # run even when PSX is closed
    python alert_worker.py --once                # one-shot, useful for cron
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone

from src import stocks
from src.whatsapp_alerts import WhatsAppError, credentials_ok, send_message


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("alert_worker")


_stop = False


def _handle_signal(signum, frame):
    global _stop
    log.info("Signal %s received — shutting down after current cycle.", signum)
    _stop = True


def _fanout(recipients: list[str], message: str, label: str) -> int:
    """Send `message` to each recipient. Returns successful-send count."""
    sent = 0
    for ph in recipients:
        try:
            send_message(ph, message)
            sent += 1
            log.info("Sent %s to %s", label, ph)
        except WhatsAppError as e:
            log.error("Send %s to %s failed: %s", label, ph, e)
    return sent


def run_once(off_hours: bool = False) -> int:
    """One polling cycle. Returns number of WhatsApp messages dispatched.

    Handles both:
      * threshold alerts (rule-based, fires once when matched, then cools down)
      * daily briefings (3x/day scheduled summaries)
    Each message is fanned out to every recipient in watchlist.recipients.
    """
    if not credentials_ok():
        log.error("WhatsApp Cloud API credentials missing — set WHATSAPP_PHONE_NUMBER_ID "
                  "and WHATSAPP_ACCESS_TOKEN in .env.")
        return 0

    watchlist = stocks.Watchlist.load()
    if not watchlist.recipients:
        log.warning("No recipients configured. Add at least one in the Streamlit sidebar.")
        return 0

    has_rules = bool(watchlist.rules)
    has_briefings = watchlist.briefings.enabled
    if not has_rules and not has_briefings:
        log.info("Nothing to do — no rules or briefings enabled.")
        return 0

    due_slots = stocks.due_briefings(watchlist) if has_briefings else []
    market_open = stocks.market_is_open()
    if not off_hours and not market_open and not due_slots:
        log.info("Market closed and no briefings due — skipping poll. Use --off-hours to override.")
        return 0

    try:
        quotes = stocks.fetch_market_watch(force=True)
    except Exception as e:
        log.error("Failed to fetch PSX snapshot: %s", e)
        return 0

    total_sent = 0

    # ---- Daily briefings ----
    for slot in due_slots:
        msg = stocks.generate_briefing(quotes, watchlist, slot=slot)
        sent = _fanout(watchlist.recipients, msg, f"briefing[{slot}]")
        if sent:
            stocks.mark_briefing_sent(watchlist, slot)
            watchlist.save()
            total_sent += sent
        time.sleep(2)

    # ---- Threshold rule alerts ----
    if has_rules and (market_open or off_hours):
        triggered = stocks.evaluate_alerts(watchlist, quotes)
        if triggered:
            log.info("%d rule(s) triggered.", len(triggered))
            for alert in triggered:
                sent = _fanout(watchlist.recipients, alert["message"], f"alert[{alert['rule'].symbol}]")
                if sent:
                    stocks.mark_fired(watchlist, alert["rule_key"])
                    watchlist.save()
                    total_sent += sent
                time.sleep(2)
        else:
            log.info("Polled %d quotes, %d rules — nothing triggered.", len(quotes), len(watchlist.rules))

    return total_sent


def main():
    parser = argparse.ArgumentParser(description="PSX → WhatsApp alert worker")
    parser.add_argument("--interval", type=int, default=180,
                        help="Polling interval in seconds (default: 180 = 3min)")
    parser.add_argument("--off-hours", action="store_true",
                        help="Keep polling even when PSX market is closed")
    parser.add_argument("--once", action="store_true",
                        help="Run a single polling cycle and exit")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    log.info("PSX alert worker starting — interval=%ds, off_hours=%s", args.interval, args.off_hours)

    if args.once:
        sent = run_once(off_hours=args.off_hours)
        log.info("Done. %d alert(s) dispatched.", sent)
        return

    while not _stop:
        try:
            run_once(off_hours=args.off_hours)
        except Exception:
            log.exception("Unhandled error in poll cycle — continuing.")

        # Sleep in 1s chunks so SIGINT is responsive.
        for _ in range(args.interval):
            if _stop:
                break
            time.sleep(1)

    log.info("Worker stopped.")


if __name__ == "__main__":
    sys.exit(main() or 0)
