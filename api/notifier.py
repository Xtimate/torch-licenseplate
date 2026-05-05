import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api.config import TELEGRAM_CHAT_ID, TELEGRAM_TOKEN


async def send_telegram_alert(
    text: str, confidence: float, country: str | None, notes: str | None = None
):

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    country_str = country if country else "Unknown"
    notes_str = f"\nNotes: {notes}" if notes else ""
    message = (
        f" *Watchlist Hit*\n"
        f"Plate: {text}\n"
        f"Confidence: {confidence:.2f}\n"
        f"Country: {country_str}\n"
        f"{notes_str}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "Markdown",
                },
            )

        except Exception as e:
            print(f"  Failed to send Telegram alert: {e}")
