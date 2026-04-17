"""Scraper for Yandex Pogoda — parses current weather and hourly forecast.

Run from the Russian VPS (138.124.77.216) to avoid captcha.
Outputs JSON to stdout or writes to a file.
"""

import json
import re
import sys
import urllib.request
from datetime import datetime


URL = "https://yandex.ru/pogoda/krasnoyarsk"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "ru-RU,ru;q=0.9",
}


def fetch_html() -> str:
    req = urllib.request.Request(URL, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8")


def parse_current(html: str) -> dict:
    """Extract current weather from the fact block."""
    result = {}

    # Temperature: from the accessibility text block
    m = re.search(
        r"воздуха\s*([−\-+]?\d+)°.*?"
        r"ощущается как\s*([−\-+]?\d+)°.*?"
        r"ветра\s*([\d,]+)\s*м/с,\s*([^.]+)\.\s*"
        r"Давление\s*(\d+)\s*мм.*?"
        r"Влажность\s*(\d+)%",
        html,
        re.DOTALL,
    )
    if m:
        result["temp"] = int(m.group(1).replace("−", "-"))
        result["feels_like"] = int(m.group(2).replace("−", "-"))
        result["wind_speed"] = float(m.group(3).replace(",", "."))
        result["wind_dir"] = m.group(4).strip()
        result["pressure_mmhg"] = int(m.group(5))
        result["humidity"] = int(m.group(6))

    # Condition (e.g. "Ясно", "Пасмурно")
    m = re.search(r'AppFact_warning__first_text[^>]*>([^<]+)<', html)
    if m:
        result["condition"] = m.group(1).strip()

    # Secondary condition (e.g. "Сегодня осадков не ожидается")
    m = re.search(r'AppFact_warning__second[^>]*>\s*([^<]+)<', html)
    if m:
        result["condition_extra"] = m.group(1).strip()

    # Yesterday temp
    m = re.search(r"Вчера[^−\-\d]*([−\-]?\d+)°", html)
    if m:
        result["yesterday_temp"] = int(m.group(1).replace("−", "-"))

    return result


def parse_hourly(html: str) -> list:
    """Extract hourly forecast from the page."""
    hours = []
    # Hourly data: time + temp + condition
    # Pattern: "20:00: -6°, ясно" from the accessibility/aria text
    for m in re.finditer(
        r"(\d{2}):00:\s*([−\-+]?\d+)°,\s*([^,]+?)(?:,\s*Ощущается|\.)",
        html,
    ):
        hours.append({
            "hour": int(m.group(1)),
            "temp": int(m.group(2).replace("−", "-")),
            "condition": m.group(3).strip(),
        })
    return hours[:24]


def parse_daily(html: str) -> list:
    """Extract daily forecast (day parts) from the page."""
    days = []
    # Pattern: "Сегодня, 17 апреля:утром температура воздуха -7°..."
    day_pattern = re.compile(
        r"(Сегодня|Завтра|Понедельник|Вторник|Среда|Четверг|Пятница|Суббота|Воскресенье)"
        r",\s*(\d+\s*\w+):"
        r"(.*?)(?=(?:Сегодня|Завтра|Понедельник|Вторник|Среда|Четверг|Пятница|Суббота|Воскресенье),\s*\d|\Z)",
        re.DOTALL,
    )
    for dm in day_pattern.finditer(html):
        day_name = dm.group(1)
        date_str = dm.group(2).strip()
        body = dm.group(3)

        parts = {}
        for part_match in re.finditer(
            r"(утром|днём|вечером|ночью)\s*(?:температура воздуха\s*)?"
            r"([−\-+]?\d+)°.*?"
            r"ветра\s*([\d,]+)\s*м/с,\s*([^,]+),\s*"
            r"влажность\s*(\d+)%.*?"
            r"давление\s*(\d+)",
            body,
            re.DOTALL,
        ):
            parts[part_match.group(1)] = {
                "temp": int(part_match.group(2).replace("−", "-")),
                "wind_speed": float(part_match.group(3).replace(",", ".")),
                "wind_dir": part_match.group(4).strip(),
                "humidity": int(part_match.group(5)),
                "pressure": int(part_match.group(6)),
            }

        if parts:
            days.append({
                "day": day_name,
                "date": date_str,
                "parts": parts,
            })

    return days[:4]  # Today + 3 days


def scrape() -> dict:
    html = fetch_html()
    return {
        "source": "Яндекс.Погода",
        "url": URL,
        "scraped_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "city": "Красноярск",
        "current": parse_current(html),
        "hourly": parse_hourly(html),
        "daily": parse_daily(html),
    }


if __name__ == "__main__":
    data = scrape()
    out = json.dumps(data, ensure_ascii=False, indent=2)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Saved to {sys.argv[1]}", file=sys.stderr)
    else:
        print(out)
