#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree as ET

import requests


AIR_XML_URL = "http://krasecology.ru/Data/air.xml"
LATEST_POST_URL = "http://krasecology.ru/air/overonepost"
SENSOR_LIST_URL = "http://krasecology.ru/Air/GetAirSensorList/{post_id}"
SENSOR_LATEST_URL = "http://krasecology.ru/air/overonepostss"
SENSOR_HISTORY_URL = "http://krasecology.ru/Main/GetAirSensorData/{sensor_code}"

DEFAULT_SENSOR_NAMES = [
    "Температура воздуха",
    "Влажность воздуха",
    "Атм. давление",
    "Скорость ветра",
    "Направление ветра",
    "Взвешенные частицы (до 2,5 мкм)",
]

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (GraphCast-lite station downloader)",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "http://krasecology.ru/Operative/AirDetailed",
}


@dataclass
class AirPost:
    post_id: int
    name: str
    address: str | None
    created_year: int | None
    lat: float
    lon: float
    raw_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download station metadata and current sensor readings from krasecology.ru"
    )
    parser.add_argument(
        "--post-id",
        type=int,
        action="append",
        default=[],
        help="Specific post id to fetch. Can be passed multiple times.",
    )
    parser.add_argument(
        "--post-name-contains",
        action="append",
        default=[],
        help="Case-insensitive substring filter for station names. Can be passed multiple times.",
    )
    parser.add_argument(
        "--sensor-name",
        action="append",
        default=[],
        help="Sensor name to fetch. Defaults to core weather sensors if omitted. Can be passed multiple times.",
    )
    parser.add_argument(
        "--timelap",
        default="day",
        help="History window to probe for each sensor. Used only to test the history endpoint.",
    )
    parser.add_argument(
        "--all-posts",
        action="store_true",
        help="Fetch all posts from the krasecology air map.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/krasecology",
        help="Directory where JSON and Markdown outputs will be written.",
    )
    return parser.parse_args()


def split_name_parts(raw_name: str) -> tuple[str, str | None, int | None]:
    cleaned = " ".join(raw_name.split())
    if "(" not in cleaned:
        return cleaned, None, None

    title, rest = cleaned.split("(", 1)
    detail = rest.rsplit(")", 1)[0].strip()
    address = None
    created_year = None
    if "создан в" in detail:
        left, right = detail.split("создан в", 1)
        address = left.replace("адрес:", "").strip().strip(",") or None
        year_text = "".join(ch for ch in right if ch.isdigit())
        if year_text:
            created_year = int(year_text)
    else:
        address = detail.replace("адрес:", "").strip().strip(",") or None
    return title.strip(), address, created_year


def fetch_json(session: requests.Session, url: str, *, params: dict[str, Any] | None = None) -> Any:
    response = session.get(url, params=params, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_text(session: requests.Session, url: str, *, params: dict[str, Any] | None = None) -> tuple[int, str]:
    response = session.get(url, params=params, headers=REQUEST_HEADERS, timeout=30)
    return response.status_code, response.text


def load_posts(session: requests.Session) -> list[AirPost]:
    response = session.get(AIR_XML_URL, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    posts: list[AirPost] = []
    for placemark in root.findall(".//k:Placemark", ns):
        raw_name = (placemark.findtext("k:name", namespaces=ns) or "").strip()
        coordinates = placemark.find(".//k:coordinates", ns)
        if not raw_name or coordinates is None or not coordinates.text:
            continue
        lon_text, lat_text, post_id_text = [part.strip() for part in coordinates.text.split(",")[:3]]
        name, address, created_year = split_name_parts(raw_name)
        posts.append(
            AirPost(
                post_id=int(post_id_text),
                name=name,
                address=address,
                created_year=created_year,
                lat=float(lat_text),
                lon=float(lon_text),
                raw_name=raw_name,
            )
        )
    return sorted(posts, key=lambda item: item.post_id)


def select_posts(posts: list[AirPost], args: argparse.Namespace) -> list[AirPost]:
    if args.all_posts:
        return posts

    post_ids = set(args.post_id)
    name_filters = [value.casefold() for value in args.post_name_contains]
    selected: list[AirPost] = []
    for post in posts:
        if post.post_id in post_ids:
            selected.append(post)
            continue
        if name_filters and any(value in post.raw_name.casefold() for value in name_filters):
            selected.append(post)
    return selected


def build_slug(post: AirPost) -> str:
    safe = []
    for char in post.name.lower():
        if char.isalnum():
            safe.append(char)
        elif char in {" ", "-", "_", "«", "»"}:
            safe.append("_")
    slug = "".join(safe).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"{post.post_id}_{slug or 'post'}"


def fetch_sensor_current(
    session: requests.Session,
    post_id: int,
    sensor_name: str,
) -> dict[str, Any]:
    encoded_name = quote(sensor_name)
    url = f"{SENSOR_LATEST_URL}?id={post_id}&sensor={encoded_name}"
    try:
        payload = fetch_json(session, url)
        return {"ok": True, "url": url, "payload": payload}
    except Exception as exc:  # pragma: no cover - external service instability
        return {"ok": False, "url": url, "error": repr(exc)}


def probe_sensor_history(
    session: requests.Session,
    sensor_code: int,
    timelap: str,
) -> dict[str, Any]:
    url = SENSOR_HISTORY_URL.format(sensor_code=sensor_code)
    status_code, text = fetch_text(session, url, params={"timelap": timelap})
    preview = text[:300]
    result: dict[str, Any] = {
        "url": f"{url}?timelap={timelap}",
        "status_code": status_code,
        "preview": preview,
    }
    try:
        result["json"] = json.loads(text)
        result["ok"] = True
    except json.JSONDecodeError:
        result["ok"] = False
    return result


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Krasecology station download", ""]
    lines.append(f"Posts fetched: {len(report['posts'])}")
    lines.append("")
    for post in report["posts"]:
        lines.append(f"## {post['name']} (post_id={post['post_id']})")
        if post.get("address"):
            lines.append(f"Address: {post['address']}")
        lines.append(f"Coordinates: {post['lat']}, {post['lon']}")
        latest_post = post.get("latest_post") or {}
        if latest_post.get("ok"):
            payload = latest_post.get("payload") or []
            if payload:
                item = payload[0]
                lines.append(
                    "Latest post status: "
                    f"dpdkmax={item.get('dpdkmax')}, last_transfer_minutes={item.get('last_transfer_minutes')}"
                )
        else:
            lines.append(f"Latest post status error: {latest_post.get('error')}")
        lines.append("")
        lines.append("### Sensors")
        lines.append("")
        for sensor in post.get("sensors", []):
            lines.append(f"- {sensor['Name']} (code={sensor['Code']}, unit={sensor.get('Unit')})")
            current_value = sensor.get("current") or {}
            if current_value.get("ok"):
                lines.append(f"  current endpoint: {current_value['url']}")
                lines.append(f"  payload: {json.dumps(current_value.get('payload'), ensure_ascii=False)}")
            else:
                lines.append(f"  current endpoint error: {current_value.get('error')}")
            history_probe = sensor.get("history_probe") or {}
            lines.append(
                f"  history probe: status={history_probe.get('status_code')} ok={history_probe.get('ok')}"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sensor_names = args.sensor_name or DEFAULT_SENSOR_NAMES

    session = requests.Session()
    posts = load_posts(session)
    selected_posts = select_posts(posts, args)
    if not selected_posts:
        raise SystemExit("No posts matched the requested filters")

    report: dict[str, Any] = {
        "source": {
            "air_xml": AIR_XML_URL,
            "latest_post": LATEST_POST_URL,
            "sensor_list": SENSOR_LIST_URL,
            "sensor_latest": SENSOR_LATEST_URL,
            "sensor_history": SENSOR_HISTORY_URL,
        },
        "filters": {
            "post_id": args.post_id,
            "post_name_contains": args.post_name_contains,
            "sensor_name": sensor_names,
            "timelap": args.timelap,
        },
        "posts": [],
    }

    for post in selected_posts:
        latest_post: dict[str, Any]
        try:
            latest_payload = fetch_json(session, LATEST_POST_URL, params={"id": post.post_id})
            latest_post = {"ok": True, "payload": latest_payload}
        except Exception as exc:  # pragma: no cover - external service instability
            latest_post = {"ok": False, "error": repr(exc)}

        sensors = fetch_json(session, SENSOR_LIST_URL.format(post_id=post.post_id))
        filtered_sensors = [sensor for sensor in sensors if sensor.get("Name") in sensor_names]
        for sensor in filtered_sensors:
            sensor["current"] = fetch_sensor_current(session, post.post_id, sensor["Name"])
            sensor["history_probe"] = probe_sensor_history(session, sensor["Code"], args.timelap)

        post_payload = asdict(post)
        post_payload["latest_post"] = latest_post
        post_payload["sensors"] = filtered_sensors
        report["posts"].append(post_payload)

        slug = build_slug(post)
        json_path = out_dir / f"{slug}.json"
        md_path = out_dir / f"{slug}.md"
        json_path.write_text(json.dumps(post_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(render_markdown({"posts": [post_payload]}), encoding="utf-8")

    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    summary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"Saved krasecology data to {out_dir}")
    for post in report["posts"]:
        print(f"- {post['post_id']}: {post['name']}")


if __name__ == "__main__":
    main()