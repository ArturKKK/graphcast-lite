"""GraphCast-lite Weather — FastAPI backend.

Serves the static frontend and forecast JSON.
No torch dependency — just reads pre-generated forecast.json.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
FORECAST_JSON = STATIC_DIR / "forecast.json"

app = FastAPI(title="GraphCast-lite Weather", docs_url=None, redoc_url=None)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/api/forecast")
async def forecast():
    if not FORECAST_JSON.exists():
        return JSONResponse(
            {"error": "Прогноз пока недоступен", "forecast": None},
            status_code=503,
        )
    with open(FORECAST_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)


@app.get("/api/status")
async def status():
    if not FORECAST_JSON.exists():
        return JSONResponse({
            "status": "no_data",
            "message": "Файл прогноза не найден",
            "last_update": None,
            "file_age_minutes": None,
        })

    mtime = os.path.getmtime(FORECAST_JSON)
    last_update = datetime.fromtimestamp(mtime, tz=timezone.utc)
    age_minutes = (datetime.now(timezone.utc) - last_update).total_seconds() / 60

    with open(FORECAST_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    return JSONResponse({
        "status": "ok" if age_minutes < 480 else "stale",
        "last_update": last_update.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "file_age_minutes": round(age_minutes, 1),
        "last_cycle": data.get("last_cycle"),
        "mos_applied": data.get("mos_applied", False),
        "n_summary_steps": len(data.get("summary", [])),
        "n_map_points": data.get("n_map_points", 0),
        "warnings": data.get("warnings", []),
    })


# Static files (CSS, JS, etc.) — mounted AFTER explicit routes
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
