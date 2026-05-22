/* GraphCast-lite Weather — Russia frontend (mirrors app.js, MSK tz) */
(function () {
  "use strict";

  const REFRESH_INTERVAL_MS = 10 * 60 * 1000;
  const MSK_OFFSET_H = 3;
  const OVERLAY_BASE = "/static/overlays_russia";

  let tMin = -10, tMax = 20;

  function updateTempRange(data) {
    var summary = getActiveSummary(data);
    var allTemps = summary.map(function(s) { return s.t2m_celsius; });
    if (data.grid_points) {
      for (var i = 0; i < data.grid_points.length; i++) {
        var pt = data.grid_points[i];
        for (var j = 0; j < pt.steps.length; j++) allTemps.push(pt.steps[j].t);
      }
    }
    var dataMin = Math.min.apply(null, allTemps);
    var dataMax = Math.max.apply(null, allTemps);
    tMin = Math.floor(dataMin / 5) * 5 - 5;
    tMax = Math.ceil(dataMax / 5) * 5 + 5;
    if (tMax - tMin < 10) { tMin -= 5; tMax += 5; }
  }

  var COLOR_STOPS = [
    { f: 0.0,  r: 49,  g: 54,  b: 149 },
    { f: 0.15, r: 69,  g: 117, b: 180 },
    { f: 0.3,  r: 116, g: 173, b: 209 },
    { f: 0.45, r: 171, g: 217, b: 233 },
    { f: 0.55, r: 254, g: 224, b: 144 },
    { f: 0.7,  r: 253, g: 174, b: 97  },
    { f: 0.85, r: 244, g: 109, b: 67  },
    { f: 1.0,  r: 215, g: 48,  b: 39  },
  ];

  var forecastData = null;
  var map = null;
  var mapMarkers = [];
  var stationMarkers = [];
  var currentStep = 0;
  var tempChart = null;
  var currentScope = "core";
  var overlayMeta = null;
  var overlayLayers = {};
  var windArrowMarkers = [];
  var layerState = { temp: true, wind: false, precip: false, pressure: false, points: false, stations: true };

  function getActiveSummary(data) {
    if (currentScope === "core") return data.summary_core || data.summary_city || [];
    if (currentScope === "city") return data.summary_city || [];
    return data.summary_region || [];
  }
  function getActivePointCount(data) {
    if (currentScope === "core") return data.n_core_points || 0;
    if (currentScope === "city") return data.n_city_points || 0;
    return data.n_region_points || 0;
  }

  function lerp(a, b, t) { return a + (b - a) * t; }

  function tempColor(t) {
    var f = (t - tMin) / (tMax - tMin);
    f = Math.max(0, Math.min(1, f));
    for (var i = 0; i < COLOR_STOPS.length - 1; i++) {
      if (f >= COLOR_STOPS[i].f && f <= COLOR_STOPS[i + 1].f) {
        var t2 = (f - COLOR_STOPS[i].f) / (COLOR_STOPS[i + 1].f - COLOR_STOPS[i].f);
        var r = Math.round(lerp(COLOR_STOPS[i].r, COLOR_STOPS[i + 1].r, t2));
        var g = Math.round(lerp(COLOR_STOPS[i].g, COLOR_STOPS[i + 1].g, t2));
        var b = Math.round(lerp(COLOR_STOPS[i].b, COLOR_STOPS[i + 1].b, t2));
        return "rgb(" + r + "," + g + "," + b + ")";
      }
    }
    var cs = f <= 0 ? COLOR_STOPS[0] : COLOR_STOPS[COLOR_STOPS.length - 1];
    return "rgb(" + cs.r + "," + cs.g + "," + cs.b + ")";
  }

  function windArrowChar(deg) {
    var arrows = ["\u2193", "\u2199", "\u2190", "\u2196", "\u2191", "\u2197", "\u2192", "\u2198"];
    return arrows[Math.round(deg / 45) % 8];
  }

  function weatherEmoji(t2m, precip, hourLocal) {
    var isNight = hourLocal < 6 || hourLocal >= 22;
    if (t2m <= -15) return "\uD83E\uDD76";
    if (precip > 2) return t2m <= 0 ? "\u2744\uFE0F" : "\uD83C\uDF27\uFE0F";
    if (precip > 0.3) return t2m <= 0 ? "\uD83C\uDF28\uFE0F" : "\uD83C\uDF26\uFE0F";
    if (isNight) return "\uD83C\uDF19";
    if (t2m >= 25) return "\u2600\uFE0F";
    return "\uD83C\uDF24\uFE0F";
  }

  function formatMskTime(utcStr) {
    var d = new Date(utcStr);
    var loc = new Date(d.getTime() + MSK_OFFSET_H * 3600000);
    var day = ("0" + loc.getUTCDate()).slice(-2);
    var mon = ("0" + (loc.getUTCMonth() + 1)).slice(-2);
    var hh = ("0" + loc.getUTCHours()).slice(-2);
    var mm = ("0" + loc.getUTCMinutes()).slice(-2);
    return day + "." + mon + " " + hh + ":" + mm;
  }
  function getMskHour(utcStr) {
    var d = new Date(utcStr);
    return new Date(d.getTime() + MSK_OFFSET_H * 3600000).getUTCHours();
  }

  function tempClass(t) {
    if (t >= 20) return "temp-warm";
    if (t <= 0) return "temp-cold";
    return "temp-mild";
  }

  function fetchForecast() {
    return fetch("/api/russia_forecast")
      .then(function (resp) { if (!resp.ok) { showError("HTTP " + resp.status); return null; } return resp.json(); })
      .then(function (data) { if (!data) return null; if (data.error) { showError(data.error); return null; } return data; })
      .catch(function (e) { showError(e.message); return null; });
  }
  function fetchOverlayMeta() {
    var bust = "?v=" + Date.now();
    return fetch(OVERLAY_BASE + "/meta.json" + bust)
      .then(function(r) { return r.ok ? r.json() : null; })
      .then(function(meta) { overlayMeta = meta; })
      .catch(function() { overlayMeta = null; });
  }
  function showError(msg) {
    var el = document.getElementById("status-text");
    if (el) { el.textContent = msg; el.style.color = "var(--accent-red)"; }
    console.error(msg);
  }

  function setupScopeToggle() {
    var btns = document.querySelectorAll(".scope-btn");
    btns.forEach(function (btn) {
      btn.addEventListener("click", function () {
        var scope = btn.getAttribute("data-scope");
        if (scope === currentScope) return;
        currentScope = scope;
        btns.forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");
        if (forecastData) renderAll(forecastData);
      });
    });
  }

  function renderAll(data) {
    updateTempRange(data);
    renderCurrentCard(data);
    renderTable(data);
    renderChart(data);
    updateLegend();
    renderMapMarkers(data, currentStep);
    renderStationMarkers(data, currentStep);
    updateOverlays(currentStep);
  }

  function renderCurrentCard(data) {
    var card = document.getElementById("current-card");
    var summary = getActiveSummary(data);
    var s = summary[0];
    if (!s) return;

    card.style.display = "";
    var nPts = getActivePointCount(data);
    var scopeNames = { core: "столицам", city: "крупным городам", region: "России" };
    var unit = currentScope === "region" ? " станций)" : " точек)";
    document.getElementById("city-avg-badge").textContent = "среднее по " + scopeNames[currentScope] + " (" + nPts + unit;

    document.getElementById("current-time").textContent = formatMskTime(s.valid_time_utc) + " (московское время)";
    document.getElementById("current-temp").textContent = s.t2m_celsius > 0 ? "+" + s.t2m_celsius : s.t2m_celsius;

    var rangeEl = document.getElementById("current-temp-range");
    if (rangeEl && s.t2m_min !== undefined) {
      var rMin = s.t2m_min > 0 ? "+" + s.t2m_min : "" + s.t2m_min;
      var rMax = s.t2m_max > 0 ? "+" + s.t2m_max : "" + s.t2m_max;
      rangeEl.textContent = "от " + rMin + " до " + rMax + "°C";
    }

    var emoji = s.precip_type_icon || weatherEmoji(s.t2m_celsius, s.precip_mm, getMskHour(s.valid_time_utc));
    document.getElementById("current-emoji").textContent = emoji;
    document.getElementById("current-wind").textContent = s.wind_speed_ms + " м/с " + s.wind_direction_text;
    document.getElementById("current-wind-arrow").textContent = windArrowChar(s.wind_direction_deg);
    document.getElementById("current-wind-arrow").style.transform = "rotate(" + s.wind_direction_deg + "deg)";
    document.getElementById("current-pressure").textContent = s.pressure_mmhg + " мм рт.ст.";
    document.getElementById("current-precip").textContent = s.precip_mm + " мм";

    var gustEl = document.getElementById("current-gust");
    gustEl.textContent = (s.wind_gust_ms && s.wind_gust_ms > s.wind_speed_ms + 0.5)
      ? "порывы " + s.wind_gust_ms + " м/с" : "";
    document.getElementById("current-precip-type").textContent = s.precip_intensity_text || "";
  }

  function renderTable(data) {
    var tbody = document.getElementById("forecast-tbody");
    tbody.innerHTML = "";
    var summary = getActiveSummary(data);
    var pts = getActivePointCount(data);
    var ptsEl = document.getElementById("table-city-pts");
    if (ptsEl) ptsEl.textContent = pts;
    var scopeEl = document.getElementById("table-scope-label");
    if (scopeEl) {
      var names = { core: "столицам", city: "крупным городам", region: "России" };
      scopeEl.textContent = names[currentScope];
    }

    for (var i = 0; i < summary.length; i++) {
      var s = summary[i];
      var hr = getMskHour(s.valid_time_utc);
      var emoji = s.precip_type_icon || weatherEmoji(s.t2m_celsius, s.precip_mm, hr);
      var tSign = s.t2m_celsius > 0 ? "+" : "";

      var windCell = "" + s.wind_speed_ms;
      if (s.wind_gust_ms && s.wind_gust_ms > s.wind_speed_ms + 0.5) {
        windCell += '<span class="gust-table"> (пор. ' + s.wind_gust_ms + ')</span>';
      }
      var precipCell = "" + s.precip_mm;
      if (s.precip_intensity_text) {
        precipCell += ' <span class="precip-label">' + s.precip_intensity_text + '</span>';
      }

      var tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + formatMskTime(s.valid_time_utc) + "</td>" +
        "<td>" + emoji + "</td>" +
        '<td class="temp-cell ' + tempClass(s.t2m_celsius) + '">' + tSign + s.t2m_celsius + "</td>" +
        "<td>" + windCell + "</td>" +
        '<td><span class="wind-dir-cell"><span class="table-wind-arrow" style="transform:rotate(' + s.wind_direction_deg + 'deg)">\u2191</span> ' + s.wind_direction_text + "</span></td>" +
        "<td>" + s.pressure_mmhg + "</td>" +
        "<td>" + precipCell + "</td>";
      tbody.appendChild(tr);
    }
  }

  function renderChart(data) {
    var ctx = document.getElementById("temp-chart").getContext("2d");
    var summary = getActiveSummary(data);
    var labels = summary.map(function (s) { return formatMskTime(s.valid_time_utc); });
    var temps = summary.map(function (s) { return s.t2m_celsius; });

    if (tempChart) tempChart.destroy();
    tempChart = new Chart(ctx, {
      type: "line",
      data: { labels: labels, datasets: [{
        label: "Температура (°C)", data: temps,
        borderColor: "#4fc3f7", backgroundColor: "rgba(79,195,247,0.1)",
        borderWidth: 2, pointBackgroundColor: temps.map(function (t) { return tempColor(t); }),
        pointRadius: 5, pointHoverRadius: 7, fill: true, tension: 0.3,
      }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: function (ctx) {
            var s = summary[ctx.dataIndex];
            var lines = ["T: " + s.t2m_celsius + "°C"];
            if (s.t2m_min !== undefined) lines.push("Диапазон: " + s.t2m_min + "..." + s.t2m_max + "°C");
            lines.push("Ветер: " + s.wind_speed_ms + " м/с " + s.wind_direction_text);
            lines.push("Давл: " + s.pressure_mmhg + " мм");
            return lines;
          } } },
        },
        scales: {
          x: { ticks: { color: "#8899aa", maxRotation: 45 }, grid: { color: "rgba(42,58,78,0.3)" } },
          y: { ticks: { color: "#8899aa", callback: function (v) { return (v > 0 ? "+" + v : v) + "°"; } },
               grid: { color: "rgba(42,58,78,0.3)" } },
        },
      },
    });
  }

  function initMap() {
    map = L.map("map", {
      center: [62, 95], zoom: 3, minZoom: 3, maxZoom: 16,
      zoomControl: true, attributionControl: false,
      worldCopyJump: false,
    });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19, maxNativeZoom: 19,
    }).addTo(map);
    document.getElementById("map-legend").innerHTML = "";
    setupHoverProbe();
  }

  // ── Hover-anywhere probe: IDW-interpolate t/ws/wd/pr/p from K nearest grid points ──
  var hoverTooltipEl = null;
  function setupHoverProbe() {
    if (!hoverTooltipEl) {
      hoverTooltipEl = document.createElement("div");
      hoverTooltipEl.className = "hover-probe-tip";
      hoverTooltipEl.style.cssText =
        "position:absolute;pointer-events:none;z-index:9999;background:rgba(10,16,24,0.92);" +
        "color:#e7ecf3;font:12px/1.4 -apple-system,system-ui,sans-serif;border-radius:6px;" +
        "padding:6px 9px;box-shadow:0 4px 12px rgba(0,0,0,0.5);border:1px solid #2a3a4e;" +
        "white-space:nowrap;display:none;transform:translate(12px,12px)";
      document.getElementById("map").appendChild(hoverTooltipEl);
    }
    map.on("mousemove", onMapHover);
    map.on("mouseout", function () { if (hoverTooltipEl) hoverTooltipEl.style.display = "none"; });
  }

  function mercatorY(latDeg) {
    var clamped = Math.max(-85.05112878, Math.min(85.05112878, latDeg));
    var latRad = clamped * Math.PI / 180;
    return Math.log(Math.tan(Math.PI * 0.25 + latRad * 0.5));
  }

  function ensureProjectedCoords(pts) {
    if (!pts || !pts.length) return;
    if (pts[0]._mx !== undefined && pts[0]._my !== undefined) return;
    for (var i = 0; i < pts.length; i++) {
      pts[i]._mx = pts[i].lon * Math.PI / 180;
      pts[i]._my = mercatorY(pts[i].lat);
    }
  }

  function onMapHover(e) {
    if (!forecastData || !hoverTooltipEl) return;
    var pts = forecastData.grid_points || [];
    if (!pts.length) return;
    var lat = e.latlng.lat, lon = e.latlng.lng;
    ensureProjectedCoords(pts);
    var mx = lon * Math.PI / 180;
    var my = mercatorY(lat);
    var K = 4, POW = 2.0;
    // Find K nearest points in projected WebMercator space (x=lon_rad,
    // y=mercator_y), identical to precip overlay rasterization.
    var best = []; // {d2, idx}
    for (var i = 0; i < pts.length; i++) {
      var dx = mx - pts[i]._mx;
      var dy = my - pts[i]._my;
      var d2 = dx * dx + dy * dy;
      if (best.length < K) {
        best.push({ d2: d2, idx: i });
        if (best.length === K) best.sort(function(a,b){return a.d2-b.d2;});
      } else if (d2 < best[K-1].d2) {
        best[K-1] = { d2: d2, idx: i };
        best.sort(function(a,b){return a.d2-b.d2;});
      }
    }
    var sd_first = pts[best[0].idx].steps[currentStep];
    if (!sd_first) return;
    // IDW (power=2) for smooth fields. If hovering on top of a grid point use
    // it directly. NB: precip is reported as the NEAREST grid-point value to
    // stay consistent with the precip overlay (Voronoi nearest-neighbour).
    if (best[0].d2 < 1e-10) {
      showHoverTip(e, sd_first.t, sd_first.ws, sd_first.wd, sd_first.p, sd_first.pr, best[0].idx, pts);
      return;
    }
    var sumW = 0, t = 0, u = 0, v = 0, pr = 0;
    for (var k = 0; k < best.length; k++) {
      var sd = pts[best[k].idx].steps[currentStep];
      if (!sd) continue;
      var w = 1.0 / Math.pow(best[k].d2, POW * 0.5);
      sumW += w;
      t += w * sd.t;
      var wdRad = sd.wd * Math.PI / 180;
      var uk = -sd.ws * Math.sin(wdRad);
      var vk = -sd.ws * Math.cos(wdRad);
      u += w * uk;
      v += w * vk;
      pr += w * sd.pr;
    }
    if (sumW <= 0) return;
    t /= sumW; u /= sumW; v /= sumW; pr /= sumW;
    var ws = Math.sqrt(u*u + v*v);
    var wd = (Math.atan2(-u, -v) * 180 / Math.PI + 360) % 360;
    // Precip: nearest-neighbour (matches overlay).
    var p = sd_first.p;
    showHoverTip(e, t, ws, wd, p, pr, best[0].idx, pts);
  }

  function showHoverTip(e, t, ws, wd, p, pr, nearestIdx, pts) {
    if (!hoverTooltipEl) return;
    var tSign = t > 0 ? "+" : "";
    var dirArrow = '<span style="display:inline-block;transform:rotate(' + wd + 'deg)">↑</span>';
    var html =
      '<div style="font-weight:600;font-size:13px">' + tSign + t.toFixed(1) + '°C</div>' +
      '<div>Ветер: ' + ws.toFixed(1) + ' м/с ' + dirArrow + ' ' + wind_dir_text(wd) + '</div>' +
      '<div>Давл: ' + pr.toFixed(1) + ' мм рт.ст.</div>';
    if (p > 0.05) html += '<div>Осадки: ' + p.toFixed(2) + ' мм/6ч</div>';
    var d = pts[nearestIdx];
    html += '<div style="color:#7889a0;font-size:10px;margin-top:3px">' +
      e.latlng.lat.toFixed(2) + '°, ' + e.latlng.lng.toFixed(2) + '° · ~' +
      haversineKm(e.latlng.lat, e.latlng.lng, d.lat, d.lon).toFixed(0) + ' км до точки сетки</div>';
    hoverTooltipEl.innerHTML = html;
    hoverTooltipEl.style.display = "block";
    var pt = e.containerPoint;
    hoverTooltipEl.style.left = pt.x + "px";
    hoverTooltipEl.style.top = pt.y + "px";
  }

  function haversineKm(lat1, lon1, lat2, lon2) {
    var R = 6371;
    var dLat = (lat2 - lat1) * Math.PI / 180;
    var dLon = (lon2 - lon1) * Math.PI / 180;
    var a = Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon/2) * Math.sin(dLon/2);
    return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  function wind_dir_text(deg) {
    var dirs = ["С","СВ","В","ЮВ","Ю","ЮЗ","З","СЗ"];
    return dirs[Math.round(((deg % 360) + 360) % 360 / 45) % 8];
  }

  function initOverlays() {
    if (!overlayMeta) return;
    var bounds = overlayMeta.bounds;
    ["temp", "wind", "precip", "pressure"].forEach(function(layer) {
      var cls = (layer === "precip") ? "leaflet-precip-nearest" : "";
      overlayLayers[layer] = L.imageOverlay("", bounds, {
        opacity: 0.9, interactive: false, zIndex: 200, className: cls
      });
    });
  }

  function updateOverlays(step) {
    if (!overlayMeta) return;
    // Cache-bust keyed to overlay generation time (meta.generated_at) so a
    // re-rendered overlay set forces browsers to refetch even when the
    // forecast JSON itself hasn't changed.
    var bust = (overlayMeta && overlayMeta.generated_at)
      ? "?v=" + encodeURIComponent(overlayMeta.generated_at)
      : (forecastData && forecastData.generated_at
        ? "?v=" + encodeURIComponent(forecastData.generated_at)
        : "?v=" + Date.now());
    ["temp", "wind", "precip", "pressure"].forEach(function(layer) {
      var url = OVERLAY_BASE + "/" + layer + "_" + step + ".png" + bust;
      overlayLayers[layer].setUrl(url);
      if (layerState[layer]) {
        if (!map.hasLayer(overlayLayers[layer])) overlayLayers[layer].addTo(map);
      } else {
        if (map.hasLayer(overlayLayers[layer])) map.removeLayer(overlayLayers[layer]);
      }
    });
    if (layerState.wind) drawWindArrows(step); else clearWindArrows();
  }

  function clearWindArrows() {
    windArrowMarkers.forEach(function(m) { m.remove(); });
    windArrowMarkers = [];
  }

  function drawWindArrows(step) {
    clearWindArrows();
    if (!overlayMeta || !overlayMeta.wind_arrows) return;
    var arrows = overlayMeta.wind_arrows[String(step)];
    if (!arrows) return;
    var wsMax = overlayMeta.ranges.wind[1];

    arrows.forEach(function(a) {
      var ws = a.ws, wd = a.wd;
      var size = 14 + (ws / wsMax) * 16;
      var f = Math.min(ws / wsMax, 1);
      var r, g, b;
      if (f < 0.2) { r = lerp(77,153,f*5); g = lerp(191,230,f*5); b = lerp(77,51,f*5); }
      else if (f < 0.5) { r = lerp(153,255,(f-0.2)/0.3); g = lerp(230,217,(f-0.2)/0.3); b = lerp(51,26,(f-0.2)/0.3); }
      else if (f < 0.75) { r = lerp(255,217,(f-0.5)/0.25); g = lerp(217,38,(f-0.5)/0.25); b = lerp(26,38,(f-0.5)/0.25); }
      else { r = lerp(217,140,(f-0.75)/0.25); g = lerp(38,0,(f-0.75)/0.25); b = lerp(38,0,(f-0.75)/0.25); }
      var color = "rgb(" + Math.round(r) + "," + Math.round(g) + "," + Math.round(b) + ")";
      var icon = L.divIcon({
        className: "",
        html: '<div style="transform:rotate(' + wd + 'deg);font-size:' + size +
              'px;color:' + color + ';text-shadow:0 0 4px rgba(0,0,0,0.9);line-height:1;font-weight:bold">↓</div>',
        iconSize: [size, size], iconAnchor: [size/2, size/2]
      });
      var m = L.marker([a.lat, a.lon], { icon: icon, interactive: false, zIndex: 300 }).addTo(map);
      windArrowMarkers.push(m);
    });
  }

  function setupLayerToggles() {
    document.querySelectorAll(".layer-btn").forEach(function(btn) {
      btn.addEventListener("click", function() {
        var layer = btn.dataset.layer;
        layerState[layer] = !layerState[layer];
        btn.classList.toggle("on", layerState[layer]);

        if (layer === "points") {
          mapMarkers.forEach(function(m) {
            if (layerState.points) { if (!map.hasLayer(m)) m.addTo(map); }
            else { if (map.hasLayer(m)) map.removeLayer(m); }
          });
        } else if (layer === "stations") {
          stationMarkers.forEach(function(m) {
            if (layerState.stations) { if (!map.hasLayer(m)) m.addTo(map); }
            else { if (map.hasLayer(m)) map.removeLayer(m); }
          });
        } else {
          updateOverlays(currentStep);
        }
        updateLegend();
      });
    });
  }

  function updateLegend() {
    var legendEl = document.getElementById("map-legend");
    if (!overlayMeta) {
      var tMinS = tMin > 0 ? "+" + tMin : "" + tMin;
      var tMaxS = tMax > 0 ? "+" + tMax : "" + tMax;
      legendEl.innerHTML =
        "<span>" + tMinS + "°C</span><div class=\"legend-bar\"></div><span>" + tMaxS + "°C</span>";
      return;
    }
    var r = overlayMeta.ranges;
    var items = [];
    if (layerState.temp) items.push(
      '<span>' + r.temp[0].toFixed(0) + '°</span>' +
      '<div class="legend-bar" style="background:linear-gradient(90deg,#1e2896,#3264be,#50a5dc,#8cd7e6,#ffeb78,#ffaa46,#f05a32,#c81e1e)"></div>' +
      '<span>' + r.temp[1].toFixed(0) + '°C</span>'
    );
    if (layerState.wind) items.push(
      '<span>0</span>' +
      '<div class="legend-bar" style="background:linear-gradient(90deg,#4dc04d,#99e633,#ffda1a,#ff801a,#d92626,#8c0000)"></div>' +
      '<span>' + r.wind[1].toFixed(0) + ' м/с</span>'
    );
    if (layerState.precip) items.push(
      '<span>0</span>' +
      '<div class="legend-bar" style="background:linear-gradient(90deg,rgba(180,217,255,0),#5999f2,#2666d9,#0d33b3,#050d73)"></div>' +
      '<span>' + r.precip[1].toFixed(1) + ' мм</span>'
    );
    if (layerState.pressure) items.push(
      '<span>' + r.pressure[0].toFixed(0) + '</span>' +
      '<div class="legend-bar" style="background:linear-gradient(90deg,#1a4d80,#268c8c,#4dc080,#b3d94d,#f2c033,#f28026)"></div>' +
      '<span>' + r.pressure[1].toFixed(0) + ' мм рт.</span>'
    );
    if (layerState.points) items.push(
      '<span style="color:var(--text-dim)">● ' + (forecastData ? forecastData.n_map_points : '~3000') + ' точек сетки GNN</span>'
    );
    if (layerState.stations) items.push(
      '<span style="color:var(--text-dim)">★ 689 станций (постпроц v3)</span>'
    );
    legendEl.innerHTML = items.join('<span style="margin:0 0.5rem;color:var(--border)">|</span>');
  }

  function renderMapMarkers(data, step) {
    for (var i = 0; i < mapMarkers.length; i++) map.removeLayer(mapMarkers[i]);
    mapMarkers = [];
    if (!data.grid_points || data.grid_points.length === 0) return;
    for (var i = 0; i < data.grid_points.length; i++) {
      var pt = data.grid_points[i];
      var sd = pt.steps[step]; if (!sd) continue;
      var tSign = sd.t > 0 ? "+" : "";
      var arrow = windArrowChar(sd.wd);

      var marker = L.circleMarker([pt.lat, pt.lon], {
        radius: 3, fillColor: "rgba(79,195,247,0.5)", fillOpacity: 0.5,
        color: "#4fc3f7", weight: 0.6, opacity: 0.6,
      });
      var tip = "<b>" + tSign + sd.t + "°C</b> &nbsp; " + sd.ws + " м/с " + arrow;
      tip += " &nbsp; " + sd.pr + " мм рт.ст.";
      if (sd.p > 0.05) { tip += "<br>" + sd.p + " мм осадки"; if (sd.pi) tip += " " + sd.pi; }
      marker.bindTooltip(tip, { className: "forecast-tip", direction: "top", offset: [0, -4] });

      if (layerState.points) marker.addTo(map);
      mapMarkers.push(marker);
    }
  }

  function renderStationMarkers(data, step) {
    for (var i = 0; i < stationMarkers.length; i++) map.removeLayer(stationMarkers[i]);
    stationMarkers = [];
    if (!data.stations) return;
    for (var i = 0; i < data.stations.length; i++) {
      var pt = data.stations[i];
      var sd = pt.steps[step]; if (!sd) continue;
      var tSign = sd.t > 0 ? "+" : "";
      var col = tempColor(sd.t);
      var arrow = windArrowChar(sd.wd);

      var marker = L.circleMarker([pt.lat, pt.lon], {
        radius: 6, fillColor: col, fillOpacity: 0.95,
        color: "#0a1018", weight: 1.4, opacity: 1,
      });
      var tip = "<b>" + pt.name + "</b><br><b>" + tSign + sd.t + "°C</b> &nbsp; " + sd.ws + " м/с " + arrow;
      if (sd.wg > sd.ws + 0.5) tip += " (пор. " + sd.wg + ")";
      tip += " &nbsp; " + sd.pr + " мм рт.ст.";
      if (sd.p > 0.05) { tip += "<br>" + sd.p + " мм осадки"; if (sd.pi) tip += " " + sd.pi; if (sd.pt) tip += " " + sd.pt; }
      marker.bindTooltip(tip, { className: "forecast-tip", direction: "top", offset: [0, -6] });

      if (layerState.stations) marker.addTo(map);
      stationMarkers.push(marker);
    }
  }

  function setupMapSlider(data) {
    var slider = document.getElementById("map-step-slider");
    var label = document.getElementById("map-step-label");
    var summary = data.summary_region || data.summary_city || data.summary_core || [];
    slider.max = Math.max(0, summary.length - 1);
    slider.value = 0;
    currentStep = 0;

    function update() {
      var step = parseInt(slider.value);
      currentStep = step;
      var h = (step + 1) * 6;
      var timeStr = summary[step] ? formatMskTime(summary[step].valid_time_utc) : "";
      label.textContent = "+" + h + "ч" + (timeStr ? " (" + timeStr + ")" : "");
      renderMapMarkers(forecastData, step);
      renderStationMarkers(forecastData, step);
      updateOverlays(step);
      updateLegend();
    }
    slider.removeEventListener("input", slider._handler);
    slider._handler = update;
    slider.addEventListener("input", update);
    update();
  }

  function init() {
    setupScopeToggle();
    setupLayerToggles();
    Promise.all([fetchForecast(), fetchOverlayMeta()]).then(function(results) {
      var data = results[0]; if (!data) return;
      forecastData = data;
      updateTempRange(forecastData);
      renderCurrentCard(forecastData);
      renderTable(forecastData);
      renderChart(forecastData);
      initMap();
      initOverlays();
      updateLegend();
      setupMapSlider(forecastData);
    });

    setInterval(function () {
      Promise.all([fetchForecast(), fetchOverlayMeta()]).then(function(results) {
        var newData = results[0];
        if (newData && forecastData && newData.generated_at !== forecastData.generated_at) {
          forecastData = newData;
          initOverlays();
          renderAll(forecastData);
          setupMapSlider(forecastData);
        }
      });
    }, REFRESH_INTERVAL_MS);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
