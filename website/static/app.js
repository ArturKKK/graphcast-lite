/* GraphCast-lite Weather — Frontend Logic */
(function () {
  "use strict";

  // ── Constants ──
  const REFRESH_INTERVAL_MS = 10 * 60 * 1000;
  const KRSK_OFFSET_H = 7;

  // Dynamic temperature range — updated from data
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

  // RdYlBu diverging
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

  // ── State ──
  var forecastData = null;
  var map = null;
  var mapMarkers = [];
  var currentStep = 0;
  var tempChart = null;
  var currentScope = "core"; // "core", "city", or "region"

  // Overlay state
  var overlayMeta = null;
  var overlayLayers = {};   // { temp: L.imageOverlay, ... }
  var windArrowMarkers = [];
  var layerState = { temp: true, wind: false, precip: false, pressure: false, points: true };

  // ── Scope helper ──
  function getActiveSummary(data) {
    if (currentScope === "core") return data.summary_core || data.summary_city || data.summary || [];
    if (currentScope === "city") return data.summary_city || data.summary || [];
    return data.summary_region || data.summary || [];
  }

  function getActivePointCount(data) {
    if (currentScope === "core") return data.n_core_points || 0;
    if (currentScope === "city") return data.n_city_points || 0;
    return data.n_region_points || data.n_city_points || 0;
  }

  // ── Utilities ──
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

  function weatherEmoji(t2m, precip, hourKrsk) {
    var isNight = hourKrsk < 6 || hourKrsk >= 22;
    if (t2m <= -15) return "\uD83E\uDD76";
    if (precip > 2) return t2m <= 0 ? "\u2744\uFE0F" : "\uD83C\uDF27\uFE0F";
    if (precip > 0.3) return t2m <= 0 ? "\uD83C\uDF28\uFE0F" : "\uD83C\uDF26\uFE0F";
    if (isNight) return "\uD83C\uDF19";
    if (t2m >= 25) return "\u2600\uFE0F";
    return "\uD83C\uDF24\uFE0F";
  }

  function formatKrskTime(utcStr) {
    var d = new Date(utcStr);
    var krsk = new Date(d.getTime() + KRSK_OFFSET_H * 3600000);
    var day = ("0" + krsk.getUTCDate()).slice(-2);
    var mon = ("0" + (krsk.getUTCMonth() + 1)).slice(-2);
    var hh = ("0" + krsk.getUTCHours()).slice(-2);
    var mm = ("0" + krsk.getUTCMinutes()).slice(-2);
    return day + "." + mon + " " + hh + ":" + mm;
  }

  function getKrskHour(utcStr) {
    var d = new Date(utcStr);
    var krsk = new Date(d.getTime() + KRSK_OFFSET_H * 3600000);
    return krsk.getUTCHours();
  }

  function tempClass(t) {
    if (t >= 20) return "temp-warm";
    if (t <= 0) return "temp-cold";
    return "temp-mild";
  }

  function relativeTime(isoStr) {
    var diff = Date.now() - new Date(isoStr).getTime();
    var mins = Math.floor(diff / 60000);
    if (mins < 1) return "\u0442\u043E\u043B\u044C\u043A\u043E \u0447\u0442\u043E";
    if (mins < 60) return mins + " \u043C\u0438\u043D. \u043D\u0430\u0437\u0430\u0434";
    var hrs = Math.floor(mins / 60);
    if (hrs < 24) return hrs + " \u0447. \u043D\u0430\u0437\u0430\u0434";
    return Math.floor(hrs / 24) + " \u0434\u043D. \u043D\u0430\u0437\u0430\u0434";
  }

  function nextUpdateText(lastCycleStr) {
    var lc = new Date(lastCycleStr);
    var nextCycle = new Date(lc.getTime() + 6 * 3600000 + 90 * 60000);
    var diff = nextCycle.getTime() - Date.now();
    if (diff <= 0) return "\u043E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0435 \u043E\u0436\u0438\u0434\u0430\u0435\u0442\u0441\u044F";
    var hrs = Math.floor(diff / 3600000);
    var mins = Math.floor((diff % 3600000) / 60000);
    if (hrs > 0) return "~" + hrs + "\u0447 " + mins + "\u043C";
    return "~" + mins + "\u043C";
  }

  // ── Data Fetching ──
  function fetchForecast() {
    return fetch("/api/forecast")
      .then(function (resp) {
        if (!resp.ok) { showError("HTTP " + resp.status); return null; }
        return resp.json();
      })
      .then(function (data) {
        if (!data) return null;
        if (data.error) { showError(data.error); return null; }
        return data;
      })
      .catch(function (e) { showError(e.message); return null; });
  }

  function showError(msg) {
    var el = document.getElementById("status-text");
    if (el) {
      el.textContent = msg;
      el.style.color = "var(--accent-red)";
    }
    console.error(msg);
  }

  // ── Scope toggle ──
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

  // ── Renderers ──

  function renderAll(data) {
    updateTempRange(data);
    renderStatus(data);
    renderCurrentCard(data);
    renderTable(data);
    renderChart(data);
    updateLegend();
    renderMapMarkers(data, currentStep);
    updateOverlays(currentStep);
  }

  function renderStatus(data) {
    // Status bar removed from UI — skip
  }

  function renderCurrentCard(data) {
    var card = document.getElementById("current-card");
    var summary = getActiveSummary(data);
    var s = summary[0];
    if (!s) return;

    card.style.display = "";
    var nPts = getActivePointCount(data);
    var scopeNames = {core: "\u0433\u043E\u0440\u043E\u0434\u0443", city: "\u0433\u043E\u0440\u043E\u0434\u0443 + \u043E\u043A\u0440.", region: "\u0440\u0435\u0433\u0438\u043E\u043D\u0443"};
    var scopeLabel = "\u0441\u0440\u0435\u0434\u043D\u0435\u0435 \u043F\u043E " + scopeNames[currentScope] + " (" + nPts + " \u0442\u043E\u0447\u0435\u043A)";
    document.getElementById("city-avg-badge").textContent = scopeLabel;

    document.getElementById("current-time").textContent = formatKrskTime(s.valid_time_utc) + " (\u043A\u0440\u0430\u0441\u043D\u043E\u044F\u0440\u0441\u043A\u043E\u0435 \u0432\u0440\u0435\u043C\u044F)";
    document.getElementById("current-temp").textContent = s.t2m_celsius > 0 ? "+" + s.t2m_celsius : s.t2m_celsius;

    // Temperature range
    var rangeEl = document.getElementById("current-temp-range");
    if (rangeEl && s.t2m_min !== undefined) {
      var rMin = s.t2m_min > 0 ? "+" + s.t2m_min : "" + s.t2m_min;
      var rMax = s.t2m_max > 0 ? "+" + s.t2m_max : "" + s.t2m_max;
      rangeEl.textContent = "\u043E\u0442 " + rMin + " \u0434\u043E " + rMax + "\u00B0C";
    }

    var emoji = s.precip_type_icon || weatherEmoji(s.t2m_celsius, s.precip_mm, getKrskHour(s.valid_time_utc));
    document.getElementById("current-emoji").textContent = emoji;

    document.getElementById("current-wind").textContent = s.wind_speed_ms + " \u043C/\u0441 " + s.wind_direction_text;
    document.getElementById("current-wind-arrow").textContent = windArrowChar(s.wind_direction_deg);
    document.getElementById("current-wind-arrow").style.transform = "rotate(" + s.wind_direction_deg + "deg)";
    document.getElementById("current-pressure").textContent = s.pressure_mmhg + " \u043C\u043C \u0440\u0442.\u0441\u0442.";
    document.getElementById("current-precip").textContent = s.precip_mm + " \u043C\u043C";

    // Gusts
    var gustEl = document.getElementById("current-gust");
    gustEl.textContent = (s.wind_gust_ms && s.wind_gust_ms > s.wind_speed_ms + 0.5)
      ? "\u043F\u043E\u0440\u044B\u0432\u044B " + s.wind_gust_ms + " \u043C/\u0441" : "";

    // Precip intensity
    var ptEl = document.getElementById("current-precip-type");
    ptEl.textContent = s.precip_intensity_text || "";
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
      var names = {core: "\u0433\u043E\u0440\u043E\u0434", city: "\u0433\u043E\u0440\u043E\u0434 + \u043E\u043A\u0440.", region: "\u0440\u0435\u0433\u0438\u043E\u043D"};
      scopeEl.textContent = names[currentScope];
    }

    for (var i = 0; i < summary.length; i++) {
      var s = summary[i];
      var hr = getKrskHour(s.valid_time_utc);
      var emoji = s.precip_type_icon || weatherEmoji(s.t2m_celsius, s.precip_mm, hr);
      var tSign = s.t2m_celsius > 0 ? "+" : "";

      // Wind with gust
      var windCell = "" + s.wind_speed_ms;
      if (s.wind_gust_ms && s.wind_gust_ms > s.wind_speed_ms + 0.5) {
        windCell += '<span class="gust-table"> (\u043F\u043E\u0440. ' + s.wind_gust_ms + ')</span>';
      }

      // Precip with intensity
      var precipCell = "" + s.precip_mm;
      if (s.precip_intensity_text) {
        precipCell += ' <span class="precip-label">' + s.precip_intensity_text + '</span>';
      }

      var tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + formatKrskTime(s.valid_time_utc) + "</td>" +
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
    var labels = summary.map(function (s) { return formatKrskTime(s.valid_time_utc); });
    var temps = summary.map(function (s) { return s.t2m_celsius; });

    if (tempChart) tempChart.destroy();

    tempChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [{
          label: "\u0422\u0435\u043C\u043F\u0435\u0440\u0430\u0442\u0443\u0440\u0430 (\u00B0C)",
          data: temps,
          borderColor: "#4fc3f7",
          backgroundColor: "rgba(79,195,247,0.1)",
          borderWidth: 2,
          pointBackgroundColor: temps.map(function (t) { return tempColor(t); }),
          pointRadius: 5,
          pointHoverRadius: 7,
          fill: true,
          tension: 0.3,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                var s = summary[ctx.dataIndex];
                var lines = [
                  "T: " + s.t2m_celsius + "\u00B0C",
                ];
                if (s.t2m_min !== undefined) {
                  lines.push("\u0414\u0438\u0430\u043F\u0430\u0437\u043E\u043D: " + s.t2m_min + "..." + s.t2m_max + "\u00B0C");
                }
                lines.push(
                  "\u0412\u0435\u0442\u0435\u0440: " + s.wind_speed_ms + " \u043C/\u0441 " + s.wind_direction_text,
                  "\u0414\u0430\u0432\u043B: " + s.pressure_mmhg + " \u043C\u043C",
                );
                return lines;
              },
            },
          },
        },
        scales: {
          x: { ticks: { color: "#8899aa", maxRotation: 45 }, grid: { color: "rgba(42,58,78,0.3)" } },
          y: {
            ticks: {
              color: "#8899aa",
              callback: function (v) { return (v > 0 ? "+" + v : v) + "\u00B0"; },
            },
            grid: { color: "rgba(42,58,78,0.3)" },
          },
        },
      },
    });
  }

  // ── Map ──

  function initMap() {
    map = L.map("map", {
      center: [56.01, 92.87],
      zoom: 8,
      zoomControl: true,
      attributionControl: false,
    });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 14,
    }).addTo(map);

    // Krasnoyarsk marker
    L.marker([56.01, 92.87], {
      icon: L.divIcon({
        className: "",
        html: '<div style="color:#4fc3f7;font-size:18px;text-shadow:0 0 6px #000;line-height:1">★</div>',
        iconSize: [20, 20], iconAnchor: [10, 10]
      }),
      zIndex: 1000
    }).addTo(map).bindTooltip("Красноярск", { direction: "right", className: "forecast-tip" });

    document.getElementById("map-legend").innerHTML = "";
  }

  function fetchOverlayMeta() {
    return fetch("/static/overlays/meta.json")
      .then(function(r) { return r.json(); })
      .then(function(meta) { overlayMeta = meta; })
      .catch(function() { overlayMeta = null; });
  }

  function initOverlays() {
    if (!overlayMeta) return;
    var bounds = overlayMeta.bounds;
    ["temp", "wind", "precip", "pressure"].forEach(function(layer) {
      overlayLayers[layer] = L.imageOverlay("", bounds, {
        opacity: 0.9, interactive: false, zIndex: 200
      });
    });
  }

  function updateOverlays(step) {
    if (!overlayMeta) return;
    ["temp", "wind", "precip", "pressure"].forEach(function(layer) {
      var url = "/static/overlays/" + layer + "_" + step + ".png";
      overlayLayers[layer].setUrl(url);
      if (layerState[layer]) {
        if (!map.hasLayer(overlayLayers[layer])) overlayLayers[layer].addTo(map);
      } else {
        if (map.hasLayer(overlayLayers[layer])) map.removeLayer(overlayLayers[layer]);
      }
    });
    // Wind arrows
    if (layerState.wind) {
      drawWindArrows(step);
    } else {
      clearWindArrows();
    }
  }

  // ── Wind arrows ──
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
      // Color by speed
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

  // ── Layer toggles ──
  function setupLayerToggles() {
    document.querySelectorAll(".layer-btn").forEach(function(btn) {
      btn.addEventListener("click", function() {
        var layer = btn.dataset.layer;
        layerState[layer] = !layerState[layer];
        btn.classList.toggle("on", layerState[layer]);

        if (layer === "points") {
          // Show/hide grid point markers
          mapMarkers.forEach(function(m) {
            if (layerState.points) { if (!map.hasLayer(m)) m.addTo(map); }
            else { if (map.hasLayer(m)) map.removeLayer(m); }
          });
        } else {
          updateOverlays(currentStep);
        }
      });
    });
  }

  function updateLegend() {
    var legendEl = document.getElementById("map-legend");
    if (!overlayMeta) {
      var tMinS = tMin > 0 ? "+" + tMin : "" + tMin;
      var tMaxS = tMax > 0 ? "+" + tMax : "" + tMax;
      legendEl.innerHTML =
        "<span>" + tMinS + "\u00B0C</span><div class=\"legend-bar\"></div><span>" + tMaxS + "\u00B0C</span>";
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
      '<span style="color:var(--text-dim)">● точки сетки GNN (0.25°) — наведите для деталей</span>'
    );
    legendEl.innerHTML = items.join('<span style="margin:0 0.5rem;color:var(--border)">|</span>');
  }

  function renderMapMarkers(data, step) {
    for (var i = 0; i < mapMarkers.length; i++) map.removeLayer(mapMarkers[i]);
    mapMarkers = [];
    if (!data.grid_points || data.grid_points.length === 0) return;

    for (var i = 0; i < data.grid_points.length; i++) {
      var pt = data.grid_points[i];
      var sd = pt.steps[step];
      if (!sd) continue;

      var tSign = sd.t > 0 ? "+" : "";
      var arrow = windArrowChar(sd.wd);

      var marker = L.circleMarker([pt.lat, pt.lon], {
        radius: 5,
        fillColor: "rgba(79,195,247,0.5)",
        fillOpacity: 0.5,
        color: "#4fc3f7",
        weight: 1,
        opacity: 0.6,
        zIndexOffset: 400,
      });

      var tip = "<b>" + tSign + sd.t + "\u00B0C</b>";
      tip += " &nbsp; " + sd.ws + " \u043C/\u0441 " + arrow;
      if (sd.wg > sd.ws + 0.5) tip += " (\u043F\u043E\u0440. " + sd.wg + ")";
      tip += " &nbsp; " + sd.pr + " \u043C\u043C \u0440\u0442.\u0441\u0442.";
      if (sd.p > 0.05) {
        tip += "<br>" + sd.p + " \u043C\u043C \u043E\u0441\u0430\u0434\u043A\u0438";
        if (sd.pi) tip += " " + sd.pi;
        if (sd.pt) tip += " " + sd.pt;
      }

      marker.bindTooltip(tip, { className: "forecast-tip", direction: "top", offset: [0, -6] });

      if (layerState.points) marker.addTo(map);
      mapMarkers.push(marker);
    }
  }

  function setupMapSlider(data) {
    var slider = document.getElementById("map-step-slider");
    var label = document.getElementById("map-step-label");
    var summary = getActiveSummary(data);
    slider.max = summary.length - 1;
    slider.value = 0;
    currentStep = 0;

    function update() {
      var step = parseInt(slider.value);
      currentStep = step;
      var h = (step + 1) * 6;
      var timeStr = summary[step] ? formatKrskTime(summary[step].valid_time_utc) : "";
      label.textContent = "+" + h + "\u0447" + (timeStr ? " (" + timeStr + ")" : "");
      renderMapMarkers(forecastData, step);
      updateOverlays(step);
      updateLegend();
    }

    slider.removeEventListener("input", slider._handler);
    slider._handler = update;
    slider.addEventListener("input", update);
    update();
  }

  // ── Yandex Weather Comparison ──

  function fetchYandexWeather() {
    return fetch("/api/yandex")
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
  }

  function renderYandexWidget(yData) {
    var container = document.getElementById("yandex-widget");
    if (!container || !yData || yData.error) {
      if (container) container.innerHTML = '<p class="yandex-error">Данные Яндекс.Погоды недоступны</p>';
      return;
    }

    var c = yData.current || {};
    var daily = yData.daily || [];

    // Current weather card
    var html = '<div class="yandex-current">';
    html += '<div class="yandex-current-main">';
    html += '<span class="yandex-temp">' + (c.temp > 0 ? '+' : '') + c.temp + '°</span>';
    html += '<span class="yandex-condition">' + (c.condition || '') + '</span>';
    html += '</div>';
    html += '<div class="yandex-current-details">';
    html += '<span>Ощущается: ' + (c.feels_like > 0 ? '+' : '') + (c.feels_like || '') + '°</span>';
    html += '<span>Ветер: ' + (c.wind_speed || '') + ' м/с, ' + (c.wind_dir || '') + '</span>';
    html += '<span>Давление: ' + (c.pressure_mmhg || '') + ' мм рт.ст.</span>';
    html += '<span>Влажность: ' + (c.humidity || '') + '%</span>';
    html += '</div>';
    html += '</div>';

    // Daily forecast table
    if (daily.length > 0) {
      html += '<table class="yandex-daily">';
      html += '<colgroup><col class="day-label"><col><col><col><col></colgroup>';
      html += '<thead><tr><th></th><th>Утро</th><th>День</th><th>Вечер</th><th>Ночь</th></tr></thead>';
      html += '<tbody>';
      for (var i = 0; i < daily.length; i++) {
        var d = daily[i];
        var p = d.parts || {};
        html += '<tr>';
        html += '<td class="yandex-day-name">' + d.day + '<br><small>' + d.date + '</small></td>';
        var parts = ['утром', 'днём', 'вечером', 'ночью'];
        for (var j = 0; j < parts.length; j++) {
          var part = p[parts[j]];
          if (part) {
            var t = part.temp;
            var tc = t <= -20 ? '#88f' : t <= -10 ? '#aaf' : t <= 0 ? '#cce' : t <= 10 ? '#ffa' : t <= 20 ? '#fa0' : '#f44';
            html += '<td><span class="yandex-day-temp" style="color:' + tc + '">' + (t > 0 ? '+' : '') + t + '°</span>';
            html += '<span class="yandex-day-wind">' + part.wind_speed + ' м/с</span></td>';
          } else {
            html += '<td>—</td>';
          }
        }
        html += '</tr>';
      }
      html += '</tbody></table>';
    }

    // Comparison with our forecast
    if (forecastData) {
      var summary = getActiveSummary(forecastData);
      if (summary.length > 0 && daily.length > 0) {
        html += '<div class="yandex-comparison">';
        html += '<h4>Сравнение с нашим прогнозом (ближайший шаг)</h4>';
        var ourT = summary[0].t2m_celsius;
        var ourW = summary[0].wind_speed;
        var ourP = summary[0].pressure_mmhg;
        var yT = c.temp;
        var yW = c.wind_speed;
        var yP = c.pressure_mmhg;
        html += '<table class="yandex-compare-table">';
        html += '<thead><tr><th></th><th>GraphCast-lite</th><th>Яндекс.Погода</th><th>Δ</th></tr></thead>';
        html += '<tbody>';
        if (ourT !== undefined && yT !== undefined) {
          var dT = (ourT - yT).toFixed(1);
          html += '<tr><td>Температура</td><td>' + ourT.toFixed(1) + '°</td><td>' + yT + '°</td><td>' + (dT > 0 ? '+' : '') + dT + '°</td></tr>';
        }
        if (ourW !== undefined && yW !== undefined) {
          var dW = (ourW - yW).toFixed(1);
          html += '<tr><td>Ветер</td><td>' + ourW.toFixed(1) + ' м/с</td><td>' + yW + ' м/с</td><td>' + (dW > 0 ? '+' : '') + dW + '</td></tr>';
        }
        if (ourP !== undefined && yP !== undefined) {
          var dP = (ourP - yP).toFixed(1);
          html += '<tr><td>Давление</td><td>' + ourP.toFixed(0) + '</td><td>' + yP + '</td><td>' + (dP > 0 ? '+' : '') + dP + '</td></tr>';
        }
        html += '</tbody></table>';
        html += '</div>';
      }
    }

    html += '<div class="yandex-footer">';
    if (yData._stale) html += '<span class="yandex-stale">⚠ кэшированные данные</span> ';
    html += '<small>Обновлено: ' + (yData.scraped_at || '—') + '</small> ';
    html += '<a href="' + yData.url + '" target="_blank" rel="noopener">Открыть на Яндексе →</a>';
    html += '</div>';

    container.innerHTML = html;
  }

  // ── Main ──

  function init() {
    setupScopeToggle();
    setupLayerToggles();

    Promise.all([fetchForecast(), fetchOverlayMeta()]).then(function(results) {
      var data = results[0];
      if (!data) return;
      forecastData = data;

      updateTempRange(forecastData);
      renderStatus(forecastData);
      renderCurrentCard(forecastData);
      renderTable(forecastData);
      renderChart(forecastData);

      initMap();
      initOverlays();
      updateLegend();
      setupMapSlider(forecastData);

    });

    // Fetch Yandex weather independently
    fetchYandexWeather().then(renderYandexWidget);

    // Auto-refresh
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
