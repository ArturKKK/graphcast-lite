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
  var currentScope = "city"; // "city" or "region"

  // ── Scope helper ──
  function getActiveSummary(data) {
    return currentScope === "city"
      ? (data.summary_city || data.summary || [])
      : (data.summary_region || data.summary || []);
  }

  function getActivePointCount(data) {
    return currentScope === "city"
      ? (data.n_city_points || 0)
      : (data.n_region_points || data.n_city_points || 0);
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
    var scopeLabel = currentScope === "city"
      ? "\u0441\u0440\u0435\u0434\u043D\u0435\u0435 \u043F\u043E \u0433\u043E\u0440\u043E\u0434\u0443 (" + data.n_city_points + " \u0442\u043E\u0447\u0435\u043A)"
      : "\u0441\u0440\u0435\u0434\u043D\u0435\u0435 \u043F\u043E \u0440\u0435\u0433\u0438\u043E\u043D\u0443 (" + data.n_region_points + " \u0442\u043E\u0447\u0435\u043A)";
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
    if (scopeEl) scopeEl.textContent = currentScope === "city" ? "\u0433\u043E\u0440\u043E\u0434" : "\u0440\u0435\u0433\u0438\u043E\u043D";

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

    document.getElementById("map-legend").innerHTML = "";
  }

  function updateLegend() {
    var legendEl = document.getElementById("map-legend");
    var tMinS = tMin > 0 ? "+" + tMin : "" + tMin;
    var tMaxS = tMax > 0 ? "+" + tMax : "" + tMax;
    legendEl.innerHTML =
      "<span>" + tMinS + "\u00B0C</span><div class=\"legend-bar\"></div><span>" + tMaxS + "\u00B0C</span>" +
      '<span style="margin-left:1rem;color:var(--text-dim)">\u25CF \u0442\u043E\u0447\u043A\u0438 \u0441\u0435\u0442\u043A\u0438 GNN (0.25\u00B0)</span>';
  }

  function renderMapMarkers(data, step) {
    for (var i = 0; i < mapMarkers.length; i++) map.removeLayer(mapMarkers[i]);
    mapMarkers = [];
    if (!data.grid_points || data.grid_points.length === 0) return;

    var summary = getActiveSummary(data);
    var validTime = summary[step] ? formatKrskTime(summary[step].valid_time_utc) : "+" + ((step + 1) * 6) + "\u0447";

    for (var i = 0; i < data.grid_points.length; i++) {
      var pt = data.grid_points[i];
      var sd = pt.steps[step];
      if (!sd) continue;

      var color = tempColor(sd.t);
      var tSign = sd.t > 0 ? "+" : "";
      var arrow = windArrowChar(sd.wd);

      var marker = L.circleMarker([pt.lat, pt.lon], {
        radius: 10,
        fillColor: color,
        fillOpacity: 0.92,
        color: "#1a2332",
        weight: 2.5,
        opacity: 1,
      }).addTo(map);

      var tip = "<b>" + tSign + sd.t + "\u00B0C</b> &nbsp; " + sd.ws + " \u043C/\u0441 " + arrow;
      if (sd.wg > sd.ws + 0.5) tip += " (\u043F. " + sd.wg + ")";
      tip += " &nbsp; " + sd.pr + " \u043C\u043C";
      if (sd.p > 0.1 && sd.pi) tip += " &nbsp; " + sd.pi;
      if (sd.p > 0.1 && sd.pt) tip += " " + sd.pt;

      marker.bindTooltip(tip, { className: "forecast-tip", direction: "top", offset: [0, -8] });
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
    }

    slider.removeEventListener("input", slider._handler);
    slider._handler = update;
    slider.addEventListener("input", update);
    update();
  }

  // ── Main ──

  function init() {
    setupScopeToggle();

    fetchForecast().then(function (data) {
      if (!data) return;
      forecastData = data;

      updateTempRange(forecastData);
      renderStatus(forecastData);
      renderCurrentCard(forecastData);
      renderTable(forecastData);
      renderChart(forecastData);

      initMap();
      updateLegend();
      setupMapSlider(forecastData);
    });

    // Auto-refresh
    setInterval(function () {
      fetchForecast().then(function (newData) {
        if (newData && forecastData && newData.generated_at !== forecastData.generated_at) {
          forecastData = newData;
          renderAll(forecastData);
          setupMapSlider(forecastData);
        }
      });
    }, REFRESH_INTERVAL_MS);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
