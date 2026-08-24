const SEVERITY_COLOR = {
  safe: '#2FD4C8', low: '#7FC97F', moderate: '#E4B94E', severe: '#D85A30',
};
const HAZARD_LABEL = {
  flood: 'Flood', landslide: 'Landslide', avalanche: 'Avalanche',
  sandstorm: 'Sandstorm', cyclone: 'Cyclone',
};

const stateSelect = document.getElementById('stateSelect');
const hazardSelect = document.getElementById('hazardSelect');
const yearBlock = document.getElementById('yearBlock');
const yearSlider = document.getElementById('yearSlider');
const yearLabel = document.getElementById('yearLabel');
const monthRow = document.getElementById('monthRow');
const monthSlider = document.getElementById('monthSlider');
const monthLabel = document.getElementById('monthLabel');
const modeHistorical = document.getElementById('modeHistorical');
const modeForecast = document.getElementById('modeForecast');
const statsPanel = document.getElementById('statsPanel');
const cellInfo = document.getElementById('cellInfo');
const loadingOverlay = document.getElementById('loadingOverlay');
const backendWarning = document.getElementById('backendWarning');
const liveDot = document.getElementById('liveDot');
const liveInfo = document.getElementById('liveInfo');
const refreshBtn = document.getElementById('refreshBtn');

document.getElementById('apiBaseShown').textContent = API_BASE;

// ---- map setup ----
const map = L.map('map', { zoomControl: true, worldCopyJump: true }).setView([22.5, 82], 5);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  maxZoom: 18,
}).addTo(map);

let statesGeoJson = null;
let selectedOutlineLayer = null;
let zoneLayer = L.layerGroup().addTo(map);
let stateHazards = {};          // { state: [hazard, ...] } from /api/hazards, filled lazily
let mode = 'historical';        // 'historical' | 'forecast'
let forecastCache = null;       // last /api/forecast response for current state+hazard

async function api(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

// ---- boot ----
Promise.all([
  fetch('india_states.geojson').then(r => r.json()),
  api('/api/states'),
]).then(async ([geo, statesResp]) => {
  statesGeoJson = geo;
  populateStateDropdown(statesResp.states);
  initBoundaries();
  loadingOverlay.classList.add('hidden');
  backendWarning.classList.remove('visible');

  stateSelect.value = statesResp.states.includes('Assam') ? 'Assam' : statesResp.states[0];
  await onStateChange();
  startLivePolling();
}).catch(err => {
  console.error(err);
  loadingOverlay.classList.add('hidden');
  backendWarning.classList.add('visible');
});

function initBoundaries() {
  L.geoJSON(statesGeoJson, {
    style: { className: 'state-outline' },
    onEachFeature: (feature, layer) => {
      layer.on('click', () => {
        const name = feature.properties.name;
        if ([...stateSelect.options].some(o => o.value === name)) {
          stateSelect.value = name;
          onStateChange();
        }
      });
    },
  }).addTo(map);
}

function populateStateDropdown(names) {
  stateSelect.innerHTML = '';
  [...names].sort().forEach(s => {
    const opt = document.createElement('option');
    opt.value = s; opt.textContent = s;
    stateSelect.appendChild(opt);
  });
}

function findStateFeature(name) {
  return statesGeoJson.features.find(f => f.properties.name === name);
}

async function onStateChange() {
  const state = stateSelect.value;

  if (selectedOutlineLayer) map.removeLayer(selectedOutlineLayer);
  const feature = findStateFeature(state);
  if (feature) {
    selectedOutlineLayer = L.geoJSON(feature, { className: 'state-outline-selected' }).addTo(map);
    map.fitBounds(selectedOutlineLayer.getBounds(), { padding: [30, 30] });
  }

  const [hazardsResp, yearsResp] = await Promise.all([
    api(`/api/hazards?state=${encodeURIComponent(state)}`),
    api(`/api/year_range?state=${encodeURIComponent(state)}`),
  ]);
  stateHazards[state] = hazardsResp.hazards;
  populateHazardDropdown(state);

  yearSlider.min = yearsResp.min_year;
  yearSlider.max = yearsResp.max_year;
  yearSlider.value = yearsResp.max_year;
  yearLabel.textContent = yearsResp.max_year;

  await render();
}

function populateHazardDropdown(state) {
  hazardSelect.innerHTML = '';
  (stateHazards[state] || []).forEach(h => {
    const opt = document.createElement('option');
    opt.value = h; opt.textContent = HAZARD_LABEL[h] || h;
    hazardSelect.appendChild(opt);
  });
}

function setMode(next) {
  mode = next;
  modeHistorical.classList.toggle('active', mode === 'historical');
  modeForecast.classList.toggle('active', mode === 'forecast');
  yearBlock.style.display = mode === 'historical' ? 'block' : 'none';
  monthRow.classList.toggle('visible', mode === 'forecast');
  forecastCache = null; // force refetch on mode switch
  render();
}

async function render() {
  const state = stateSelect.value;
  const hazard = hazardSelect.value;
  if (!state || !hazard) return;

  zoneLayer.clearLayers();
  statsPanel.innerHTML = '<span class="empty">Loading…</span>';

  let cells, resolution, liveMeta = null, badgeLabel = null;

  try {
    if (mode === 'historical') {
      const year = Number(yearSlider.value);
      yearLabel.textContent = year;
      const data = await api(`/api/zones?state=${encodeURIComponent(state)}&hazard=${hazard}&year=${year}`);
      cells = data.cells; resolution = data.resolution; liveMeta = data.live;
      badgeLabel = liveMeta ? 'LIVE' : null;
    } else {
      if (!forecastCache || forecastCache.state !== state || forecastCache.hazard !== hazard) {
        forecastCache = await api(`/api/forecast?state=${encodeURIComponent(state)}&hazard=${hazard}&months=6`);
        forecastCache.state = state; forecastCache.hazard = hazard;
      }
      const idx = Number(monthSlider.value);
      const monthData = forecastCache.months[idx];
      monthLabel.textContent = monthData.month_label;
      cells = monthData.cells; resolution = forecastCache.resolution;
      badgeLabel = monthData.live_adjusted ? 'LIVE' : 'FORECAST';
    }
  } catch (err) {
    console.error(err);
    statsPanel.innerHTML = '<span class="empty">Could not reach backend — see the warning at the bottom of the sidebar.</span>';
    return;
  }

  const half = resolution / 2;
  const counts = { safe: 0, low: 0, moderate: 0, severe: 0 };

  cells.forEach(([lat, lon, severity, confidence]) => {
    counts[severity] = (counts[severity] || 0) + 1;
    const bounds = [[lat - half, lon - half], [lat + half, lon + half]];
    const rect = L.rectangle(bounds, {
      className: 'zone-cell',
      color: SEVERITY_COLOR[severity],
      weight: 1,
      fillColor: SEVERITY_COLOR[severity],
      fillOpacity: 0.55,
    });
    rect.bindTooltip(`${severity} · ${Math.round(confidence * 100)}% confidence`, { sticky: true });
    rect.on('click', () => {
      const periodLabel = mode === 'historical' ? yearSlider.value : monthLabel.textContent;
      cellInfo.innerHTML = `
        <div class="stat-row"><span>Location</span><span class="stat-num">${lat.toFixed(2)}, ${lon.toFixed(2)}</span></div>
        <div class="stat-row"><span>State</span><span class="stat-num">${state}</span></div>
        <div class="stat-row"><span>Hazard</span><span class="stat-num">${HAZARD_LABEL[hazard] || hazard}</span></div>
        <div class="stat-row"><span>Predicted severity</span><span class="stat-num" style="color:${SEVERITY_COLOR[severity]}">${severity}</span></div>
        <div class="stat-row"><span>Model confidence</span><span class="stat-num">${Math.round(confidence * 100)}%</span></div>
        <div class="stat-row"><span>Period</span><span class="stat-num">${periodLabel}</span></div>
        ${liveMeta ? `<div class="stat-row"><span>Live signal</span><span class="stat-num">${liveMeta.summary}</span></div>` : ''}
      `;
    });
    zoneLayer.addLayer(rect);
  });

  const total = cells.length || 1;
  const badge = badgeLabel === 'LIVE' ? '<span class="live-badge">LIVE</span>'
    : badgeLabel === 'FORECAST' ? '<span class="forecast-badge">FORECAST</span>' : '';
  statsPanel.innerHTML = cells.length === 0
    ? '<span class="empty">No data for this combination</span>'
    : `<div style="margin-bottom:8px;">${badge}</div>` + ['safe', 'low', 'moderate', 'severe'].map(s => `
        <div class="stat-row">
          <span><span class="swatch" style="background:${SEVERITY_COLOR[s]};width:9px;height:9px;margin-right:6px;"></span>${s}</span>
          <span class="stat-num">${counts[s] || 0} cells (${Math.round(100 * (counts[s] || 0) / total)}%)</span>
        </div>
      `).join('');
}

// ---- live data panel ----
async function refreshLiveStatus() {
  try {
    const status = await api('/api/live/status');
    const db = status.db;
    const hasSignals = db.active_signals > 0;
    liveDot.className = 'live-dot' + (hasSignals ? ' on' : db.documents_crawled_total > 0 ? ' stale' : '');
    const lastCrawl = db.last_crawl_epoch
      ? new Date(db.last_crawl_epoch * 1000).toLocaleString()
      : 'never yet';
    const keyNote = status.last_crawl.firecrawl_configured
      ? ''
      : '<div class="empty" style="margin-top:6px;">No FIRECRAWL_API_KEY set on the backend — crawling is skipped. See backend/.env.example.</div>';
    liveInfo.innerHTML = `
      <div class="stat-row"><span>Last crawl</span><span class="stat-num">${lastCrawl}</span></div>
      <div class="stat-row"><span>Documents crawled</span><span class="stat-num">${db.documents_crawled_total}</span></div>
      <div class="stat-row"><span>Active signals</span><span class="stat-num">${db.active_signals}</span></div>
      <div class="stat-row"><span>Auto-refresh every</span><span class="stat-num">${status.crawl_interval_minutes} min</span></div>
      ${keyNote}
    `;
  } catch (err) {
    liveDot.className = 'live-dot';
    liveInfo.innerHTML = '<span class="empty">Backend unreachable</span>';
  }
}

function startLivePolling() {
  refreshLiveStatus();
  setInterval(refreshLiveStatus, 30000);
}

refreshBtn.addEventListener('click', async () => {
  refreshBtn.disabled = true;
  refreshBtn.textContent = 'Crawling…';
  try {
    await fetch(`${API_BASE}/api/live/refresh`, { method: 'POST' });
    await refreshLiveStatus();
    forecastCache = null;
    await render(); // pick up any newly-applied live adjustment immediately
  } finally {
    refreshBtn.disabled = false;
    refreshBtn.textContent = 'Refresh now';
  }
});

stateSelect.addEventListener('change', onStateChange);
hazardSelect.addEventListener('change', () => { forecastCache = null; render(); });
yearSlider.addEventListener('input', render);
monthSlider.addEventListener('input', render);
modeHistorical.addEventListener('click', () => setMode('historical'));
modeForecast.addEventListener('click', () => setMode('forecast'));
