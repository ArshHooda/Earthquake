const USGS_BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query";

const filterForm = document.getElementById("filterForm");
const startDateInput = document.getElementById("startDate");
const endDateInput = document.getElementById("endDate");
const minMagnitudeInput = document.getElementById("minMagnitude");

const totalEventsEl = document.getElementById("totalEvents");
const largestMagEl = document.getElementById("largestMag");
const avgMagEl = document.getElementById("avgMag");
const activeRegionEl = document.getElementById("activeRegion");
const eventsTableBody = document.getElementById("eventsTableBody");
const statusEl = document.getElementById("status");

const map = L.map("map", {
  worldCopyJump: true,
  minZoom: 2,
}).setView([20, 0], 2);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
  maxZoom: 19,
}).addTo(map);

let markersLayer = L.layerGroup().addTo(map);

function toISODate(date) {
  return date.toISOString().split("T")[0];
}

function setDefaultDates() {
  const today = new Date();
  const past = new Date();
  past.setDate(today.getDate() - 7);
  startDateInput.value = toISODate(past);
  endDateInput.value = toISODate(today);
}

function notify(message) {
  statusEl.textContent = message;
  statusEl.classList.add("show");
  setTimeout(() => {
    statusEl.classList.remove("show");
  }, 2400);
}

function magnitudeColor(mag) {
  if (mag >= 6) return "#ef4444";
  if (mag >= 5) return "#f97316";
  if (mag >= 4) return "#eab308";
  return "#38bdf8";
}

function extractRegion(place) {
  if (!place) return "Unknown";
  const parts = place.split(",");
  return parts.length > 1 ? parts[parts.length - 1].trim() : parts[0].trim();
}

function updateStats(events) {
  const magnitudes = events
    .map((event) => event.properties.mag)
    .filter((mag) => typeof mag === "number");

  totalEventsEl.textContent = events.length.toLocaleString();

  if (magnitudes.length === 0) {
    largestMagEl.textContent = "N/A";
    avgMagEl.textContent = "N/A";
    activeRegionEl.textContent = "N/A";
    return;
  }

  const maxMag = Math.max(...magnitudes);
  const avgMag = magnitudes.reduce((sum, mag) => sum + mag, 0) / magnitudes.length;

  const regionCount = events.reduce((acc, event) => {
    const region = extractRegion(event.properties.place);
    acc[region] = (acc[region] || 0) + 1;
    return acc;
  }, {});

  const [topRegion] = Object.entries(regionCount).sort((a, b) => b[1] - a[1])[0] || ["Unknown"];

  largestMagEl.textContent = maxMag.toFixed(1);
  avgMagEl.textContent = avgMag.toFixed(2);
  activeRegionEl.textContent = topRegion;
}

function updateTable(events) {
  eventsTableBody.innerHTML = "";

  const recentEvents = [...events]
    .sort((a, b) => b.properties.time - a.properties.time)
    .slice(0, 30);

  if (recentEvents.length === 0) {
    const row = document.createElement("tr");
    row.innerHTML = `<td colspan="4">No events found for this filter.</td>`;
    eventsTableBody.appendChild(row);
    return;
  }

  recentEvents.forEach((event) => {
    const { properties, geometry } = event;
    const row = document.createElement("tr");
    const mag = properties.mag ?? 0;
    const depth = geometry?.coordinates?.[2] ?? 0;

    row.innerHTML = `
      <td>${new Date(properties.time).toISOString().replace("T", " ").replace(".000Z", "")}</td>
      <td>${properties.place || "Unknown"}</td>
      <td class="${mag >= 5 ? "mag-high" : ""}">${mag.toFixed(1)}</td>
      <td>${depth.toFixed(1)}</td>
    `;
    eventsTableBody.appendChild(row);
  });
}

function updateMap(events) {
  markersLayer.clearLayers();

  const bounds = [];

  events.forEach((event) => {
    const [longitude, latitude, depth] = event.geometry.coordinates;
    const mag = event.properties.mag || 0;

    const marker = L.circleMarker([latitude, longitude], {
      radius: Math.max(5, mag * 2.2),
      color: magnitudeColor(mag),
      fillOpacity: 0.6,
      weight: 1,
    });

    marker.bindPopup(`
      <strong>${event.properties.place || "Unknown"}</strong><br/>
      Magnitude: ${mag.toFixed(1)}<br/>
      Depth: ${Number(depth).toFixed(1)} km<br/>
      Time: ${new Date(event.properties.time).toUTCString()}
    `);

    marker.addTo(markersLayer);
    bounds.push([latitude, longitude]);
  });

  if (bounds.length > 0) {
    map.fitBounds(bounds, { padding: [20, 20], maxZoom: 5 });
  } else {
    map.setView([20, 0], 2);
  }
}

async function fetchEarthquakes(startDate, endDate, minMagnitude) {
  const params = new URLSearchParams({
    format: "geojson",
    starttime: startDate,
    endtime: endDate,
    minmagnitude: minMagnitude,
    orderby: "time",
    limit: "500",
  });

  const response = await fetch(`${USGS_BASE_URL}?${params.toString()}`);

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  const data = await response.json();
  return data.features || [];
}

async function loadData() {
  const startDate = startDateInput.value;
  const endDate = endDateInput.value;
  const minMagnitude = minMagnitudeInput.value;

  if (new Date(startDate) > new Date(endDate)) {
    notify("Start date cannot be after end date.");
    return;
  }

  notify("Loading earthquake data...");

  try {
    const events = await fetchEarthquakes(startDate, endDate, minMagnitude);
    updateStats(events);
    updateTable(events);
    updateMap(events);
    notify(`Loaded ${events.length} event(s).`);
  } catch (error) {
    notify("Could not load data. Please try again.");
    console.error(error);
  }
}

filterForm.addEventListener("submit", (event) => {
  event.preventDefault();
  loadData();
});

setDefaultDates();
loadData();
