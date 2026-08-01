import { useEffect, useRef, useState } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { reverseGeocode } from '../geocode'

function PredictionRow({ p, rank }) {
  const [place, setPlace] = useState(null)

  useEffect(() => {
    let cancelled = false
    reverseGeocode(p.lat, p.lon).then((name) => {
      if (!cancelled) setPlace(name)
    })
    return () => {
      cancelled = true
    }
  }, [p.lat, p.lon])

  return (
    <li>
      <span className="place-name">
        #{rank + 1} {place || <em className="place-loading">looking up location…</em>}
      </span>
      <span className="coords">
        {p.lat.toFixed(4)}, {p.lon.toFixed(4)} · p={p.prob.toFixed(4)}
      </span>
    </li>
  )
}

function RunColumn({ title, accent, run }) {
  if (!run) return null
  return (
    <div className={`run-column run-column-${accent}`}>
      <h4>{title}</h4>
      {run.top1_distance_km != null && (
        <p className="distance">
          Distance from ground truth: <strong>{run.top1_distance_km.toFixed(1)} km</strong>
        </p>
      )}
      {run.threshold_hits && (
        <div className="thresholds">
          {Object.entries(run.threshold_hits).map(([k, hit]) => (
            <span key={k} className={`pill ${hit ? 'pill-green' : 'pill-gray'}`}>
              {k} {hit ? '✓' : '✗'}
            </span>
          ))}
        </div>
      )}
      <ol className="predictions">
        {run.predictions.map((p, i) => (
          <PredictionRow key={i} p={p} rank={i} />
        ))}
      </ol>
    </div>
  )
}

// Rank 0 (top-1) is drawn biggest/most opaque; lower-ranked candidates fade
// out, so the map reads as a confidence cloud rather than a flat list of pins.
function markerStyle(color, rank) {
  return {
    color,
    radius: Math.max(10 - rank * 1.5, 4),
    fillOpacity: Math.max(0.85 - rank * 0.15, 0.25),
    opacity: Math.max(0.85 - rank * 0.15, 0.25),
  }
}

export default function ResultsPanel({ result, groundTruth, onSave, saved }) {
  const mapInstance = useRef(null)

  // Callback ref, not useRef+useEffect(,[]): the map <div> below only exists
  // in the DOM once a result has arrived (see the early `return null`
  // before first predict), so a mount-once effect would fire while the ref
  // was still null and never get a second chance -- the map silently never
  // initialized. A callback ref fires whenever the node actually attaches,
  // whenever that happens to be.
  function attachMap(node) {
    if (!node || mapInstance.current) return
    mapInstance.current = L.map(node).setView([20, 0], 2)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
    }).addTo(mapInstance.current)
  }

  useEffect(() => {
    const map = mapInstance.current
    if (!map || !result) return

    map.eachLayer((layer) => {
      if (layer instanceof L.Marker || layer instanceof L.CircleMarker) map.removeLayer(layer)
    })

    const points = []

    if (groundTruth) {
      const marker = L.marker([groundTruth.lat, groundTruth.lon]).bindTooltip('Ground truth').addTo(map)
      reverseGeocode(groundTruth.lat, groundTruth.lon).then((name) => marker.setTooltipContent(`Ground truth — ${name}`))
      points.push([groundTruth.lat, groundTruth.lon])
    }

    result.baseline.predictions.forEach((p, rank) => {
      const marker = L.circleMarker([p.lat, p.lon], markerStyle('#2563eb', rank))
        .bindTooltip(`Baseline #${rank + 1} (p=${p.prob.toFixed(3)})`)
        .addTo(map)
      reverseGeocode(p.lat, p.lon).then((name) =>
        marker.setTooltipContent(`Baseline #${rank + 1} — ${name} (p=${p.prob.toFixed(3)})`),
      )
      points.push([p.lat, p.lon])
    })

    result.intervention.predictions.forEach((p, rank) => {
      const marker = L.circleMarker([p.lat, p.lon], markerStyle('#dc2626', rank))
        .bindTooltip(`Intervention #${rank + 1} (p=${p.prob.toFixed(3)})`)
        .addTo(map)
      reverseGeocode(p.lat, p.lon).then((name) =>
        marker.setTooltipContent(`Intervention #${rank + 1} — ${name} (p=${p.prob.toFixed(3)})`),
      )
      points.push([p.lat, p.lon])
    })

    if (points.length) map.fitBounds(points, { padding: [40, 40], maxZoom: 8 })
  }, [result, groundTruth])

  if (!result) return null

  return (
    <div className="results-panel">
      <div className="panel-title-row">
        <h2 className="panel-title">Results</h2>
        <div className="panel-title-actions">
          <span className="pill pill-gray">
            {result.in_region_patch_count}/{result.grid_size * result.grid_size} patches inside regions
          </span>
          <button type="button" className="ghost-button" onClick={onSave} disabled={saved}>
            {saved ? '✓ Saved' : '💾 Save for comparison'}
          </button>
        </div>
      </div>
      <div className="results-columns">
        <RunColumn title="● Baseline" accent="blue" run={result.baseline} />
        <RunColumn title="● Intervention" accent="red" run={result.intervention} />
      </div>
      <div ref={attachMap} className="map" />
      <p className="hint">Click or hover over a map point to see its rank and place name.</p>
    </div>
  )
}
