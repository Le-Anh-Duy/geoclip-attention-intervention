function fmtTop1(run) {
  const p = run.predictions[0]
  if (!p) return '—'
  const dist = run.top1_distance_km != null ? ` (${run.top1_distance_km.toFixed(1)}km)` : ''
  return `${p.lat.toFixed(3)}, ${p.lon.toFixed(3)}${dist}`
}

export default function SavedRuns({ runs, onDelete }) {
  if (runs.length === 0) return null

  return (
    <div className="saved-runs">
      <h2 className="panel-title">Saved comparisons ({runs.length})</h2>
      <div className="saved-runs-list">
        {runs.map((r) => (
          <div className="saved-run" key={r.id}>
            <img src={r.thumbnail} alt="" className="saved-run-thumb" />
            <div className="saved-run-info">
              <div className="saved-run-meta">
                <span>{r.savedAt}</span>
                <span className="pill pill-gray">Regions: {r.regionCount}</span>
                <span className="pill pill-amber">Active layers: {r.activeLayerCount}</span>
              </div>
              <div className="saved-run-metrics">
                <span className="metric-blue">Baseline: {fmtTop1(r.result.baseline)}</span>
                <span className="metric-red">Intervention: {fmtTop1(r.result.intervention)}</span>
              </div>
            </div>
            <button type="button" className="ghost-button" onClick={() => onDelete(r.id)}>
              Delete
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}
