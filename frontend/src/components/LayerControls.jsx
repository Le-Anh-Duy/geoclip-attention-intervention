// One row per CLIP vision-tower layer (24 for ViT-L/14 -- small enough to
// list flat, no grouping/virtualization needed). Sends {layerIdx: [a, b]}
// up to App, which forwards it as `layer_configs` to POST /predict; the
// backend applies it in backend/app/intervention.py
// (_intervened_attention_forward / key_scale_for_layer).
export default function LayerControls({ numLayers, layerConfigs, onChange }) {
  function setLayer(idx, key, value) {
    const current = layerConfigs[idx] || [1, 1]
    const next = key === 'a' ? [value, current[1]] : [current[0], value]
    onChange({ ...layerConfigs, [idx]: next })
  }

  return (
    <div className="layer-controls">
      <div className="layer-controls-header">
        <span className="hint">Mặc định a=b=1 (không can thiệp)</span>
        <button type="button" className="ghost-button" onClick={() => onChange({})}>
          Reset tất cả
        </button>
      </div>
      <div className="layer-table">
        <div className="layer-row layer-row-head">
          <span>Layer</span>
          <span>a — trong vùng</span>
          <span>b — ngoài vùng</span>
        </div>
        {Array.from({ length: numLayers }, (_, idx) => {
          const [a, b] = layerConfigs[idx] || [1, 1]
          const active = a !== 1 || b !== 1
          return (
            <div className={`layer-row${active ? ' layer-row-active' : ''}`} key={idx}>
              <span className="layer-idx">{idx}</span>
              <span className="ab-field">
                <input
                  type="range"
                  min="0"
                  max="3"
                  step="0.1"
                  value={a}
                  onChange={(e) => setLayer(idx, 'a', parseFloat(e.target.value))}
                />
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  className="ab-number"
                  value={a}
                  onChange={(e) => setLayer(idx, 'a', parseFloat(e.target.value) || 0)}
                />
              </span>
              <span className="ab-field">
                <input
                  type="range"
                  min="0"
                  max="3"
                  step="0.1"
                  value={b}
                  onChange={(e) => setLayer(idx, 'b', parseFloat(e.target.value))}
                />
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  className="ab-number"
                  value={b}
                  onChange={(e) => setLayer(idx, 'b', parseFloat(e.target.value) || 0)}
                />
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
