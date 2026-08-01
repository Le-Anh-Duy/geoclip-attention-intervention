import { useState } from 'react'

// layerConfigs always stores additive logits biases. "Scale" is only an
// equivalent display: scale = exp(bias), bias = log(scale).
export default function LayerControls({ numLayers, layerConfigs, onChange }) {
  const [displayMode, setDisplayMode] = useState('bias')

  function setLayer(idx, key, bias) {
    const current = layerConfigs[idx] || [0, 0]
    const next = key === 'a' ? [bias, current[1]] : [current[0], bias]
    onChange({ ...layerConfigs, [idx]: next })
  }

  function displayValue(bias) {
    return displayMode === 'bias' ? bias : Number(Math.exp(bias).toPrecision(5))
  }

  function setDisplayedValue(idx, key, value) {
    const parsed = Number(value)
    if (!Number.isFinite(parsed) || (displayMode === 'scale' && parsed <= 0)) return
    setLayer(idx, key, displayMode === 'bias' ? parsed : Math.log(parsed))
  }

  const isBias = displayMode === 'bias'

  return (
    <div className="layer-controls">
      <div className="layer-controls-header">
        <span className="hint">
          Display{' '}
          <select
            value={displayMode}
            aria-label="Intervention parameter display mode"
            onChange={(e) => setDisplayMode(e.target.value)}
          >
            <option value="bias">Direct bias</option>
            <option value="scale">Equivalent scale</option>
          </select>{' '}
          — no-op: {isBias ? '0' : '1'}
        </span>
        <button type="button" className="ghost-button" onClick={() => onChange({})}>
          Reset all
        </button>
      </div>
      <div className="layer-table">
        <div className="layer-row layer-row-head">
          <span>Layer</span>
          <span>a — inside</span>
          <span>b — outside</span>
        </div>
        {Array.from({ length: numLayers }, (_, idx) => {
          const [a, b] = layerConfigs[idx] || [0, 0]
          const active = a !== 0 || b !== 0
          return (
            <div className={`layer-row${active ? ' layer-row-active' : ''}`} key={idx}>
              <span className="layer-idx">{idx}</span>
              {[
                ['a', a],
                ['b', b],
              ].map(([key, bias]) => (
                <span className="ab-field" key={key}>
                  <input
                    type="range"
                    min="-5"
                    max="5"
                    step="0.1"
                    value={bias}
                    aria-label={`Layer ${idx} ${key} ${displayMode}`}
                    onChange={(e) => setLayer(idx, key, Number(e.target.value))}
                  />
                  <input
                    type="number"
                    step={isBias ? '0.1' : '0.01'}
                    min={isBias ? '-20' : '0.000001'}
                    className="ab-number"
                    value={displayValue(bias)}
                    onChange={(e) => setDisplayedValue(idx, key, e.target.value)}
                  />
                </span>
              ))}
            </div>
          )
        })}
      </div>
    </div>
  )
}
