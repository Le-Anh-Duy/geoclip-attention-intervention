import { useEffect, useState } from 'react'
import ImageRegionSelector from './components/ImageRegionSelector'
import LayerControls from './components/LayerControls'
import ResultsPanel from './components/ResultsPanel'
import SavedRuns from './components/SavedRuns'
import StatusBadge from './components/StatusBadge'
import { getModelInfo, predict } from './api'
import './App.css'

const SAVED_RUNS_KEY = 'geoclip-saved-runs'

function loadSavedRuns() {
  try {
    return JSON.parse(localStorage.getItem(SAVED_RUNS_KEY) || '[]')
  } catch {
    return []
  }
}

// Downscaled JPEG data URL so saved runs stay small in localStorage.
function makeThumbnail(file, maxWidth = 96) {
  return new Promise((resolve) => {
    const img = new Image()
    const objectUrl = URL.createObjectURL(file)
    img.onload = () => {
      const scale = maxWidth / img.width
      const canvas = document.createElement('canvas')
      canvas.width = maxWidth
      canvas.height = Math.round(img.height * scale)
      canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height)
      URL.revokeObjectURL(objectUrl)
      resolve(canvas.toDataURL('image/jpeg', 0.6))
    }
    img.src = objectUrl
  })
}

function App() {
  const [modelInfo, setModelInfo] = useState(null)
  const [image, setImage] = useState(null)
  const [regions, setRegions] = useState([])
  const [layerConfigs, setLayerConfigs] = useState({})
  const [gtLat, setGtLat] = useState('')
  const [gtLon, setGtLon] = useState('')
  const [topK, setTopK] = useState(5)
  const [result, setResult] = useState(null)
  const [resultSaved, setResultSaved] = useState(false)
  const [savedRuns, setSavedRuns] = useState(loadSavedRuns)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // /model-info 503s until the backend finishes loading the model (see
  // StatusBadge / GET /health), so keep retrying instead of a one-shot fetch.
  useEffect(() => {
    if (modelInfo) return
    let cancelled = false
    const id = setInterval(async () => {
      try {
        const info = await getModelInfo()
        if (!cancelled) {
          setModelInfo(info)
          clearInterval(id)
        }
      } catch {
        // still loading / backend offline -- StatusBadge shows this, keep retrying silently
      }
    }, 2000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [modelInfo])

  const groundTruth = gtLat !== '' && gtLon !== '' ? { lat: parseFloat(gtLat), lon: parseFloat(gtLon) } : null

  async function handleRun() {
    if (!image) {
      setError('Please select an image')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const res = await predict({ image, regions, layerConfigs, groundTruth, topK })
      setResult(res)
      setResultSaved(false)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const activeLayerCount = Object.values(layerConfigs).filter(([a, b]) => a !== 0 || b !== 0).length

  async function handleSaveRun() {
    if (!result || !image) return
    const thumbnail = await makeThumbnail(image)
    const entry = {
      id: Date.now(),
      savedAt: new Date().toLocaleString('en-US'),
      thumbnail,
      regionCount: regions.length,
      activeLayerCount,
      result,
    }
    const next = [entry, ...savedRuns]
    setSavedRuns(next)
    setResultSaved(true)
    try {
      localStorage.setItem(SAVED_RUNS_KEY, JSON.stringify(next))
    } catch {
      // localStorage quota exceeded -- keep it in memory for this session at least
    }
  }

  function handleDeleteRun(id) {
    const next = savedRuns.filter((r) => r.id !== id)
    setSavedRuns(next)
    localStorage.setItem(SAVED_RUNS_KEY, JSON.stringify(next))
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-title">
          <span className="app-badge">GeoCLIP</span>
          <h1>Attention-intervention sandbox</h1>
          <StatusBadge predicting={loading} />
        </div>
        <p className="app-subtitle">
          Draw regions of interest, tune <code>a</code>/<code>b</code> for each attention layer, and compare the result
          with the baseline &mdash; no Grounding DINO required.
        </p>
      </header>

      <a className="docs-link" href="./docs.html" aria-label="Open documentation" title="Documentation">
        ?
      </a>

      {error && <p className="error">⚠ {error}</p>}

      <div className="layout">
        <section className="panel">
          <h2 className="panel-title">1. Image &amp; regions of interest</h2>
          <ImageRegionSelector onImageChange={setImage} regions={regions} onRegionsChange={setRegions} />

          <h2 className="panel-title panel-title-spaced">2. Ground truth (optional)</h2>
          <div className="ground-truth">
            <label>
              <span>Lat</span>
              <input value={gtLat} onChange={(e) => setGtLat(e.target.value)} placeholder="e.g. 22.0964" />
            </label>
            <label>
              <span>Lon</span>
              <input value={gtLon} onChange={(e) => setGtLon(e.target.value)} placeholder="e.g. -159.5261" />
            </label>
            <label className="topk-field">
              <span>Top-k</span>
              <input
                type="number"
                min="1"
                max="20"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value, 10) || 5)}
              />
            </label>
          </div>

          <button type="button" className="run-button" onClick={handleRun} disabled={loading}>
            {loading ? (
              <>
                <span className="spinner" /> Running on CPU…
              </>
            ) : (
              'Run inference'
            )}
          </button>
        </section>

        <section className="panel">
          <div className="panel-title-row">
            <h2 className="panel-title">3. Attention layers</h2>
            {activeLayerCount > 0 && <span className="pill pill-amber">Active layers: {activeLayerCount}</span>}
          </div>
          {modelInfo ? (
            <LayerControls numLayers={modelInfo.num_layers} layerConfigs={layerConfigs} onChange={setLayerConfigs} />
          ) : (
            <p className="status-line">Loading model information…</p>
          )}
        </section>
      </div>

      <ResultsPanel result={result} groundTruth={groundTruth} onSave={handleSaveRun} saved={resultSaved} />
      <SavedRuns runs={savedRuns} onDelete={handleDeleteRun} />
    </div>
  )
}

export default App
