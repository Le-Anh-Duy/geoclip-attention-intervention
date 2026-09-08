import { useRef, useState } from 'react'

// Regions are stored as fractions (0-1) of the DISPLAYED image's bounding
// box. Because the <img> preserves aspect ratio (width 100%, height auto),
// fraction-of-displayed-width == fraction-of-natural-width, so these
// coordinates line up directly with the backend's region->patch mapping
// (backend/app/intervention.py build_in_region_mask), which expects
// fractions of the ORIGINAL uploaded image -- no extra conversion needed.
export default function ImageRegionSelector({
  onImageChange,
  regions,
  onRegionsChange,
  proposals = [],
  selectedProposalIndexes = [],
  onToggleProposal,
}) {
  const containerRef = useRef(null)
  const [imgSrc, setImgSrc] = useState(null)
  const [drawing, setDrawing] = useState(null)
  const [dragOver, setDragOver] = useState(false)

  function loadFile(file) {
    if (!file || !file.type.startsWith('image/')) return
    onImageChange(file)
    setImgSrc(URL.createObjectURL(file))
    onRegionsChange([])
  }

  function handleFile(e) {
    loadFile(e.target.files[0])
  }

  function handleDragOver(e) {
    e.preventDefault()
    setDragOver(true)
  }

  function handleDragLeave() {
    setDragOver(false)
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    loadFile(e.dataTransfer.files[0])
  }

  function fractionFromEvent(e) {
    const rect = containerRef.current.getBoundingClientRect()
    return {
      x: Math.min(Math.max((e.clientX - rect.left) / rect.width, 0), 1),
      y: Math.min(Math.max((e.clientY - rect.top) / rect.height, 0), 1),
    }
  }

  function handleMouseDown(e) {
    if (!imgSrc) return
    const { x, y } = fractionFromEvent(e)
    setDrawing({ x0: x, y0: y, x1: x, y1: y })
  }

  function handleMouseMove(e) {
    if (!drawing) return
    const { x, y } = fractionFromEvent(e)
    setDrawing((d) => ({ ...d, x1: x, y1: y }))
  }

  function handleMouseUp() {
    if (!drawing) return
    const x = Math.min(drawing.x0, drawing.x1)
    const y = Math.min(drawing.y0, drawing.y1)
    const w = Math.abs(drawing.x1 - drawing.x0)
    const h = Math.abs(drawing.y1 - drawing.y0)
    setDrawing(null)
    if (w < 0.01 || h < 0.01) return
    onRegionsChange([...regions, { x, y, w, h }])
  }

  function removeRegion(idx) {
    onRegionsChange(regions.filter((_, i) => i !== idx))
  }

  return (
    <div className="region-selector">
      <label
        className={`file-drop${imgSrc ? ' file-drop-compact' : ''}${dragOver ? ' file-drop-active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input type="file" accept="image/*" onChange={handleFile} />
        {imgSrc ? 'Replace image' : '📷 Drop an image here, or click to choose'}
      </label>

      {imgSrc && (
        <>
          <div
            ref={containerRef}
            className="region-canvas"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <img src={imgSrc} alt="upload preview" draggable={false} />
            {proposals.map((proposal) => {
              const r = proposal.region
              const selected = selectedProposalIndexes.includes(proposal.index)
              return (
                <button
                  type="button"
                  key={`proposal-${proposal.index}`}
                  className={`proposal-box${selected ? ' proposal-box-selected' : ''}`}
                  style={{ left: `${r.x * 100}%`, top: `${r.y * 100}%`, width: `${r.w * 100}%`, height: `${r.h * 100}%` }}
                  onMouseDown={(event) => event.stopPropagation()}
                  onClick={(event) => {
                    event.stopPropagation()
                    onToggleProposal?.(proposal.index)
                  }}
                  title={`WeDetect #${proposal.index + 1} · objectness=${proposal.objectness.toFixed(3)} · ${proposal.patch_count} patches`}
                >
                  <span>{proposal.index + 1}</span>
                </button>
              )
            })}
            {regions.map((r, i) => (
              <div
                key={i}
                className="region-box"
                style={{ left: `${r.x * 100}%`, top: `${r.y * 100}%`, width: `${r.w * 100}%`, height: `${r.h * 100}%` }}
                onClick={() => removeRegion(i)}
                title="Click to remove this region"
              />
            ))}
            {drawing && (
              <div
                className="region-box drawing"
                style={{
                  left: `${Math.min(drawing.x0, drawing.x1) * 100}%`,
                  top: `${Math.min(drawing.y0, drawing.y1) * 100}%`,
                  width: `${Math.abs(drawing.x1 - drawing.x0) * 100}%`,
                  height: `${Math.abs(drawing.y1 - drawing.y0) * 100}%`,
                }}
              />
            )}
          </div>
          <p className="hint">
            Drag on the image to draw one or more regions.
            {regions.length > 0 && <span className="pill pill-green">Selected regions: {regions.length} — click to remove</span>}
            {proposals.length > 0 && (
              <span className="pill pill-blue">
                Selected WeDetect proposals: {selectedProposalIndexes.length}/{proposals.length}
              </span>
            )}
          </p>
        </>
      )}
    </div>
  )
}
