import { useRef, useState } from 'react'

// Regions are stored as fractions (0-1) of the DISPLAYED image's bounding
// box. Because the <img> preserves aspect ratio (width 100%, height auto),
// fraction-of-displayed-width == fraction-of-natural-width, so these
// coordinates line up directly with the backend's region->patch mapping
// (backend/app/intervention.py build_in_region_mask), which expects
// fractions of the ORIGINAL uploaded image -- no extra conversion needed.
export default function ImageRegionSelector({ onImageChange, regions, onRegionsChange }) {
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
        className={`file-drop${dragOver ? ' file-drop-active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input type="file" accept="image/*" onChange={handleFile} />
        {imgSrc ? 'Chọn ảnh khác (hoặc kéo-thả)' : '📷 Kéo-thả ảnh vào đây, hoặc bấm để chọn'}
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
            {regions.map((r, i) => (
              <div
                key={i}
                className="region-box"
                style={{ left: `${r.x * 100}%`, top: `${r.y * 100}%`, width: `${r.w * 100}%`, height: `${r.h * 100}%` }}
                onClick={() => removeRegion(i)}
                title="Click để xoá vùng này"
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
            Kéo chuột trên ảnh để vẽ vùng (nhiều vùng được).
            {regions.length > 0 && <span className="pill pill-green">{regions.length} vùng — click để xoá</span>}
          </p>
        </>
      )}
    </div>
  )
}
