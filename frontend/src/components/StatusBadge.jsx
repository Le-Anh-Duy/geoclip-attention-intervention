// Polls GET /health every 2s so the UI never looks dead while the backend
// downloads/loads the CLIP weights (see backend/app/main.py _load_model_background).
// `predicting` overrides the display to "busy" the instant a /predict call
// starts, without waiting for the next poll tick.
import { useEffect, useState } from 'react'
import { getHealth } from '../api'

const LABELS = {
  loading: { text: 'Đang tải/khởi tạo model...', cls: 'status-loading' },
  ready: { text: 'Model sẵn sàng', cls: 'status-ready' },
  busy: { text: 'Đang chạy inference...', cls: 'status-busy' },
  error: { text: 'Lỗi khi tải model', cls: 'status-error' },
  offline: { text: 'Không kết nối được backend', cls: 'status-error' },
}

export default function StatusBadge({ predicting }) {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    let cancelled = false
    async function poll() {
      try {
        const data = await getHealth()
        if (!cancelled) setHealth(data)
      } catch {
        if (!cancelled) setHealth({ state: 'offline', detail: null })
      }
    }
    poll()
    const id = setInterval(poll, 2000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  const state = predicting ? 'busy' : health?.state || 'offline'
  const info = LABELS[state] || LABELS.offline

  return (
    <div className={`status-badge ${info.cls}`} title={health?.detail || ''}>
      <span className="status-dot" />
      {info.text}
    </div>
  )
}
