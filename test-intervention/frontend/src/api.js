const API_BASE = 'http://127.0.0.1:8000'

export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw new Error('unreachable')
  return res.json()
}

export async function getModelInfo() {
  const res = await fetch(`${API_BASE}/model-info`)
  if (!res.ok) throw new Error('Failed to load model info')
  return res.json()
}

export async function predict({ image, regions, layerConfigs, groundTruth, topK }) {
  const form = new FormData()
  form.append('image', image)
  form.append('regions', JSON.stringify(regions))
  form.append('layer_configs', JSON.stringify(layerConfigs))
  if (groundTruth) form.append('ground_truth', JSON.stringify(groundTruth))
  form.append('top_k', String(topK))

  const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: form })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Predict failed: ${detail}`)
  }
  return res.json()
}
