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

export async function generateProposals({ image, scoreThreshold = 0.4, iouThreshold = 0.7, topK = 50 }) {
  const form = new FormData()
  form.append('image', image)
  form.append('score_threshold', String(scoreThreshold))
  form.append('iou_threshold', String(iouThreshold))
  form.append('top_k', String(topK))

  const res = await fetch(`${API_BASE}/proposals`, { method: 'POST', body: form })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Proposal generation failed: ${detail}`)
  }
  return res.json()
}

export async function evaluateProposals({
  image,
  layerConfigs,
  groundTruth,
  topK,
  scoreThreshold = 0.4,
  iouThreshold = 0.7,
  proposalTopK = 20,
}) {
  const form = new FormData()
  form.append('image', image)
  form.append('layer_configs', JSON.stringify(layerConfigs))
  if (groundTruth) form.append('ground_truth', JSON.stringify(groundTruth))
  form.append('top_k', String(topK))
  form.append('score_threshold', String(scoreThreshold))
  form.append('iou_threshold', String(iouThreshold))
  form.append('proposal_top_k', String(proposalTopK))

  const res = await fetch(`${API_BASE}/evaluate-proposals`, { method: 'POST', body: form })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Proposal evaluation failed: ${detail}`)
  }
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
