// Reverse-geocodes lat/lon into a human-readable "City, Country" string via
// Nominatim (OpenStreetMap) -- same service already relied on for map tiles,
// so no extra dependency/API key needed. Nominatim's usage policy caps
// unauthenticated use at ~1 request/sec, so calls are cached and queued
// sequentially with a delay instead of firing in parallel.
const cache = new Map()
let queue = Promise.resolve()

function cacheKey(lat, lon) {
  return `${lat.toFixed(3)},${lon.toFixed(3)}`
}

async function fetchPlaceName(lat, lon) {
  try {
    const res = await fetch(
      `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}&zoom=10&addressdetails=1`,
    )
    if (!res.ok) throw new Error('reverse geocode failed')
    const data = await res.json()
    const a = data.address || {}
    const place = a.city || a.town || a.village || a.county || a.state
    const country = a.country
    const label = [place, country].filter(Boolean).join(', ')
    return label || data.display_name || 'Unknown location (possibly at sea)'
  } catch {
    return 'Location lookup unavailable'
  } finally {
    await new Promise((resolve) => setTimeout(resolve, 1100))
  }
}

export function reverseGeocode(lat, lon) {
  const key = cacheKey(lat, lon)
  if (cache.has(key)) return cache.get(key)

  const promise = queue.then(() => fetchPlaceName(lat, lon))
  queue = promise
  cache.set(key, promise)
  return promise
}
