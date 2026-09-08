// Reverse-geocodes lat/lon into a human-readable "City, Country" string.
//
// openstreetmap.org is unreachable from some networks -- including the one
// this demo has been run on, where every nominatim.openstreetmap.org request
// fails to connect -- so Nominatim cannot be the only provider. Providers are
// tried in order and the first usable answer wins. When all of them fail the
// result is null, and the caller shows the coordinates alone rather than an
// error string repeated once per prediction.
const cache = new Map()
let queue = Promise.resolve()

const REQUEST_TIMEOUT_MS = 6000
const BETWEEN_REQUESTS_MS = 200

const PROVIDERS = [
  {
    // No API key, CORS-enabled, and reachable where openstreetmap.org is not.
    url: (lat, lon) =>
      `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=en`,
    parse: (d) =>
      [d.city || d.locality || d.principalSubdivision, d.countryName].filter(Boolean).join(', '),
  },
  {
    // Nominatim caps unauthenticated use at ~1 request/sec; it is the fallback
    // rather than the primary, so that budget is rarely touched.
    url: (lat, lon) =>
      `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}&zoom=10&addressdetails=1`,
    parse: (d) => {
      const a = d.address || {}
      const place = a.city || a.town || a.village || a.county || a.state
      return [place, a.country].filter(Boolean).join(', ') || d.display_name || ''
    },
  },
]

function cacheKey(lat, lon) {
  return `${lat.toFixed(3)},${lon.toFixed(3)}`
}

async function fetchPlaceName(lat, lon) {
  for (const provider of PROVIDERS) {
    try {
      const res = await fetch(provider.url(lat, lon), {
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      })
      if (!res.ok) continue
      const label = provider.parse(await res.json())
      if (label) return label
    } catch {
      // Unreachable, timed out or malformed -- fall through to the next one.
    }
  }
  return null
}

export function reverseGeocode(lat, lon) {
  const key = cacheKey(lat, lon)
  if (cache.has(key)) return cache.get(key)

  const promise = queue.then(() => fetchPlaceName(lat, lon))
  queue = promise.then(() => new Promise((resolve) => setTimeout(resolve, BETWEEN_REQUESTS_MS)))
  // Only successes are worth keeping: a failure is usually the network, and
  // caching it would prevent the name ever appearing once the network returns.
  promise.then((label) => {
    if (label === null) cache.delete(key)
  })
  cache.set(key, promise)
  return promise
}
