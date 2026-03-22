type CacheEntry = {
  key: string
  value: string
  lastUsed: number
}

const CACHE_LIMIT = 3
const cache = new Map<string, CacheEntry>()

function touch(key: string) {
  const e = cache.get(key)
  if (!e) return
  e.lastUsed = Date.now()
  cache.set(key, e)
}

function put(key: string, value: string) {
  cache.set(key, { key, value, lastUsed: Date.now() })
  if (cache.size <= CACHE_LIMIT) return
  let oldestKey: string | null = null
  let oldestTs = Number.POSITIVE_INFINITY
  for (const [k, v] of cache.entries()) {
    if (v.lastUsed < oldestTs) {
      oldestTs = v.lastUsed
      oldestKey = k
    }
  }
  if (oldestKey) cache.delete(oldestKey)
}

async function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => resolve(String(reader.result || ''))
    reader.onerror = () => reject(new Error('blobToDataUrl failed'))
    reader.readAsDataURL(blob)
  })
}

export function extractPureBase64FromDataUrl(dataUrl: string): string | null {
  if (!dataUrl) return null
  if (dataUrl.startsWith('data:')) {
    const idx = dataUrl.indexOf(',')
    return idx >= 0 ? dataUrl.slice(idx + 1) : null
  }
  return dataUrl
}

export async function getPureBase64FromImageSource(src: string | null | undefined): Promise<string | null> {
  if (!src || typeof src !== 'string') return null

  if (src.startsWith('data:')) {
    return extractPureBase64FromDataUrl(src)
  }

  if (src.startsWith('/api/')) {
    const hit = cache.get(src)
    if (hit) {
      touch(src)
      return hit.value
    }
    const resp = await fetch(src)
    if (!resp.ok) return null
    const blob = await resp.blob()
    const dataUrl = await blobToDataUrl(blob)
    const pure = extractPureBase64FromDataUrl(dataUrl)
    if (!pure) return null
    put(src, pure)
    return pure
  }

  return src
}

export function clearImageSourceBase64Cache(): void {
  cache.clear()
}

