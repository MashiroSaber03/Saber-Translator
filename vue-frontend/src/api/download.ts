export interface ApiDownloadResult {
  blob: Blob
  filename: string
}

export interface DownloadBlobOptions {
  url: string
  fallbackFilename: string
  fallbackErrorMessage: string
  init?: RequestInit
}

export function parseContentDispositionFilename(
  contentDisposition: string | null | undefined,
  fallbackFilename: string,
): string {
  if (!contentDisposition) return fallbackFilename

  const encodedMatch = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i)
  if (encodedMatch?.[1]) {
    try {
      return decodeURIComponent(encodedMatch[1].trim())
    } catch {
      return encodedMatch[1].trim()
    }
  }

  const plainMatch = contentDisposition.match(/filename="?([^";]+)"?/i)
  return plainMatch?.[1]?.trim() || fallbackFilename
}

export async function readApiErrorMessage(
  response: Response,
  fallbackMessage: string,
): Promise<string> {
  const text = await response.text()
  if (!text) return fallbackMessage

  try {
    const parsed = JSON.parse(text) as {
      error?: unknown
      message?: unknown
    }
    if (typeof parsed.error === 'string' && parsed.error) return parsed.error
    if (typeof parsed.message === 'string' && parsed.message) return parsed.message
    if (
      parsed.error
      && typeof parsed.error === 'object'
      && 'message' in parsed.error
      && typeof parsed.error.message === 'string'
      && parsed.error.message
    ) {
      return parsed.error.message
    }
  } catch {
    return text
  }

  return text
}

export async function downloadBlob(options: DownloadBlobOptions): Promise<ApiDownloadResult> {
  const response = options.init
    ? await fetch(options.url, options.init)
    : await fetch(options.url)
  if (!response.ok) {
    throw new Error(await readApiErrorMessage(response, options.fallbackErrorMessage))
  }

  const filename = parseContentDispositionFilename(
    response.headers.get('content-disposition'),
    options.fallbackFilename,
  )
  return {
    blob: await response.blob(),
    filename,
  }
}
