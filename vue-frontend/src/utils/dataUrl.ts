export function extractBase64Payload(dataUrlOrPayload: string | null | undefined): string {
  if (!dataUrlOrPayload) {
    return ''
  }

  const marker = 'base64,'
  const markerIndex = dataUrlOrPayload.indexOf(marker)
  if (markerIndex < 0) {
    return dataUrlOrPayload
  }

  return dataUrlOrPayload.slice(markerIndex + marker.length)
}

export function toImageDataUrl(payloadOrDataUrl: string, mimeType = 'image/png'): string {
  if (payloadOrDataUrl.startsWith('data:') || payloadOrDataUrl.startsWith('/api/')) {
    return payloadOrDataUrl
  }

  return `data:${mimeType};base64,${payloadOrDataUrl}`
}

function readWithFileReader(
  source: Blob,
  read: (reader: FileReader, source: Blob) => void,
  errorMessage: string
): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    const resolveResult = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result)
        return
      }

      reject(new Error(errorMessage))
    }
    reader.onload = resolveResult
    reader.onloadend = resolveResult
    reader.onerror = () => reject(new Error(errorMessage))

    read(reader, source)
  })
}

export function readBlobAsDataUrl(blob: Blob, errorMessage = '读取文件失败'): Promise<string> {
  return readWithFileReader(blob, (reader, source) => reader.readAsDataURL(source), errorMessage)
}

export function readFileAsText(file: File, errorMessage = '读取文件失败'): Promise<string> {
  return readWithFileReader(file, (reader, source) => reader.readAsText(source), errorMessage)
}
