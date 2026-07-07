export function triggerUrlDownload(url: string, filename = ''): void {
  const link = document.createElement('a')
  link.href = url
  link.download = filename

  try {
    document.body.appendChild(link)
    link.click()
  } finally {
    link.remove()
  }
}

export function triggerBlobDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  try {
    triggerUrlDownload(url, filename)
  } finally {
    URL.revokeObjectURL(url)
  }
}
