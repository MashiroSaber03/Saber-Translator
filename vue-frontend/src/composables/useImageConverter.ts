import { ref } from 'vue'
import { readBlobAsDataUrl } from '@/utils/dataUrl'

export interface ImageConvertResult {
  success: boolean
  base64?: string
  error?: string
  width?: number
  height?: number
}

export interface BatchConvertProgress {
  total: number
  completed: number
  currentFile?: string
}

export function useImageConverter() {
  const isConverting = ref(false)
  const convertProgress = ref<BatchConvertProgress>({ total: 0, completed: 0 })

  async function urlToBase64(
    url: string,
    outputFormat: 'image/png' | 'image/jpeg' | 'image/webp' = 'image/png',
    quality: number = 0.92
  ): Promise<ImageConvertResult> {
    if (url.startsWith('data:')) {
      try {
        const dimensions = await getImageDimensions(url)
        return {
          success: true,
          base64: url,
          width: dimensions.width,
          height: dimensions.height
        }
      } catch {
        return {
          success: true,
          base64: url
        }
      }
    }

    return new Promise((resolve) => {
      const img = new Image()

      img.crossOrigin = 'anonymous'

      img.onload = () => {
        try {
          const canvas = document.createElement('canvas')
          canvas.width = img.naturalWidth || img.width
          canvas.height = img.naturalHeight || img.height

          const ctx = canvas.getContext('2d')
          if (!ctx) {
            resolve({
              success: false,
              error: '无法创建 Canvas 上下文'
            })
            return
          }

          ctx.drawImage(img, 0, 0)

          const base64 = canvas.toDataURL(outputFormat, quality)

          resolve({
            success: true,
            base64,
            width: canvas.width,
            height: canvas.height
          })
        } catch (error) {
          resolve({
            success: false,
            error: `Canvas 转换失败: ${error instanceof Error ? error.message : '未知错误'}`
          })
        }
      }

      img.onerror = () => {
        resolve({
          success: false,
          error: '图片加载失败'
        })
      }

      if (url.startsWith('http')) {
        const separator = url.includes('?') ? '&' : '?'
        img.src = `${url}${separator}_t=${Date.now()}`
      } else {
        img.src = url
      }
    })
  }

  async function getImageDimensions(url: string): Promise<{ width: number; height: number }> {
    return new Promise((resolve, reject) => {
      const img = new Image()

      img.onload = () => {
        resolve({
          width: img.naturalWidth || img.width,
          height: img.naturalHeight || img.height
        })
      }

      img.onerror = () => {
        reject(new Error('图片加载失败'))
      }

      img.src = url
    })
  }

  async function batchUrlToBase64(
    urls: string[],
    outputFormat: 'image/png' | 'image/jpeg' | 'image/webp' = 'image/png',
    quality: number = 0.92
  ): Promise<ImageConvertResult[]> {
    isConverting.value = true
    convertProgress.value = { total: urls.length, completed: 0 }

    const results: ImageConvertResult[] = []

    try {
      for (let i = 0; i < urls.length; i++) {
        const url = urls[i]
        if (!url) continue

        convertProgress.value = {
          total: urls.length,
          completed: i,
          currentFile: `图片 ${i + 1}`
        }

        const result = await urlToBase64(url, outputFormat, quality)
        results.push(result)

        convertProgress.value.completed = i + 1
      }
    } finally {
      isConverting.value = false
      convertProgress.value = { total: 0, completed: 0 }
    }

    return results
  }

  function base64ToBlob(base64: string): Blob | null {
    try {
      const matches = base64.match(/^data:([^;]+);base64,(.+)$/)
      if (!matches) {
        return null
      }

      const mimeType = matches[1] || 'image/png'
      const data = matches[2] || ''

      const byteString = atob(data)
      const arrayBuffer = new ArrayBuffer(byteString.length)
      const uint8Array = new Uint8Array(arrayBuffer)

      for (let i = 0; i < byteString.length; i++) {
        uint8Array[i] = byteString.charCodeAt(i)
      }

      return new Blob([uint8Array], { type: mimeType })
    } catch {
      return null
    }
  }

  async function blobToBase64(blob: Blob): Promise<string> {
    return readBlobAsDataUrl(blob, 'FileReader 读取失败')
  }

  async function fileToBase64(file: File): Promise<string> {
    return blobToBase64(file)
  }

  async function createImageFromBase64(base64: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image()

      img.onload = () => {
        resolve(img)
      }

      img.onerror = () => {
        reject(new Error('图片创建失败'))
      }

      img.src = base64
    })
  }

  async function resizeImage(
    base64: string,
    maxWidth: number,
    maxHeight: number,
    outputFormat: 'image/png' | 'image/jpeg' | 'image/webp' = 'image/png',
    quality: number = 0.92
  ): Promise<ImageConvertResult> {
    try {
      const img = await createImageFromBase64(base64)

      let width = img.naturalWidth || img.width
      let height = img.naturalHeight || img.height

      if (width > maxWidth || height > maxHeight) {
        const ratio = Math.min(maxWidth / width, maxHeight / height)
        width = Math.round(width * ratio)
        height = Math.round(height * ratio)
      }

      const canvas = document.createElement('canvas')
      canvas.width = width
      canvas.height = height

      const ctx = canvas.getContext('2d')
      if (!ctx) {
        return {
          success: false,
          error: '无法创建 Canvas 上下文'
        }
      }

      ctx.drawImage(img, 0, 0, width, height)

      return {
        success: true,
        base64: canvas.toDataURL(outputFormat, quality),
        width,
        height
      }
    } catch (error) {
      return {
        success: false,
        error: `调整图片大小失败: ${error instanceof Error ? error.message : '未知错误'}`
      }
    }
  }

  function isValidBase64Image(base64: string): boolean {
    if (!base64 || typeof base64 !== 'string') {
      return false
    }

    const pattern = /^data:image\/(png|jpeg|jpg|gif|webp|bmp|svg\+xml);base64,[A-Za-z0-9+/]+=*$/
    return pattern.test(base64)
  }

  function getBase64MimeType(base64: string): string | null {
    const matches = base64.match(/^data:([^;]+);base64,/)
    return matches && matches[1] ? matches[1] : null
  }

  function getBase64Extension(base64: string): string {
    const mimeType = getBase64MimeType(base64)
    if (!mimeType) return 'png'

    const mimeToExt: Record<string, string> = {
      'image/png': 'png',
      'image/jpeg': 'jpg',
      'image/jpg': 'jpg',
      'image/gif': 'gif',
      'image/webp': 'webp',
      'image/bmp': 'bmp',
      'image/svg+xml': 'svg'
    }

    return mimeToExt[mimeType] || 'png'
  }

  function getBase64Size(base64: string): number {
    const matches = base64.match(/^data:[^;]+;base64,(.+)$/)
    if (!matches || !matches[1]) return 0

    const data = matches[1]
    const paddingMatch = data.match(/=+$/)
    const padding = paddingMatch ? paddingMatch[0].length : 0
    return Math.floor((data.length * 3) / 4) - padding
  }

  return {
    isConverting,
    convertProgress,
    urlToBase64,
    batchUrlToBase64,
    base64ToBlob,
    blobToBase64,
    fileToBase64,
    createImageFromBase64,
    resizeImage,
    getImageDimensions,
    isValidBase64Image,
    getBase64MimeType,
    getBase64Extension,
    getBase64Size
  }
}
