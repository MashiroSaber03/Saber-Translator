/**
 * 图片转换组合式函数
 * 处理图片 URL 转 Base64、跨域图片处理等功能
 * 用于会话保存、导出功能和 Canvas 操作
 */

import { ref } from 'vue'

/**
 * 图片转换结果
 */
export interface ImageConvertResult {
  /** 是否成功 */
  success: boolean
  /** Base64 数据（成功时） */
  base64?: string
  /** 错误信息（失败时） */
  error?: string
  /** 图片宽度 */
  width?: number
  /** 图片高度 */
  height?: number
}

/**
 * 批量转换进度
 */
export interface BatchConvertProgress {
  /** 总数 */
  total: number
  /** 已完成数 */
  completed: number
  /** 当前处理的文件名 */
  currentFile?: string
}

/**
 * 图片转换组合式函数
 */
export function useImageConverter() {
  // 状态
  const isConverting = ref(false)
  const convertProgress = ref<BatchConvertProgress>({ total: 0, completed: 0 })

  /**
   * 将图片 URL 转换为 Base64
   * 支持跨域图片处理（通过 Canvas）
   * @param url - 图片 URL（可以是 http/https URL 或 data URL）
   * @param outputFormat - 输出格式，默认 'image/png'
   * @param quality - 图片质量（0-1），仅对 jpeg 格式有效
   * @returns 转换结果
   */
  async function urlToBase64(
    url: string,
    outputFormat: 'image/png' | 'image/jpeg' | 'image/webp' = 'image/png',
    quality: number = 0.92
  ): Promise<ImageConvertResult> {
    // 如果已经是 Base64 格式，直接返回
    if (url.startsWith('data:')) {
      // 提取图片尺寸（需要加载图片）
      try {
        const dimensions = await getImageDimensions(url)
        return {
          success: true,
          base64: url,
          width: dimensions.width,
          height: dimensions.height
        }
      } catch {
        // 无法获取尺寸，但数据有效
        return {
          success: true,
          base64: url
        }
      }
    }

    return new Promise((resolve) => {
      const img = new Image()
      
      // 设置跨域属性，允许 Canvas 读取跨域图片
      img.crossOrigin = 'anonymous'

      img.onload = () => {
        try {
          // 创建 Canvas
          const canvas = document.createElement('canvas')
          canvas.width = img.naturalWidth || img.width
          canvas.height = img.naturalHeight || img.height

          // 绘制图片到 Canvas
          const ctx = canvas.getContext('2d')
          if (!ctx) {
            resolve({
              success: false,
              error: '无法创建 Canvas 上下文'
            })
            return
          }

          ctx.drawImage(img, 0, 0)

          // 转换为 Base64
          const base64 = canvas.toDataURL(outputFormat, quality)

          resolve({
            success: true,
            base64,
            width: canvas.width,
            height: canvas.height
          })
        } catch (error) {
          // Canvas 污染错误（跨域图片未正确配置 CORS）
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

      // 添加时间戳避免缓存问题（对于跨域图片）
      if (url.startsWith('http')) {
        const separator = url.includes('?') ? '&' : '?'
        img.src = `${url}${separator}_t=${Date.now()}`
      } else {
        img.src = url
      }
    })
  }

  /**
   * 获取图片尺寸
   * @param url - 图片 URL
   * @returns 图片尺寸
   */
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

  /**
   * 批量将图片 URL 转换为 Base64
   * @param urls - 图片 URL 数组
   * @param outputFormat - 输出格式
   * @param quality - 图片质量
   * @returns 转换结果数组
   */
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

  /**
   * 将 Base64 转换为 Blob
   * @param base64 - Base64 字符串
   * @returns Blob 对象
   */
  function base64ToBlob(base64: string): Blob | null {
    try {
      // 提取 MIME 类型和数据
      const matches = base64.match(/^data:([^;]+);base64,(.+)$/)
      if (!matches) {
        console.error('无效的 Base64 格式')
        return null
      }

      const mimeType = matches[1] || 'image/png'
      const data = matches[2] || ''

      // 解码 Base64
      const byteString = atob(data)
      const arrayBuffer = new ArrayBuffer(byteString.length)
      const uint8Array = new Uint8Array(arrayBuffer)

      for (let i = 0; i < byteString.length; i++) {
        uint8Array[i] = byteString.charCodeAt(i)
      }

      return new Blob([uint8Array], { type: mimeType })
    } catch (error) {
      console.error('Base64 转 Blob 失败:', error)
      return null
    }
  }

  /**
   * 将 Blob 转换为 Base64
   * @param blob - Blob 对象
   * @returns Base64 字符串
   */
  async function blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          resolve(reader.result)
        } else {
          reject(new Error('FileReader 返回了非字符串结果'))
        }
      }

      reader.onerror = () => {
        reject(new Error('FileReader 读取失败'))
      }

      reader.readAsDataURL(blob)
    })
  }

  /**
   * 将 File 对象转换为 Base64
   * @param file - File 对象
   * @returns Base64 字符串
   */
  async function fileToBase64(file: File): Promise<string> {
    return blobToBase64(file)
  }

  /**
   * 从 Base64 创建图片元素
   * @param base64 - Base64 字符串
   * @returns HTMLImageElement
   */
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

  /**
   * 调整图片大小
   * @param base64 - 原始 Base64 图片
   * @param maxWidth - 最大宽度
   * @param maxHeight - 最大高度
   * @param outputFormat - 输出格式
   * @param quality - 图片质量
   * @returns 调整后的 Base64 图片
   */
  async function resizeImage(
    base64: string,
    maxWidth: number,
    maxHeight: number,
    outputFormat: 'image/png' | 'image/jpeg' | 'image/webp' = 'image/png',
    quality: number = 0.92
  ): Promise<ImageConvertResult> {
    try {
      const img = await createImageFromBase64(base64)
      
      // 计算缩放比例
      let width = img.naturalWidth || img.width
      let height = img.naturalHeight || img.height

      if (width > maxWidth || height > maxHeight) {
        const ratio = Math.min(maxWidth / width, maxHeight / height)
        width = Math.round(width * ratio)
        height = Math.round(height * ratio)
      }

      // 创建 Canvas 并绘制
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

  /**
   * 验证 Base64 图片格式是否有效
   * @param base64 - Base64 字符串
   * @returns 是否有效
   */
  function isValidBase64Image(base64: string): boolean {
    if (!base64 || typeof base64 !== 'string') {
      return false
    }

    // 检查是否是有效的 data URL 格式
    const pattern = /^data:image\/(png|jpeg|jpg|gif|webp|bmp|svg\+xml);base64,[A-Za-z0-9+/]+=*$/
    return pattern.test(base64)
  }

  /**
   * 获取 Base64 图片的 MIME 类型
   * @param base64 - Base64 字符串
   * @returns MIME 类型或 null
   */
  function getBase64MimeType(base64: string): string | null {
    const matches = base64.match(/^data:([^;]+);base64,/)
    return matches && matches[1] ? matches[1] : null
  }

  /**
   * 获取 Base64 图片的文件扩展名
   * @param base64 - Base64 字符串
   * @returns 文件扩展名
   */
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

  /**
   * 计算 Base64 图片的大小（字节）
   * @param base64 - Base64 字符串
   * @returns 大小（字节）
   */
  function getBase64Size(base64: string): number {
    // 移除 data URL 前缀
    const matches = base64.match(/^data:[^;]+;base64,(.+)$/)
    if (!matches || !matches[1]) return 0

    const data = matches[1]
    // Base64 编码后大小约为原始大小的 4/3
    // 每个 Base64 字符代表 6 位，所以 4 个字符 = 3 字节
    const paddingMatch = data.match(/=+$/)
    const padding = paddingMatch ? paddingMatch[0].length : 0
    return Math.floor((data.length * 3) / 4) - padding
  }

  return {
    // 状态
    isConverting,
    convertProgress,

    // 核心转换方法
    urlToBase64,
    batchUrlToBase64,
    base64ToBlob,
    blobToBase64,
    fileToBase64,

    // 图片处理方法
    createImageFromBase64,
    resizeImage,
    getImageDimensions,

    // 工具方法
    isValidBase64Image,
    getBase64MimeType,
    getBase64Extension,
    getBase64Size
  }
}
