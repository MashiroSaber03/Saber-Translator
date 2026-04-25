/**
 * OCR 结果类型定义
 */

export interface OcrResult {
  text: string
  confidence: number | null
  confidenceSupported: boolean
  engine: string
  primaryEngine: string
  fallbackUsed: boolean
}
