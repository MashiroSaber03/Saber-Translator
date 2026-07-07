export const CUSTOM_AI_VISION_PROVIDER_ID_FRONTEND = 'custom'
export const DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE = 32

export const OCR_ENGINES = [
  { value: 'manga_ocr', label: 'MangaOCR (本地)', type: 'local', description: '日语漫画专用' },
  { value: 'paddle_ocr', label: 'PaddleOCR (本地)', type: 'local', description: '多语言支持' },
  { value: 'baidu_ocr', label: '百度OCR (云端)', type: 'cloud', description: '需要 API Key' },
  { value: 'ai_vision', label: 'AI视觉OCR (云端)', type: 'cloud', description: '支持多服务商' },
] as const

export const TEXT_DETECTORS = [
  { value: 'ctd', label: 'CTD (Comic Text Detector)' },
  { value: 'yolo', label: 'YOLO' },
  { value: 'default', label: 'Default (DBNet)' },
] as const
