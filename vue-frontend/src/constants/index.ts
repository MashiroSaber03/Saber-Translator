// Detection, OCR, color extraction and background repair each have one pool.
export const MAX_DEEP_LEARNING_CONCURRENCY = 4

export const IMAGE_EXPORT_FORMAT_OPTIONS = [
  { label: 'JPG（体积小）', value: 'jpeg' },
  { label: 'PNG（无损）', value: 'png' },
]

export * from './prompts'

export * from './rateLimits'

export * from './edit'

export * from './storage'

export * from './ocr'

export * from './webImport'

export * from './routes'

export * from './bookshelf'
