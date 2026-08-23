export const CUSTOM_AI_PROFILE_KINDS = [
  'chatVision',
  'embedding',
  'reranker',
  'imageGen',
] as const

export type CustomAiProfileKind = typeof CUSTOM_AI_PROFILE_KINDS[number]

export interface CustomAiProfile {
  id: string
  name: string
  kind: CustomAiProfileKind
  baseUrl: string
  apiKey: string
  model: string
}

export type CustomAiProfilePayload = Omit<CustomAiProfile, 'apiKey'>

export const CUSTOM_AI_PROFILE_KIND_LABELS: Record<CustomAiProfileKind, string> = {
  chatVision: '对话 / 视觉',
  embedding: 'Embedding',
  reranker: 'Reranker',
  imageGen: '图像生成',
}
