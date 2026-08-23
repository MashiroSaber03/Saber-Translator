export const CUSTOM_AI_PROFILE_KINDS = [
  'chat',
  'vision',
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
  chat: '对话 / 文本',
  vision: '视觉',
  embedding: 'Embedding',
  reranker: 'Reranker',
  imageGen: '图像生成',
}
