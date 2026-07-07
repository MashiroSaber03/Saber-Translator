export type NoteType = 'text' | 'qa'

export interface NoteData {
  id: string
  type: NoteType
  content: string
  pageNum?: number
  createdAt?: string
  updatedAt?: string
  title?: string
  tags?: string[]
  question?: string
  answer?: string
  citations?: Array<{ page: number; content: string }>
  comment?: string
}

export interface QAHistory {
  id: string
  question: string
  answer: string
  sources?: Array<{
    page_num: number
    content: string
    score?: number
  }>
  created_at: string
}

export interface OverviewTemplateMeta {
  name: string
  icon: string
  description: string
}

export interface GeneratedTemplate {
  template_key: string
  template_name?: string
  content?: string
  generated_at?: string
}
