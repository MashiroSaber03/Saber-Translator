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
