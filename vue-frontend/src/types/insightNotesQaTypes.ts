export type NoteType = 'text' | 'qa'

export interface NoteData {
  citations: Array<{ page: number; content: string }>
  comment?: string
  content: string
  createdAt: string
  id: string
  pageNum?: number
  question?: string
  revision: number
  tags: string[]
  title: string
  type: NoteType
  updatedAt: string
}

export type NoteUpdateInput = Partial<
  Pick<
    NoteData,
    'citations' | 'comment' | 'content' | 'pageNum' | 'question' | 'tags' | 'title' | 'type'
  >
>
