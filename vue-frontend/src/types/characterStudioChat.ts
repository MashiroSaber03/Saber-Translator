export interface CharacterStudioChatAttachment {
  attachment_id: string
  filename: string
  mime_type: string
  asset_path: string
  created_at: string
}

export interface CharacterStudioChatMessage {
  message_id: string
  role: 'user' | 'assistant'
  content: string
  attachments: CharacterStudioChatAttachment[]
  runtime_log: Array<Record<string, unknown>>
  variables_snapshot: Record<string, unknown>
  generation_meta: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface CharacterStudioChatSummaryBlock {
  summary_id: string
  content: string
  created_at: string
  covered_message_ids: string[]
}

export interface CharacterStudioChatSessionSummary {
  session_id: string
  title: string
  revision?: number
  generation?: number
  message_count: number
  updated_at: string
  archived_at?: string | null
  last_message_excerpt?: string
}

export interface CharacterStudioChatSession {
  session_id: string
  doc_id: string
  title: string
  created_at: string
  updated_at: string
  archived_at?: string | null
  greeting_source?: Record<string, unknown>
  summary_blocks: CharacterStudioChatSummaryBlock[]
  messages: CharacterStudioChatMessage[]
  variables: Record<string, unknown>
  _runtime?: Record<string, unknown>
  last_prompt_preview: string
  revision?: number
  generation?: number
}

export interface CharacterStudioGreetingOption {
  greeting_id: string
  label: string
  content: string
  source: Record<string, unknown>
}
