import type { BrowserSessionDto, BrowserSessionImportResult, DomainPreference } from '../types'
export type StudioTab = 'translate' | 'settings' | 'tasks'
export interface StudioState {
  open: boolean
  title: string
  preference: DomainPreference
  tab: StudioTab
  view: 'idle' | 'candidates' | 'progress'
  notice: { title: string; message: string; tone: 'ready' | 'busy' | 'error' }
  candidates: {
    id: string
    sourceUrl: string | null
    width: number
    height: number
  }[]
  session: BrowserSessionDto | null
  originalPageIds: string[]
  translated: boolean
  preparation: { processed: number; total: number; failed: number } | null
  uploadError: { count: number; message: string } | null
  retryStart: boolean
  terms: { source?: string; target?: string }[]
  imported: BrowserSessionImportResult | null
}
export type StudioAction =
  | 'ready'
  | 'tab'
  | 'close'
  | 'drag-start'
  | 'preference'
  | 'discover'
  | 'confirm'
  | 'back'
  | 'toggle-global'
  | 'toggle-page'
  | 'retry-page'
  | 'retry-uploads'
  | 'retry-start'
  | 'restart'
  | 'stop-discovery'
  | 'cancel'
  | 'books'
  | 'import'
  | 'disable'
  | 'delete-adaptation'
  | 'diagnostics'
