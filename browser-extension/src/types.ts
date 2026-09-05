export type DetectionMethod = 'adapter' | 'dom-agent' | 'similar'
export type TranslationMode = 'standard' | 'hq'
export type PageState = 'queued' | 'translating' | 'completed' | 'failed' | 'cancelled'

export interface LearnedRule {
  selector: string
  kind: 'image' | 'canvas' | 'background'
  confirmedAt: number
}

export interface PanelPosition {
  x: number
  y: number
}

export interface DomainPreference {
  disabled: boolean
  method: DetectionMethod
  mode: TranslationMode
  glossaryEnabled: boolean
  autoTermsEnabled: boolean
  panelOpen: boolean
  panelPosition?: PanelPosition
  fabPosition?: PanelPosition
  rule?: LearnedRule
}

export interface ExtensionSettings {
  token: string
  serverPort: number
  domains: Record<string, DomainPreference>
}

export interface ActiveBrowserSession {
  sessionId: string
  discovery: {
    stopped: boolean
    usingAdapter: boolean
    rule: LearnedRule | null
  }
}

export interface BrowserPageDto {
  id: string
  clientPageKey: string
  ordinal: number
  pageId: string | null
  state: PageState
  resultReady: boolean
  retryCount: number
  error: { code: string; message: string } | null
}

export interface BrowserSessionDto {
  id: string
  pageUrl: string
  pageTitle: string
  bookId: string
  chapterId: string
  mode: TranslationMode
  glossaryEnabled: boolean
  autoTermsEnabled: boolean
  state: 'idle' | 'queued' | 'translating' | 'completed' | 'partial' | 'failed' | 'cancelled'
  pendingStart: boolean
  expiresAt: string | null
  counts: Record<PageState | 'total', number>
  pages: BrowserPageDto[]
}

export interface BrowserLibraryBook {
  id: string
  title: string
  chapterCount: number
}

export type BrowserSessionImportCommand =
  | {
      destination: 'new'
      bookTitle: string
      chapterTitle: string
    }
  | {
      destination: 'existing'
      targetBookId: string
      chapterTitle: string
    }

export interface BrowserSessionImportResult {
  destination: 'new' | 'existing'
  bookId: string
  bookTitle: string
  chapterId: string
  chapterTitle: string
  importedPages: number
  omittedPages: number
  termsAdded: number
}

export interface DomNodeSummary {
  id: string
  tag: string
  classes: string[]
  parent: string
  attributes: Record<string, string>
  rect: { width: number; height: number; top: number; left: number }
  naturalSize: { width: number; height: number }
}

export interface DomDetectionResult {
  nodeIds: string[]
  selector: string
}

export interface UploadSource {
  kind: 'url' | 'data-url'
  value: string
}

export interface UploadPageRequest {
  sessionId: string
  clientPageKey: string
  ordinal: number
  logicalPath: string
  sourceUrl?: string
  source: UploadSource
}

export interface ResultImagePayload {
  base64: string
  mimeType: string
}

export interface BackgroundError {
  code: string
  message: string
  retryable: boolean
}

export type BackgroundResponse<T> =
  | { ok: true; data: T }
  | { ok: false; error: BackgroundError }

export type BackgroundRequest =
  | { type: 'get-preference'; hostname: string }
  | { type: 'set-preference'; hostname: string; preference: DomainPreference }
  | { type: 'get-popup-state' }
  | { type: 'save-connection'; token: string; serverPort: number }
  | { type: 'status' }
  | { type: 'hash-source'; value: string }
  | { type: 'get-active-session'; pageUrl: string }
  | ({ type: 'set-active-session'; pageUrl: string } & ActiveBrowserSession)
  | { type: 'clear-active-session'; pageUrl: string; sessionId?: string }
  | { type: 'create-session'; payload: Record<string, unknown> }
  | { type: 'get-session'; sessionId: string; touch?: boolean }
  | { type: 'patch-session'; sessionId: string; payload: Record<string, unknown> }
  | { type: 'start-session'; sessionId: string }
  | { type: 'upload-page'; payload: UploadPageRequest }
  | { type: 'retry-page'; sessionId: string; browserPageId: string }
  | { type: 'fetch-result'; sessionId: string; browserPageId: string }
  | { type: 'get-terms'; sessionId: string }
  | { type: 'cancel-session'; sessionId: string }
  | { type: 'list-library-books' }
  | {
      type: 'import-session'
      sessionId: string
      payload: BrowserSessionImportCommand
    }
  | { type: 'dom-detection'; payload: Record<string, unknown> }

export interface ContextTranslateMessage {
  type: 'context-translate-image'
  srcUrl: string
}
