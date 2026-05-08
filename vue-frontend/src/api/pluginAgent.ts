import { apiClient } from './client'
import type { PluginData } from '@/types'
import { serializeOpenAICompatibleOptionsForApi } from '@/utils/openaiOptions'
import type { OpenAICompatibleOptions } from '@/types/settings'

export type PluginAgentMode = 'create' | 'modify'
export type PluginAgentRunState =
  | 'drafting'
  | 'awaiting_target_lock'
  | 'ready'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'

export interface PluginAgentTargetProposal {
  plugin_id: string
  display_name: string
  supported_steps: string[]
  supported_modes: string[]
}

export interface PluginAgentLockedTarget extends PluginAgentTargetProposal {
  mode: PluginAgentMode
  plugin_dir: string
}

export interface PluginAgentMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
}

export type PluginAgentEventType =
  | 'assistant'
  | 'assistant_delta'
  | 'done'
  | 'error'
  | 'log'
  | 'state'
  | 'tool_call'
  | 'tool_result'
  | 'validation'

export interface PluginAgentAssistantDeltaPayload {
  stream_id: string
  phase: string
  delta: string
  content: string
}

export interface PluginAgentAssistantPayload {
  stream_id?: string
  phase?: string
  message: string
}

export interface PluginAgentStatePayload {
  run_state: PluginAgentRunState
  label?: string
  message?: string
  locked_target?: PluginAgentLockedTarget | null
  pending_target?: PluginAgentTargetProposal | null
}

export interface PluginAgentToolCallPayload {
  group_id: string
  tool: string
  summary: string
  args_preview?: Record<string, unknown>
}

export interface PluginAgentToolResultPayload {
  group_id: string
  tool: string
  summary: string
  success: boolean
  changed_files?: string[]
  file_previews?: Record<string, string>
  debug_result?: Record<string, unknown>
}

export interface PluginAgentValidationPayload {
  summary: string
  success: boolean
  details: Record<string, unknown>
}

export interface PluginAgentDonePayload {
  summary?: string
  message: string
  validation?: Record<string, unknown>
  refresh_result?: Record<string, unknown> | null
  run_state: PluginAgentRunState
}

export interface PluginAgentErrorPayload {
  summary?: string
  message: string
  run_state: PluginAgentRunState
}

export interface PluginAgentLogPayload {
  message: string
  phase?: string
  refresh_result?: Record<string, unknown> | null
}

export type PluginAgentEventPayload =
  | PluginAgentAssistantDeltaPayload
  | PluginAgentAssistantPayload
  | PluginAgentDonePayload
  | PluginAgentErrorPayload
  | PluginAgentLogPayload
  | PluginAgentStatePayload
  | PluginAgentToolCallPayload
  | PluginAgentToolResultPayload
  | PluginAgentValidationPayload
  | Record<string, unknown>

export interface PluginAgentEvent<T = PluginAgentEventPayload> {
  id: number
  type: PluginAgentEventType | string
  payload: T
  timestamp: string
}

export interface PluginAgentSession {
  session_id: string
  mode: PluginAgentMode
  run_state: PluginAgentRunState
  selected_plugin_id?: string | null
  pending_target?: PluginAgentTargetProposal | null
  locked_target?: PluginAgentLockedTarget | null
  messages: PluginAgentMessage[]
  events: PluginAgentEvent[]
  touched_files: string[]
  file_previews: Record<string, string>
  last_validation?: Record<string, unknown> | null
  last_error?: string | null
  created_at: string
  updated_at: string
  execution_started_at?: string | null
  execution_finished_at?: string | null
}

export interface PluginAgentOverviewSection {
  title: string
  items: string[]
}

export interface PluginAgentSettingsResponse {
  success: boolean
  overview: string[]
  overview_sections?: PluginAgentOverviewSection[]
  prompt_examples: string[]
  providers: Array<{ value: string; label: string }>
  plugins: PluginData[]
  session?: PluginAgentSession | null
  error?: string
}

export interface PluginAgentSessionResponse {
  success: boolean
  session: PluginAgentSession
  error?: string
}

export interface PluginAgentAgentConfig {
  provider: string
  apiKey: string
  modelName: string
  customBaseUrl?: string
  openaiOptions: OpenAICompatibleOptions
}

function serializeAgentConfig(config: PluginAgentAgentConfig) {
  return {
    provider: config.provider,
    api_key: config.apiKey,
    model_name: config.modelName,
    custom_base_url: config.customBaseUrl || '',
    openai_options: serializeOpenAICompatibleOptionsForApi(config.openaiOptions),
  }
}

export async function getPluginAgentSettings(): Promise<PluginAgentSettingsResponse> {
  return apiClient.get<PluginAgentSettingsResponse>('/api/plugins/agent/settings')
}

export async function createPluginAgentSession(payload: {
  mode: PluginAgentMode
  plugin_id?: string
}): Promise<PluginAgentSessionResponse> {
  return apiClient.post<PluginAgentSessionResponse>('/api/plugins/agent/sessions', payload)
}

export async function getPluginAgentSession(sessionId: string): Promise<PluginAgentSessionResponse> {
  return apiClient.get<PluginAgentSessionResponse>(`/api/plugins/agent/sessions/${encodeURIComponent(sessionId)}`)
}

export async function deletePluginAgentSession(sessionId: string): Promise<{ success: boolean; deleted: boolean }> {
  return apiClient.delete<{ success: boolean; deleted: boolean }>(`/api/plugins/agent/sessions/${encodeURIComponent(sessionId)}`)
}

export async function sendPluginAgentMessage(
  sessionId: string,
  payload: {
    content: string
    agentConfig: PluginAgentAgentConfig
  },
): Promise<PluginAgentSessionResponse> {
  return apiClient.post<PluginAgentSessionResponse>(
    `/api/plugins/agent/sessions/${encodeURIComponent(sessionId)}/messages`,
    {
      content: payload.content,
      agent_config: serializeAgentConfig(payload.agentConfig),
    },
  )
}

export async function lockPluginAgentTarget(
  sessionId: string,
  proposal: PluginAgentTargetProposal,
): Promise<PluginAgentSessionResponse> {
  return apiClient.post<PluginAgentSessionResponse>(
    `/api/plugins/agent/sessions/${encodeURIComponent(sessionId)}/lock-target`,
    { proposal },
  )
}

export async function startPluginAgentExecution(
  sessionId: string,
  agentConfig: PluginAgentAgentConfig,
): Promise<PluginAgentSessionResponse> {
  return apiClient.post<PluginAgentSessionResponse>(
    `/api/plugins/agent/sessions/${encodeURIComponent(sessionId)}/start`,
    { agent_config: serializeAgentConfig(agentConfig) },
  )
}

export async function cancelPluginAgentExecution(
  sessionId: string,
): Promise<{ success: boolean; cancelled: boolean }> {
  return apiClient.post<{ success: boolean; cancelled: boolean }>(
    `/api/plugins/agent/sessions/${encodeURIComponent(sessionId)}/cancel`,
  )
}

export async function subscribePluginAgentEvents(
  sessionId: string,
  options: {
    afterId?: number
    signal?: AbortSignal
    onEvent: (event: PluginAgentEvent) => void
    onError: (error: string) => void
  },
): Promise<void> {
  const response = await fetch(
    `/api/plugins/agent/sessions/${encodeURIComponent(sessionId)}/events?after_id=${options.afterId || 0}`,
    {
      method: 'GET',
      signal: options.signal,
      headers: {
        Accept: 'text/event-stream',
      },
    },
  )

  if (!response.ok) {
    const text = await response.text()
    options.onError(text || `HTTP ${response.status}`)
    return
  }

  const reader = response.body?.getReader()
  if (!reader) {
    options.onError('无法读取响应流')
    return
  }

  const decoder = new TextDecoder()
  let buffer = ''
  let eventType = ''
  let eventData = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const rawLine of lines) {
        const line = rawLine.trimEnd()
        if (line.startsWith('event:')) {
          eventType = line.slice(6).trim()
        } else if (line.startsWith('data:')) {
          eventData = line.slice(5).trim()
        } else if (line === '' && eventType && eventData) {
          try {
            const parsed = JSON.parse(eventData) as PluginAgentEvent
            options.onEvent({ ...parsed, type: eventType })
          } catch (error) {
            console.error('解析插件 Agent SSE 失败:', error)
          }
          eventType = ''
          eventData = ''
        }
      }
    }
  } catch (error) {
    if (!options.signal?.aborted) {
      options.onError(error instanceof Error ? error.message : '事件流订阅失败')
    }
  } finally {
    reader.releaseLock()
  }
}
