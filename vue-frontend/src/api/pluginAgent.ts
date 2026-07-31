import { apiClient } from './client'
import type { PluginData } from '@/types'
import type { OpenAICompatibleOptions } from '@/types/settings'
import { getProviderOptionsForCapability } from '@/config/aiProviders'
import { getPlugins } from './plugin'
import { jobsApi } from './v2/jobs'
import { newIdempotencyKey } from './v2/content'

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
  job_id?: string | null
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

const sessionJobs = new Map<string, string>()
const jobEventCursors = new Map<string, number>()
let activeSessionId = ''

function pluginAgentSessionEndpoint(sessionId: string, suffix = ''): string {
  return `/api/v2/plugin-agent/sessions/${encodeURIComponent(sessionId)}${suffix}`
}

export async function getPluginAgentSettings(): Promise<PluginAgentSettingsResponse> {
  const pluginResult = await getPlugins()
  let session: PluginAgentSession | null = null
  if (activeSessionId) {
    try {
      session = (await getPluginAgentSession(activeSessionId)).session
    } catch {
      activeSessionId = ''
    }
  }
  return {
    success: true,
    overview: [
      '规划对话是短 API；开始执行后任务进入全局后端队列。',
      '浏览器关闭不会终止插件生成，执行日志由任务中心持久保存。',
      '插件代码只在 Worker 临时工作区运行，校验成功后发布不可变 v3 版本。',
    ],
    overview_sections: [],
    prompt_examples: [
      '创建一个 after_translate 插件，把“老师”替换为“导师”，替换表可配置。',
      '创建一个 before_render 插件，统一开启描边并设置最小字号。',
      '修改现有插件，增加 after_ocr 文本清洗，保留原有行为。',
    ],
    providers: getProviderOptionsForCapability('pluginAgent'),
    plugins: pluginResult.plugins,
    session,
  }
}

export async function createPluginAgentSession(payload: {
  mode: PluginAgentMode
  plugin_id?: string
}): Promise<PluginAgentSessionResponse> {
  const result = await apiClient.post<{ session: PluginAgentSession }>(
    '/api/v2/plugin-agent/sessions',
    {
      mode: payload.mode,
      ...(payload.plugin_id ? { pluginId: payload.plugin_id } : {}),
    },
  )
  activeSessionId = result.session.session_id
  return { success: true, session: result.session }
}

export async function getPluginAgentSession(sessionId: string): Promise<PluginAgentSessionResponse> {
  const result = await apiClient.get<{ session: PluginAgentSession }>(
    pluginAgentSessionEndpoint(sessionId),
  )
  if (result.session.job_id) {
    sessionJobs.set(sessionId, result.session.job_id)
  }
  return { success: true, session: result.session }
}

export async function deletePluginAgentSession(sessionId: string): Promise<{ success: boolean; deleted: boolean }> {
  const result = await apiClient.delete<{ deleted: boolean }>(
    pluginAgentSessionEndpoint(sessionId),
  )
  if (activeSessionId === sessionId) activeSessionId = ''
  sessionJobs.delete(sessionId)
  jobEventCursors.delete(sessionId)
  return { success: true, deleted: result.deleted }
}

export async function sendPluginAgentMessage(
  sessionId: string,
  payload: {
    content: string
    agentConfig: PluginAgentAgentConfig
  },
): Promise<PluginAgentSessionResponse> {
  const result = await apiClient.post<{ session: PluginAgentSession }>(
    pluginAgentSessionEndpoint(sessionId, '/messages'),
    { content: payload.content },
  )
  return { success: true, session: result.session }
}

export async function lockPluginAgentTarget(
  sessionId: string,
  proposal: PluginAgentTargetProposal,
): Promise<PluginAgentSessionResponse> {
  const result = await apiClient.post<{ session: PluginAgentSession }>(
    pluginAgentSessionEndpoint(sessionId, '/lock-target'),
    { proposal },
  )
  return { success: true, session: result.session }
}

export async function startPluginAgentExecution(
  sessionId: string,
  _agentConfig: PluginAgentAgentConfig,
): Promise<PluginAgentSessionResponse> {
  const result = await apiClient.post<{
    session: PluginAgentSession
    jobId: string
  }>(
    pluginAgentSessionEndpoint(sessionId, '/start'),
    {},
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
  result.session.job_id = result.jobId
  sessionJobs.set(sessionId, result.jobId)
  return { success: true, session: result.session }
}

export async function cancelPluginAgentExecution(
  sessionId: string,
): Promise<{ success: boolean; cancelled: boolean }> {
  const jobId = sessionJobs.get(sessionId)
  if (!jobId) return { success: false, cancelled: false }
  await jobsApi.cancel(jobId)
  return { success: true, cancelled: true }
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
  const jobId = sessionJobs.get(sessionId)
  if (!jobId) {
    options.onError('插件 Agent 任务 ID 不存在')
    return
  }
  try {
    const cursor = jobEventCursors.get(sessionId) || 0
    const response = await apiClient.get<{
      items: Array<{
        eventId: number
        type: string
        payload: Record<string, unknown>
        createdAt: string
      }>
    }>(
      `/api/v2/jobs/${encodeURIComponent(jobId)}/events?after=${cursor}&limit=200`,
      { signal: options.signal },
    )
    let sawPluginTerminal = false
    for (const event of response.items) {
      jobEventCursors.set(
        sessionId,
        Math.max(
          jobEventCursors.get(sessionId) || 0,
          event.eventId,
        ),
      )
      if (event.type.startsWith('plugin_agent_')) {
        const type = event.type.slice('plugin_agent_'.length)
        sawPluginTerminal ||= type === 'done' || type === 'error'
        options.onEvent({
          id: event.eventId,
          type,
          payload: event.payload,
          timestamp: event.createdAt,
        })
        continue
      }
      if (event.type !== 'job_finished' || sawPluginTerminal) continue
      const status = String(event.payload.status || event.payload.to || '')
      const runState = status === 'cancelled' ? 'cancelled' : 'failed'
      if (!['cancelled', 'completed_with_errors', 'failed', 'interrupted'].includes(status)) {
        continue
      }
      options.onEvent({
        id: event.eventId,
        type: 'error',
        payload: {
          run_state: runState,
          message: status === 'cancelled'
            ? '插件 Agent 任务已取消'
            : '插件 Agent 执行未成功，请在任务中心查看错误详情',
        },
        timestamp: event.createdAt,
      })
    }
    if (response.items.length === 0) {
      await new Promise(resolve => window.setTimeout(resolve, 750))
    }
  } catch (error) {
    if (!options.signal?.aborted) {
      options.onError(error instanceof Error ? error.message : '任务事件加载失败')
    }
  }
}
