import { apiClient } from './client'
import { getProviderOptionsForCapability } from '@/config/aiProviders'
import { getPlugins, type PluginData } from './plugin'
import { jobsApi, type V2JobEvent } from './v2/jobs'
import { newIdempotencyKey } from './v2/content'
import { assertBackendActionAllowed } from '@/services/backendAccessGate'
import type { components } from '@/api/generated/v2'

export type PluginAgentMode = components['schemas']['PluginAgentSession']['mode']
export type PluginAgentRunState = components['schemas']['PluginAgentSession']['run_state']
export type PluginAgentTargetProposal = components['schemas']['PluginAgentTarget']
export type PluginAgentLockedTarget = components['schemas']['PluginAgentLockedTarget']
export type PluginAgentSession = components['schemas']['PluginAgentSession']

type PluginAgentSessionEnvelope = components['schemas']['PluginAgentSessionEnvelope']
type PluginAgentSessionCreateCommand = components['schemas']['PluginAgentSessionCreateCommand']
type PluginAgentStartResult = components['schemas']['PluginAgentStartResult']
type V2DeletedResponse = components['schemas']['DeletedResponse']

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

export type PluginAgentEvent<T = PluginAgentEventPayload> = Omit<
  components['schemas']['PluginAgentEvent'],
  'payload'
> & {
  eventKey?: string
  payload: T
}

export interface PluginAgentOverviewSection {
  title: string
  items: string[]
}

export interface PluginAgentSettings {
  overview: string[]
  overview_sections: PluginAgentOverviewSection[]
  prompt_examples: string[]
  providers: Array<{ value: string; label: string }>
  plugins: PluginData[]
  session: PluginAgentSession | null
}

let activeSessionId = ''

function pluginAgentSessionEndpoint(sessionId: string, suffix = ''): string {
  return `/api/v2/plugin-agent/sessions/${encodeURIComponent(sessionId)}${suffix}`
}

export async function getPluginAgentSettings(): Promise<PluginAgentSettings> {
  const pluginResult = await getPlugins()
  let session: PluginAgentSession | null = null
  if (activeSessionId) {
    try {
      session = await getPluginAgentSession(activeSessionId)
    } catch (error) {
      if (
        error
        && typeof error === 'object'
        && 'status' in error
        && error.status === 404
      ) {
        activeSessionId = ''
      } else {
        throw error
      }
    }
  }
  return {
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
    plugins: pluginResult,
    session,
  }
}

export async function createPluginAgentSession(payload: {
  mode: PluginAgentMode
  plugin_id?: string
}): Promise<PluginAgentSession> {
  const command: PluginAgentSessionCreateCommand = {
    mode: payload.mode,
    ...(payload.plugin_id ? { pluginId: payload.plugin_id } : {}),
  }
  const result = await apiClient.post<PluginAgentSessionEnvelope>(
    '/api/v2/plugin-agent/sessions',
    command
  )
  activeSessionId = result.session.session_id
  return result.session
}

export async function getPluginAgentSession(sessionId: string): Promise<PluginAgentSession> {
  const result = await apiClient.get<PluginAgentSessionEnvelope>(
    pluginAgentSessionEndpoint(sessionId)
  )
  return result.session
}

export async function deletePluginAgentSession(sessionId: string): Promise<void> {
  await apiClient.delete<V2DeletedResponse>(pluginAgentSessionEndpoint(sessionId))
  if (activeSessionId === sessionId) activeSessionId = ''
}

export async function sendPluginAgentMessage(
  sessionId: string,
  payload: {
    content: string
  }
): Promise<PluginAgentSession> {
  assertBackendActionAllowed()
  const result = await apiClient.post<PluginAgentSessionEnvelope>(
    pluginAgentSessionEndpoint(sessionId, '/messages'),
    { content: payload.content }
  )
  return result.session
}

export async function lockPluginAgentTarget(
  sessionId: string,
  proposal: PluginAgentTargetProposal
): Promise<PluginAgentSession> {
  const result = await apiClient.post<PluginAgentSessionEnvelope>(
    pluginAgentSessionEndpoint(sessionId, '/lock-target'),
    { proposal }
  )
  return result.session
}

export async function startPluginAgentExecution(sessionId: string): Promise<PluginAgentSession> {
  assertBackendActionAllowed()
  const result = await apiClient.post<PluginAgentStartResult>(
    pluginAgentSessionEndpoint(sessionId, '/start'),
    {},
    { headers: { 'Idempotency-Key': newIdempotencyKey() } }
  )
  result.session.job_id = result.jobId
  return result.session
}

export async function cancelPluginAgentExecution(jobId: string): Promise<void> {
  await jobsApi.cancel(jobId)
}

export function pluginAgentEventFromJobEvent(event: V2JobEvent): PluginAgentEvent | null {
  if (event.type.startsWith('plugin_agent_')) {
    return {
      id: event.eventId,
      eventKey: `job:${event.eventId}`,
      type: event.type.slice('plugin_agent_'.length),
      payload: event.payload,
      timestamp: event.createdAt ?? '',
    }
  }
  const activeState = {
    job_started: ['running', '执行中'],
    job_request_pause: ['pausing', '正在暂停'],
    job_paused: ['paused', '已暂停'],
    job_resume: ['running', '执行中'],
    job_request_cancel: ['cancelling', '正在取消'],
  }[event.type] as [PluginAgentRunState, string] | undefined
  if (activeState) {
    return {
      id: event.eventId,
      eventKey: `job:${event.eventId}`,
      type: 'state',
      payload: {
        run_state: activeState[0],
        label: activeState[1],
      },
      timestamp: event.createdAt ?? '',
    }
  }
  if (!['job_failed', 'job_cancelled', 'job_finished'].includes(event.type)) return null
  const status =
    event.type === 'job_failed'
      ? 'failed'
      : event.type === 'job_cancelled'
        ? 'cancelled'
        : String(event.payload.status || '')
  if (!['cancelled', 'completed_with_errors', 'failed', 'interrupted'].includes(status)) {
    return null
  }
  return {
    id: event.eventId,
    eventKey: `job:${event.eventId}`,
    type: 'error',
    payload: {
      run_state: status === 'cancelled' ? 'cancelled' : 'failed',
      message:
        status === 'cancelled'
          ? '插件 Agent 任务已取消'
          : '插件 Agent 执行未成功，请在任务中心查看错误详情',
    },
    timestamp: event.createdAt ?? '',
  }
}

export async function listPluginAgentJobEvents(
  jobId: string,
  afterId = 0
): Promise<{ cursor: number; events: PluginAgentEvent[] }> {
  const events: PluginAgentEvent[] = []
  let sawPluginTerminal = false
  let cursor = Math.max(0, Math.floor(afterId))
  const pageSize = 1000
  while (true) {
    const response = await jobsApi.events(jobId, {
      after: cursor,
      limit: pageSize,
    })
    for (const event of response.items) {
      cursor = Math.max(cursor, event.eventId)
      const mapped = pluginAgentEventFromJobEvent(event)
      if (!mapped) continue
      if (mapped.type === 'done' || mapped.type === 'error') {
        if (sawPluginTerminal && mapped.type === 'error') continue
        sawPluginTerminal = true
      }
      events.push(mapped)
    }
    if (response.items.length < pageSize) break
  }
  return { cursor, events }
}
