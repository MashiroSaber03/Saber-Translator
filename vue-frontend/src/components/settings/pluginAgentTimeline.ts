import type {
  PluginAgentAssistantDeltaPayload,
  PluginAgentAssistantPayload,
  PluginAgentDonePayload,
  PluginAgentErrorPayload,
  PluginAgentEvent,
  PluginAgentLogPayload,
  PluginAgentStatePayload,
  PluginAgentToolCallPayload,
  PluginAgentToolResultPayload,
  PluginAgentValidationPayload,
} from '@/api/pluginAgent'

interface PluginAgentStepDetail {
  label: string
  content: string
}

export interface PluginAgentTimelineItem {
  id: string
  kind: 'assistant' | 'done' | 'error' | 'log' | 'state' | 'tool' | 'validation'
  badge: string
  title: string
  summary: string
  content: string
  markdown: boolean
  status: 'error' | 'info' | 'streaming' | 'success' | 'waiting'
  timestampLabel: string
  details: PluginAgentStepDetail[]
}

export function buildTimelineItems(
  events: PluginAgentEvent[],
  displayContentMap: Record<string, string>,
  displayTargetMap: Record<string, string>,
): PluginAgentTimelineItem[] {
  const items: PluginAgentTimelineItem[] = []
  const assistantItems = new Map<string, PluginAgentTimelineItem>()
  const toolItems = new Map<string, PluginAgentTimelineItem>()

  for (const event of events) {
    if (event.type === 'assistant_delta') {
      const payload = event.payload as PluginAgentAssistantDeltaPayload
      let item = assistantItems.get(payload.stream_id)
      const displayContent = displayContentMap[payload.stream_id] || payload.content || payload.delta
      if (!item) {
        item = {
          id: `assistant-${payload.stream_id}`,
          kind: 'assistant',
          badge: 'Agent',
          title: '正在编写插件',
          summary: 'Agent 正在输出当前开发说明',
          content: displayContent,
          markdown: true,
          status: 'streaming',
          timestampLabel: formatTimestamp(event.timestamp),
          details: [],
        }
        assistantItems.set(payload.stream_id, item)
        items.push(item)
      } else {
        item.content = displayContent
        item.timestampLabel = formatTimestamp(event.timestamp)
      }
      continue
    }

    if (event.type === 'assistant') {
      const payload = event.payload as PluginAgentAssistantPayload
      if (payload.phase === 'planning') {
        continue
      }
      const streamId = payload.stream_id || `assistant-${event.id}`
      const displayContent = displayContentMap[streamId] || payload.message
      const targetContent = displayTargetMap[streamId] || payload.message
      let item = assistantItems.get(streamId)
      if (!item) {
        item = {
          id: `assistant-${streamId}`,
          kind: 'assistant',
          badge: 'Agent',
          title: '开发说明',
          summary: 'Agent 给出了当前执行说明',
          content: displayContent,
          markdown: true,
          status: displayContent === targetContent ? 'success' : 'streaming',
          timestampLabel: formatTimestamp(event.timestamp),
          details: [],
        }
        assistantItems.set(streamId, item)
        items.push(item)
      } else {
        item.content = displayContent
        item.status = displayContent === targetContent ? 'success' : 'streaming'
        item.timestampLabel = formatTimestamp(event.timestamp)
      }
      continue
    }

    if (event.type === 'tool_call') {
      const payload = event.payload as PluginAgentToolCallPayload
      const details: PluginAgentStepDetail[] = []
      if (payload.args_preview && Object.keys(payload.args_preview).length > 0) {
        details.push({
          label: '参数摘要',
          content: formatEventPayload(payload.args_preview),
        })
      }
      const item: PluginAgentTimelineItem = {
        id: `tool-${payload.group_id}`,
        kind: 'tool',
        badge: '工具',
        title: payload.summary || payload.tool,
        summary: payload.summary || payload.tool,
        content: '',
        markdown: false,
        status: 'streaming',
        timestampLabel: formatTimestamp(event.timestamp),
        details,
      }
      toolItems.set(payload.group_id, item)
      items.push(item)
      continue
    }

    if (event.type === 'tool_result') {
      const payload = event.payload as PluginAgentToolResultPayload
      const item = toolItems.get(payload.group_id)
      const details: PluginAgentStepDetail[] = []
      if (payload.changed_files && payload.changed_files.length > 0) {
        details.push({
          label: '触达文件',
          content: payload.changed_files.join('\n'),
        })
      }
      if (item) {
        item.summary = payload.summary
        item.status = payload.success ? 'success' : 'error'
        item.timestampLabel = formatTimestamp(event.timestamp)
        item.details = [...item.details, ...details]
      } else {
        items.push({
          id: `tool-result-${payload.group_id}`,
          kind: 'tool',
          badge: '工具',
          title: payload.summary || payload.tool,
          summary: payload.summary || payload.tool,
          content: '',
          markdown: false,
          status: payload.success ? 'success' : 'error',
          timestampLabel: formatTimestamp(event.timestamp),
          details,
        })
      }
      continue
    }

    if (event.type === 'validation') {
      const payload = event.payload as PluginAgentValidationPayload
      items.push({
        id: `validation-${event.id}`,
        kind: 'validation',
        badge: '校验',
        title: payload.success ? '插件校验通过' : '插件校验失败',
        summary: payload.summary,
        content: '',
        markdown: false,
        status: payload.success ? 'success' : 'error',
        timestampLabel: formatTimestamp(event.timestamp),
        details: [],
      })
      continue
    }

    if (event.type === 'done') {
      const payload = event.payload as PluginAgentDonePayload
      items.push({
        id: `done-${event.id}`,
        kind: 'done',
        badge: '完成',
        title: payload.summary || '插件开发任务已完成',
        summary: '插件已通过校验并完成刷新。',
        content: payload.message,
        markdown: true,
        status: 'success',
        timestampLabel: formatTimestamp(event.timestamp),
        details: [],
      })
      continue
    }

    if (event.type === 'error') {
      const payload = event.payload as PluginAgentErrorPayload
      items.push({
        id: `error-${event.id}`,
        kind: 'error',
        badge: '错误',
        title: payload.summary || '插件开发任务失败',
        summary: payload.message,
        content: '',
        markdown: false,
        status: 'error',
        timestampLabel: formatTimestamp(event.timestamp),
        details: [],
      })
      continue
    }

    if (event.type === 'log') {
      const payload = event.payload as PluginAgentLogPayload
      items.push({
        id: `log-${event.id}`,
        kind: 'log',
        badge: '日志',
        title: '运行日志',
        summary: payload.message,
        content: '',
        markdown: false,
        status: 'info',
        timestampLabel: formatTimestamp(event.timestamp),
        details: [],
      })
      continue
    }

    if (event.type === 'state') {
      const payload = event.payload as PluginAgentStatePayload
      if (payload.run_state === 'drafting') {
        continue
      }
      items.push({
        id: `state-${event.id}`,
        kind: 'state',
        badge: '状态',
        title: payload.label || payload.run_state,
        summary: payload.message || '',
        content: '',
        markdown: false,
        status: mapRunStateToCardStatus(payload.run_state),
        timestampLabel: formatTimestamp(event.timestamp),
        details: [],
      })
    }
  }

  return items
}

function mapRunStateToCardStatus(runState: string): PluginAgentTimelineItem['status'] {
  if (runState === 'failed' || runState === 'cancelled') {
    return 'error'
  }
  if (runState === 'completed') {
    return 'success'
  }
  if (runState === 'running') {
    return 'streaming'
  }
  if (runState === 'awaiting_target_lock' || runState === 'ready') {
    return 'waiting'
  }
  return 'info'
}

function formatTimestamp(timestamp: string): string {
  const match = timestamp.match(/T(\d{2}:\d{2}:\d{2})/)
  return match?.[1] || timestamp
}

function formatEventPayload(payload: unknown): string {
  return JSON.stringify(payload, null, 2)
}
