import type { CharacterStudioChatSession } from '@/types/characterStudio'
import { findRegenerationUserMessageIndex } from '@/stores/characterStudioChatSession'

type StudioChatMessage = CharacterStudioChatSession['messages'][number]

export interface PendingAttachmentCard {
  id: string
  file: File
  previewUrl: string
}

export function canEditStudioChatMessage(message: StudioChatMessage): boolean {
  return message.role === 'user'
}

export function canRegenerateStudioChatMessage(
  message: StudioChatMessage, messages: StudioChatMessage[],
): boolean {
  return message.role === 'assistant' && findRegenerationUserMessageIndex(messages, message.message_id) >= 0
}

export function attachmentTypeLabel(mimeType: string): string {
  if (mimeType.startsWith('image/')) return '图片'
  return mimeType.split('/').pop() || '附件'
}

export function formatSessionTime(value: string): string {
  if (!value) return '未知时间'
  return new Date(value).toLocaleString('zh-CN', { hour12: false })
}

export function summarizeStudioRuntimeLog(item: Record<string, unknown>): string {
  if (item.type === 'regex') return `命中正则: ${String(item.scriptName || item.pattern || '未知脚本')}`
  if (item.type === 'lorebook') return `命中世界书: ${String(item.comment || '未命名条目')}`
  if (item.type === 'task') return `执行任务: ${String(item.name || '未命名任务')}`
  if (item.type === 'task_error') return `任务未执行: ${String(item.name || '未命名任务')} · ${String(item.message || '')}`
  return JSON.stringify(item)
}
