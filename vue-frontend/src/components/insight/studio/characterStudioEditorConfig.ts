import type { UiIconName } from '@/components/ui/iconRegistry'

type StudioTabItem<Value extends string> = {
  value: Value
  label: string
  iconName: UiIconName
  glyph: string
}

export const editorTabItems = [
  { value: 'overview', label: '概览', iconName: 'target', glyph: '◈' },
  { value: 'character', label: '角色设定', iconName: 'users', glyph: '🧬' },
  { value: 'greetings', label: '问候语', iconName: 'message', glyph: '💬' },
  { value: 'lorebook', label: '世界书', iconName: 'book-open', glyph: '📚' },
  { value: 'scripts', label: '脚本任务', iconName: 'settings', glyph: '⚙️' },
  { value: 'export', label: '导出诊断', iconName: 'download', glyph: '⬇' },
] as const satisfies ReadonlyArray<StudioTabItem<'overview' | 'character' | 'greetings' | 'lorebook' | 'scripts' | 'export'>>

export const scriptTabItems = [
  { value: 'regex', label: '正则脚本', iconName: 'file-text', glyph: '∑' },
  { value: 'tasks', label: '状态任务', iconName: 'clock', glyph: '∞' },
] as const satisfies ReadonlyArray<StudioTabItem<'regex' | 'tasks'>>

export const freezeItems = [
  { key: 'identity', label: '角色设定' },
  { key: 'greetings', label: '问候语' },
  { key: 'lorebook', label: '世界书' },
  { key: 'regex', label: '正则脚本' },
  { key: 'state-tasks', label: '状态任务' },
] as const

export const studioPreviewTabs = [
  { value: 'chat', label: '聊天', iconName: 'message', glyph: '💬' },
  { value: 'assistant', label: '卡片助手', iconName: 'sparkles', glyph: '🧠' },
  { value: 'runtime', label: '运行日志', iconName: 'file-text', glyph: '📟' },
] as const satisfies ReadonlyArray<StudioTabItem<'chat' | 'assistant' | 'runtime'>>
