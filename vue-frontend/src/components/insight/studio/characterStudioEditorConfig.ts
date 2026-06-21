export const editorTabItems = [
  { value: 'overview', label: '概览', icon: '◈' },
  { value: 'character', label: '角色设定', icon: '🧬' },
  { value: 'greetings', label: '问候语', icon: '💬' },
  { value: 'lorebook', label: '世界书', icon: '📚' },
  { value: 'scripts', label: '脚本任务', icon: '⚙️' },
  { value: 'export', label: '导出诊断', icon: '⬇' },
] as const

export const scriptTabItems = [
  { value: 'regex', label: '正则脚本', icon: '∑' },
  { value: 'tasks', label: '状态任务', icon: '∞' },
] as const

export const freezeItems = [
  { key: 'identity', label: '角色设定' },
  { key: 'greetings', label: '问候语' },
  { key: 'lorebook', label: '世界书' },
  { key: 'regex', label: '正则脚本' },
  { key: 'state-tasks', label: '状态任务' },
] as const

export const studioPreviewTabs = [
  { value: 'chat', label: '聊天', icon: '💬' },
  { value: 'assistant', label: '卡片助手', icon: '🧠' },
  { value: 'runtime', label: '运行日志', icon: '📟' },
] as const
