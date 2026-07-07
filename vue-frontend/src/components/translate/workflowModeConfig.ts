import type { WorkflowMode } from '@/types/workflow'

export interface WorkflowModeConfig {
  mode: WorkflowMode
  label: string
  startLabel: string
  supportsPageSelection: boolean
  isDangerous: boolean
}

export const DEFAULT_WORKFLOW_MODE: WorkflowMode = 'translate-current'

export const WORKFLOW_MODE_CONFIGS: WorkflowModeConfig[] = [
  {
    mode: 'translate-current',
    label: '翻译当前图片',
    startLabel: '启动翻译当前图片',
    supportsPageSelection: false,
    isDangerous: false,
  },
  {
    mode: 'translate-batch',
    label: '翻译所有图片',
    startLabel: '启动批量翻译',
    supportsPageSelection: true,
    isDangerous: false,
  },
  {
    mode: 'hq-batch',
    label: '高质量翻译',
    startLabel: '启动高质量翻译',
    supportsPageSelection: true,
    isDangerous: false,
  },
  {
    mode: 'proofread-batch',
    label: 'AI 校对',
    startLabel: '启动 AI 校对',
    supportsPageSelection: true,
    isDangerous: false,
  },
  {
    mode: 'remove-current',
    label: '仅消除当前文字',
    startLabel: '启动当前图片消字',
    supportsPageSelection: false,
    isDangerous: false,
  },
  {
    mode: 'remove-batch',
    label: '消除所有图片文字',
    startLabel: '启动批量消字',
    supportsPageSelection: true,
    isDangerous: false,
  },
  {
    mode: 'retry-failed',
    label: '重新翻译失败图片',
    startLabel: '启动失败重试',
    supportsPageSelection: false,
    isDangerous: false,
  },
  {
    mode: 'delete-current',
    label: '删除当前图片',
    startLabel: '删除当前图片',
    supportsPageSelection: false,
    isDangerous: true,
  },
  {
    mode: 'clear-all',
    label: '清除所有图片',
    startLabel: '清除所有图片',
    supportsPageSelection: false,
    isDangerous: true,
  },
]

const WORKFLOW_MODE_VALUES = new Set<WorkflowMode>(
  WORKFLOW_MODE_CONFIGS.map(config => config.mode)
)

export function isWorkflowMode(value: unknown): value is WorkflowMode {
  return typeof value === 'string' && WORKFLOW_MODE_VALUES.has(value as WorkflowMode)
}
