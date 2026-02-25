/**
 * 翻译页工作流模式类型定义
 * 用于左侧栏统一的「模式选择器 + 启动按钮」
 */

/**
 * 工作流模式
 */
export type WorkflowMode =
  | 'translate-current'
  | 'translate-batch'
  | 'hq-batch'
  | 'proofread-batch'
  | 'remove-current'
  | 'remove-batch'
  | 'retry-failed'
  | 'delete-current'
  | 'clear-all'

/**
 * 页面范围
 */
export interface WorkflowRange {
  startPage: number
  endPage: number
}

/**
 * 启动工作流请求
 */
export interface WorkflowRunRequest {
  mode: WorkflowMode
  range?: WorkflowRange
}

/**
 * 工作流模式展示配置
 */
export interface WorkflowModeConfig {
  mode: WorkflowMode
  label: string
  startLabel: string
  supportsRange: boolean
  isDangerous: boolean
}

/**
 * 默认工作流模式
 */
export const DEFAULT_WORKFLOW_MODE: WorkflowMode = 'translate-current'

/**
 * 工作流模式配置表
 */
export const WORKFLOW_MODE_CONFIGS: WorkflowModeConfig[] = [
  {
    mode: 'translate-current',
    label: '翻译当前图片',
    startLabel: '启动翻译当前图片',
    supportsRange: false,
    isDangerous: false
  },
  {
    mode: 'translate-batch',
    label: '翻译所有图片',
    startLabel: '启动批量翻译',
    supportsRange: true,
    isDangerous: false
  },
  {
    mode: 'hq-batch',
    label: '高质量翻译',
    startLabel: '启动高质量翻译',
    supportsRange: true,
    isDangerous: false
  },
  {
    mode: 'proofread-batch',
    label: 'AI 校对',
    startLabel: '启动 AI 校对',
    supportsRange: true,
    isDangerous: false
  },
  {
    mode: 'remove-current',
    label: '仅消除当前文字',
    startLabel: '启动当前图片消字',
    supportsRange: false,
    isDangerous: false
  },
  {
    mode: 'remove-batch',
    label: '消除所有图片文字',
    startLabel: '启动批量消字',
    supportsRange: true,
    isDangerous: false
  },
  {
    mode: 'retry-failed',
    label: '重新翻译失败图片',
    startLabel: '启动失败重试',
    supportsRange: false,
    isDangerous: false
  },
  {
    mode: 'delete-current',
    label: '删除当前图片',
    startLabel: '删除当前图片',
    supportsRange: false,
    isDangerous: true
  },
  {
    mode: 'clear-all',
    label: '清除所有图片',
    startLabel: '清除所有图片',
    supportsRange: false,
    isDangerous: true
  }
]
