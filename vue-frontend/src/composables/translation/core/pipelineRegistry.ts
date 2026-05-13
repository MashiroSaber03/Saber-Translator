import type { TranslationMode } from './types'

export type AtomicStepType =
  | 'detection'
  | 'ocr'
  | 'color'
  | 'autoGlossary'
  | 'translate'
  | 'aiTranslate'
  | 'inpaint'
  | 'render'
  | 'save'

type BaseAtomicStepType = Exclude<AtomicStepType, 'save'>
export type ParallelPoolStepName = Exclude<AtomicStepType, 'aiTranslate'>

export interface PipelineModeDefinition {
  baseStepChain: BaseAtomicStepType[]
}

export interface PipelineResolutionFlags {
  removeTextWithOcr: boolean
  autoSaveEnabled: boolean
}

const PIPELINE_MODE_DEFINITIONS: Record<TranslationMode, PipelineModeDefinition> = {
  standard: {
    baseStepChain: ['detection', 'ocr', 'color', 'autoGlossary', 'translate', 'inpaint', 'render'],
  },
  hq: {
    baseStepChain: ['detection', 'ocr', 'color', 'autoGlossary', 'aiTranslate', 'inpaint', 'render'],
  },
  proofread: {
    baseStepChain: ['aiTranslate', 'render'],
  },
  removeText: {
    baseStepChain: ['detection', 'inpaint', 'render'],
  },
}

const STEP_LABELS: Record<AtomicStepType, string> = {
  detection: '气泡检测',
  ocr: '文字识别',
  color: '颜色提取',
  autoGlossary: '自动术语提取',
  translate: '翻译',
  aiTranslate: 'AI翻译',
  inpaint: '背景修复',
  render: '渲染',
  save: '保存',
}

function cloneStepChain<T extends AtomicStepType | BaseAtomicStepType | ParallelPoolStepName>(
  stepChain: readonly T[],
): T[] {
  return [...stepChain]
}

export const STEP_CHAIN_CONFIGS: Record<TranslationMode, AtomicStepType[]> = {
  standard: getBaseStepChain('standard'),
  hq: getBaseStepChain('hq'),
  proofread: getBaseStepChain('proofread'),
  removeText: getBaseStepChain('removeText'),
}

export function getBaseStepChain(mode: TranslationMode): AtomicStepType[] {
  return cloneStepChain(PIPELINE_MODE_DEFINITIONS[mode].baseStepChain)
}

export function getStepLabel(step: AtomicStepType): string {
  return STEP_LABELS[step]
}

export function resolveSequentialStepChain(
  mode: TranslationMode,
  flags: PipelineResolutionFlags,
): AtomicStepType[] {
  const chainConfig = getBaseStepChain(mode)

  if (mode === 'removeText' && flags.removeTextWithOcr) {
    const detectionIndex = chainConfig.indexOf('detection')
    if (detectionIndex !== -1 && !chainConfig.includes('ocr')) {
      chainConfig.splice(detectionIndex + 1, 0, 'ocr')
    }
  }

  if (flags.autoSaveEnabled) {
    chainConfig.push('save')
  }

  return chainConfig
}

export function resolveParallelPoolChain(
  mode: TranslationMode,
  flags: PipelineResolutionFlags,
): ParallelPoolStepName[] {
  return resolveSequentialStepChain(mode, flags).map<ParallelPoolStepName>((step) =>
    step === 'aiTranslate' ? 'translate' : step,
  )
}
