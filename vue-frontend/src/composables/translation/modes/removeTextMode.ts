import type { PipelineConfig, ExecutionScope, PageSelection } from '../core/types'
import { createModeConfig, type PageModeOptions } from './modeConfig'

export interface RemoveTextModeOptions extends PageModeOptions {
  pageSelection?: PageSelection
}

export function getRemoveTextModeConfig(
  scope: ExecutionScope = 'current',
  options?: RemoveTextModeOptions,
): PipelineConfig {
  return createModeConfig('removeText', scope, options)
}
