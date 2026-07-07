import type { PipelineConfig, ExecutionScope, PageSelection } from '../core/types'
import { createModeConfig, type PageModeOptions } from './modeConfig'

export interface StandardModeOptions extends PageModeOptions {
  pageSelection?: PageSelection
}

export function getStandardModeConfig(
  scope: ExecutionScope = 'current',
  options?: StandardModeOptions,
): PipelineConfig {
  return createModeConfig('standard', scope, options)
}
