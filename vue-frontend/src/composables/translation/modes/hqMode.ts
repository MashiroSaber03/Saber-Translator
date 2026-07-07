import type { PipelineConfig, ExecutionScope } from '../core/types'
import { createBatchModeConfig, type BatchModeOptions } from './modeConfig'

export type HqModeOptions = BatchModeOptions

export function getHqModeConfig(
  scope: ExecutionScope = 'all',
  options?: HqModeOptions,
): PipelineConfig {
  return createBatchModeConfig('hq', scope, options)
}
