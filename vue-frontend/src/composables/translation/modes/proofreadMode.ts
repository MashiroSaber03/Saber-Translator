import type { PipelineConfig, ExecutionScope } from '../core/types'
import { createBatchModeConfig, type BatchModeOptions } from './modeConfig'

export type ProofreadModeOptions = BatchModeOptions

export function getProofreadModeConfig(
  scope: ExecutionScope = 'all',
  options?: ProofreadModeOptions,
): PipelineConfig {
  return createBatchModeConfig('proofread', scope, options)
}
