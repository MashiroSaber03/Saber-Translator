import type {
  BatchOptions,
  ExecutionScope,
  PageSelection,
  PipelineConfig,
  TranslationMode,
} from '../core/types'

export interface PageModeOptions {
  pageSelection?: PageSelection
}

export type BatchModeOptions = PageModeOptions & BatchOptions

const DEFAULT_BATCH_OPTIONS: Required<BatchOptions> = {
  batchSize: 3,
  maxRetries: 2,
  rpmLimit: 10,
}

export function createModeConfig(
  mode: TranslationMode,
  scope: ExecutionScope,
  options?: PageModeOptions,
): PipelineConfig {
  return {
    mode,
    scope,
    pageSelection: options?.pageSelection,
  }
}

export function createBatchModeConfig(
  mode: Extract<TranslationMode, 'hq' | 'proofread'>,
  scope: ExecutionScope,
  options?: BatchModeOptions,
): PipelineConfig {
  return {
    ...createModeConfig(mode, scope, options),
    batchOptions: {
      batchSize: options?.batchSize ?? DEFAULT_BATCH_OPTIONS.batchSize,
      maxRetries: options?.maxRetries ?? DEFAULT_BATCH_OPTIONS.maxRetries,
      rpmLimit: options?.rpmLimit ?? DEFAULT_BATCH_OPTIONS.rpmLimit,
    },
  }
}
