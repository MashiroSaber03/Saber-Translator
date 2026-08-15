import type { OpenAICompatibleOptions } from '@/types/settings'
import { deepClone } from './deepClone'

export const DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES = 1

export interface OpenAICompatibleOptionsWire {
  request: {
    force_json_output: boolean
    temperature: number | null
    extra_body: Record<string, unknown>
  }
  execution: {
    use_stream: boolean
    rpm_limit: number
    transport_retries: number
    business_retries: number
  }
}

function cloneRecordOrUndefined(
  value: Record<string, unknown> | undefined,
): Record<string, unknown> | undefined {
  return value === undefined ? undefined : deepClone(value)
}

export interface OpenAiOptionsOverrides {
  request?: Partial<OpenAICompatibleOptions['request']>
  execution?: Partial<OpenAICompatibleOptions['execution']>
}

export function createDefaultOpenAiOptions(
  overrides?: OpenAiOptionsOverrides,
): OpenAICompatibleOptions {
  return {
    request: {
      forceJsonOutput: false,
      ...overrides?.request,
    },
    execution: {
      useStream: false,
      rpmLimit: 0,
      transportRetries: DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
      businessRetries: 0,
      ...overrides?.execution,
    },
  }
}

export function cloneOpenAiOptions(options: OpenAICompatibleOptions): OpenAICompatibleOptions {
  return deepClone(options)
}

export interface OpenAiOptionsPatch {
  rpmLimit?: number
  transportRetries?: number
  businessRetries?: number
  forceJsonOutput?: boolean
  useStream?: boolean
  extraBody?: Record<string, unknown>
}

const OPENAI_OPTIONS_PATCH_FIELD_NAMES = [
  'rpmLimit',
  'transportRetries',
  'businessRetries',
  'forceJsonOutput',
  'useStream',
  'extraBody',
] as const

export function omitOpenAiOptionsPatchFields<T extends object>(
  updates: T
): Omit<T, keyof OpenAiOptionsPatch> {
  const scopedUpdates = { ...updates } as Record<string, unknown>
  for (const fieldName of OPENAI_OPTIONS_PATCH_FIELD_NAMES) {
    delete scopedUpdates[fieldName]
  }

  return scopedUpdates as Omit<T, keyof OpenAiOptionsPatch>
}

export function applyOpenAiOptionsPatch(
  options: OpenAICompatibleOptions,
  updates: OpenAiOptionsPatch
): OpenAICompatibleOptions {
  if (updates.rpmLimit !== undefined) options.execution.rpmLimit = updates.rpmLimit
  if (updates.transportRetries !== undefined) {
    options.execution.transportRetries = updates.transportRetries
  }
  if (updates.businessRetries !== undefined) {
    options.execution.businessRetries = updates.businessRetries
  }
  if (updates.forceJsonOutput !== undefined) {
    options.request.forceJsonOutput = updates.forceJsonOutput
  }
  if (updates.useStream !== undefined) options.execution.useStream = updates.useStream
  if (Object.prototype.hasOwnProperty.call(updates, 'extraBody')) {
    options.request.extraBody = cloneRecordOrUndefined(updates.extraBody)
  }

  return options
}

export function deserializeOpenAICompatibleOptionsFromApi(
  wire: OpenAICompatibleOptionsWire,
): OpenAICompatibleOptions {
  return {
    request: {
      forceJsonOutput: wire.request.force_json_output,
      ...(wire.request.temperature === null
        ? {}
        : { temperature: wire.request.temperature }),
      extraBody: deepClone(wire.request.extra_body),
    },
    execution: {
      useStream: wire.execution.use_stream,
      rpmLimit: wire.execution.rpm_limit,
      transportRetries: wire.execution.transport_retries,
      businessRetries: wire.execution.business_retries,
    },
  }
}

export function serializeOpenAICompatibleOptionsForApi(
  options: OpenAICompatibleOptions
): OpenAICompatibleOptionsWire {
  return {
    request: {
      force_json_output: options.request.forceJsonOutput,
      temperature: options.request.temperature ?? null,
      extra_body: cloneRecordOrUndefined(options.request.extraBody) ?? {},
    },
    execution: {
      use_stream: options.execution.useStream,
      rpm_limit: options.execution.rpmLimit,
      transport_retries: options.execution.transportRetries,
      business_retries: options.execution.businessRetries,
    },
  }
}
