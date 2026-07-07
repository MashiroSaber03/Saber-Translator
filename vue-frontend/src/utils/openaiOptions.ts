import type { OpenAICompatibleOptions } from '@/types/settings'
import { deepClone } from './deepClone'

export const DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES = 1

export interface OpenAICompatibleOptionsWire {
  request: {
    force_json_output: boolean
    temperature?: number
    extra_body?: Record<string, unknown>
  }
  execution: {
    use_stream: boolean
    rpm_limit: number
    transport_retries: number
    business_retries: number
  }
}

function cloneRecordOrUndefined(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  return deepClone(value as Record<string, unknown>)
}

function parseNumberOrFallback(value: unknown, fallback: number): number {
  if (value === undefined || value === null || value === '') return fallback
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function parseOptionalNumberOrFallback(value: unknown, fallback?: number): number | undefined {
  if (value === undefined || value === null || value === '') return fallback
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function parseBooleanOrFallback(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback
}

function recordOrEmpty(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

export function createDefaultOpenAiOptions(
  overrides?: Partial<OpenAICompatibleOptions>
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
  return createDefaultOpenAiOptions(deepClone(options))
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

export function normalizeOpenAiOptions(
  raw: unknown,
  defaults?: Partial<OpenAICompatibleOptions>
): OpenAICompatibleOptions {
  const normalized = createDefaultOpenAiOptions(defaults)
  const candidate = recordOrEmpty(raw)
  const request = recordOrEmpty(candidate.request)
  const execution = recordOrEmpty(candidate.execution)

  normalized.request.forceJsonOutput = parseBooleanOrFallback(
    request.forceJsonOutput,
    normalized.request.forceJsonOutput
  )

  const temperature = request.temperature
  if (temperature !== undefined && temperature !== null && temperature !== '') {
    normalized.request.temperature = parseOptionalNumberOrFallback(
      temperature,
      normalized.request.temperature
    )
  }

  normalized.request.extraBody = cloneRecordOrUndefined(request.extraBody)

  normalized.execution.useStream = parseBooleanOrFallback(
    execution.useStream,
    normalized.execution.useStream
  )

  normalized.execution.rpmLimit = parseNumberOrFallback(
    execution.rpmLimit ?? normalized.execution.rpmLimit,
    normalized.execution.rpmLimit
  )

  normalized.execution.transportRetries = parseNumberOrFallback(
    execution.transportRetries ?? normalized.execution.transportRetries,
    normalized.execution.transportRetries
  )

  normalized.execution.businessRetries = parseNumberOrFallback(
    execution.businessRetries ?? normalized.execution.businessRetries,
    normalized.execution.businessRetries
  )

  return normalized
}

export function deserializeOpenAICompatibleOptionsFromApi(
  raw: unknown,
  defaults?: Partial<OpenAICompatibleOptions>
): OpenAICompatibleOptions {
  const normalized = createDefaultOpenAiOptions(defaults)
  const candidate = recordOrEmpty(raw)
  const request = recordOrEmpty(candidate.request)
  const execution = recordOrEmpty(candidate.execution)

  normalized.request.forceJsonOutput = parseBooleanOrFallback(
    request.force_json_output,
    normalized.request.forceJsonOutput
  )

  const temperature = request.temperature
  if (temperature !== undefined && temperature !== null && temperature !== '') {
    normalized.request.temperature = parseOptionalNumberOrFallback(
      temperature,
      normalized.request.temperature
    )
  }

  normalized.request.extraBody = cloneRecordOrUndefined(request.extra_body)

  normalized.execution.useStream = parseBooleanOrFallback(
    execution.use_stream,
    normalized.execution.useStream
  )
  normalized.execution.rpmLimit = parseNumberOrFallback(
    execution.rpm_limit ?? normalized.execution.rpmLimit,
    normalized.execution.rpmLimit
  )
  normalized.execution.transportRetries = parseNumberOrFallback(
    execution.transport_retries ?? normalized.execution.transportRetries,
    normalized.execution.transportRetries
  )
  normalized.execution.businessRetries = parseNumberOrFallback(
    execution.business_retries ?? normalized.execution.businessRetries,
    normalized.execution.businessRetries
  )

  return normalized
}

export function serializeOpenAICompatibleOptionsForApi(
  options: OpenAICompatibleOptions
): OpenAICompatibleOptionsWire {
  return {
    request: {
      force_json_output: options.request.forceJsonOutput,
      ...(options.request.temperature !== undefined
        ? { temperature: options.request.temperature }
        : {}),
      ...(options.request.extraBody !== undefined
        ? { extra_body: cloneRecordOrUndefined(options.request.extraBody) }
        : {}),
    },
    execution: {
      use_stream: options.execution.useStream,
      rpm_limit: options.execution.rpmLimit,
      transport_retries: options.execution.transportRetries,
      business_retries: options.execution.businessRetries,
    },
  }
}
