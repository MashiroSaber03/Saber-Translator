import type { OpenAICompatibleOptions } from '@/types/settings'

export const DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES = 1

function parseNumberOrFallback(value: unknown, fallback: number): number {
  if (value === undefined || value === null || value === '') return fallback
  const parsed = Number(value)
  return Number.isNaN(parsed) ? fallback : parsed
}

export function createDefaultOpenAiOptions(
  overrides?: Partial<OpenAICompatibleOptions>
): OpenAICompatibleOptions {
  return {
    request: {
      forceJsonOutput: false,
      ...overrides?.request
    },
    execution: {
      useStream: false,
      rpmLimit: 0,
      transportRetries: DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
      businessRetries: 0,
      ...overrides?.execution
    }
  }
}

export function cloneOpenAiOptions(options: OpenAICompatibleOptions): OpenAICompatibleOptions {
  return createDefaultOpenAiOptions(JSON.parse(JSON.stringify(options)))
}

export function normalizeOpenAiOptions(
  raw: unknown,
  legacy?: {
    forceJsonOutput?: unknown
    isJsonMode?: unknown
    temperature?: unknown
    useStream?: unknown
    rpmLimit?: unknown
    maxRetries?: unknown
    transportRetries?: unknown
    businessRetries?: unknown
  },
  defaults?: Partial<OpenAICompatibleOptions>
): OpenAICompatibleOptions {
  const normalized = createDefaultOpenAiOptions(defaults)
  const candidate = (raw && typeof raw === 'object') ? raw as Record<string, any> : {}
  const request = (candidate.request && typeof candidate.request === 'object') ? candidate.request as Record<string, any> : {}
  const execution = (candidate.execution && typeof candidate.execution === 'object') ? candidate.execution as Record<string, any> : {}

  normalized.request.forceJsonOutput = Boolean(
    request.forceJsonOutput
    ?? request.force_json_output
    ?? legacy?.forceJsonOutput
    ?? legacy?.isJsonMode
    ?? normalized.request.forceJsonOutput
  )

  const temperature = request.temperature ?? legacy?.temperature
  if (temperature !== undefined && temperature !== null && temperature !== '') {
    normalized.request.temperature = Number(temperature)
  }

  normalized.execution.useStream = Boolean(
    execution.useStream
    ?? execution.use_stream
    ?? legacy?.useStream
    ?? normalized.execution.useStream
  )

  normalized.execution.rpmLimit = parseNumberOrFallback(
    execution.rpmLimit
    ?? execution.rpm_limit
    ?? legacy?.rpmLimit,
    normalized.execution.rpmLimit
  )

  normalized.execution.transportRetries = parseNumberOrFallback(
    execution.transportRetries
    ?? execution.transport_retries
    ?? legacy?.transportRetries,
    normalized.execution.transportRetries
  )

  normalized.execution.businessRetries = parseNumberOrFallback(
    execution.businessRetries
    ?? execution.business_retries
    ?? execution.maxRetries
    ?? execution.max_retries
    ?? legacy?.businessRetries
    ?? legacy?.maxRetries,
    normalized.execution.businessRetries
  )

  return normalized
}

export function serializeOpenAICompatibleOptionsForApi(options: OpenAICompatibleOptions) {
  return {
    request: {
      force_json_output: options.request.forceJsonOutput,
      ...(options.request.temperature !== undefined ? { temperature: options.request.temperature } : {})
    },
    execution: {
      use_stream: options.execution.useStream,
      rpm_limit: options.execution.rpmLimit,
      transport_retries: options.execution.transportRetries,
      business_retries: options.execution.businessRetries
    }
  }
}
