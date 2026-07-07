export interface OpenAICompatibleRequestOptions {
  forceJsonOutput: boolean
  temperature?: number
  extraBody?: Record<string, unknown>
}

export interface OpenAICompatibleExecutionOptions {
  useStream: boolean
  rpmLimit: number
  transportRetries: number
  businessRetries: number
}

export interface OpenAICompatibleOptions {
  request: OpenAICompatibleRequestOptions
  execution: OpenAICompatibleExecutionOptions
}
