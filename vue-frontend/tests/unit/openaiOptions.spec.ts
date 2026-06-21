import { describe, expect, it } from 'vitest'

import {
  deserializeOpenAICompatibleOptionsFromApi,
  normalizeOpenAiOptions,
  serializeOpenAICompatibleOptionsForApi,
} from '@/utils/openaiOptions'

describe('openai options schema boundaries', () => {
  it('normalizes only the current frontend option shape', () => {
    const options = normalizeOpenAiOptions({
      request: {
        force_json_output: true,
        extra_body: { top_p: 0.9 },
      },
      execution: {
        use_stream: true,
        rpm_limit: 12,
        transport_retries: 2,
        business_retries: 3,
      },
    })

    expect(options.request.forceJsonOutput).toBe(false)
    expect(options.request.extraBody).toBeUndefined()
    expect(options.execution.useStream).toBe(false)
    expect(options.execution.rpmLimit).toBe(0)
    expect(options.execution.transportRetries).toBe(1)
    expect(options.execution.businessRetries).toBe(0)
  })

  it('serializes current frontend options to the backend wire shape', () => {
    const payload = serializeOpenAICompatibleOptionsForApi({
      request: {
        forceJsonOutput: true,
        temperature: 0.2,
        extraBody: { top_p: 0.9 },
      },
      execution: {
        useStream: true,
        rpmLimit: 12,
        transportRetries: 2,
        businessRetries: 3,
      },
    })

    expect(payload).toEqual({
      request: {
        force_json_output: true,
        temperature: 0.2,
        extra_body: { top_p: 0.9 },
      },
      execution: {
        use_stream: true,
        rpm_limit: 12,
        transport_retries: 2,
        business_retries: 3,
      },
    })
  })

  it('deserializes backend wire options through the API adapter', () => {
    const options = deserializeOpenAICompatibleOptionsFromApi({
      request: {
        force_json_output: true,
        temperature: 0.2,
        extra_body: { top_p: 0.9 },
      },
      execution: {
        use_stream: true,
        rpm_limit: 12,
        transport_retries: 2,
        business_retries: 3,
      },
    })

    expect(options).toEqual({
      request: {
        forceJsonOutput: true,
        temperature: 0.2,
        extraBody: { top_p: 0.9 },
      },
      execution: {
        useStream: true,
        rpmLimit: 12,
        transportRetries: 2,
        businessRetries: 3,
      },
    })
  })
})
