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

  it('falls back when numeric option fields are not finite', () => {
    const frontendOptions = normalizeOpenAiOptions(
      {
        request: {
          temperature: Number.POSITIVE_INFINITY,
        },
        execution: {
          rpmLimit: 'Infinity',
          transportRetries: Number.NEGATIVE_INFINITY,
          businessRetries: Number.NaN,
        },
      },
      {
        request: {
          temperature: 0.6,
        },
        execution: {
          rpmLimit: 8,
          transportRetries: 2,
          businessRetries: 3,
        },
      }
    )

    expect(frontendOptions.request.temperature).toBe(0.6)
    expect(frontendOptions.execution.rpmLimit).toBe(8)
    expect(frontendOptions.execution.transportRetries).toBe(2)
    expect(frontendOptions.execution.businessRetries).toBe(3)

    const apiOptions = deserializeOpenAICompatibleOptionsFromApi(
      {
        request: {
          temperature: 'Infinity',
        },
        execution: {
          rpm_limit: Number.POSITIVE_INFINITY,
          transport_retries: '-Infinity',
          business_retries: Number.NaN,
        },
      },
      {
        request: {
          temperature: 0.4,
        },
        execution: {
          rpmLimit: 4,
          transportRetries: 5,
          businessRetries: 6,
        },
      }
    )

    expect(apiOptions.request.temperature).toBe(0.4)
    expect(apiOptions.execution.rpmLimit).toBe(4)
    expect(apiOptions.execution.transportRetries).toBe(5)
    expect(apiOptions.execution.businessRetries).toBe(6)
  })

  it('falls back when boolean option fields are not real booleans', () => {
    const frontendOptions = normalizeOpenAiOptions({
      request: {
        forceJsonOutput: 'false',
      },
      execution: {
        useStream: 'false',
      },
    })

    expect(frontendOptions.request.forceJsonOutput).toBe(false)
    expect(frontendOptions.execution.useStream).toBe(false)

    const apiOptions = deserializeOpenAICompatibleOptionsFromApi({
      request: {
        force_json_output: 'false',
      },
      execution: {
        use_stream: 'false',
      },
    })

    expect(apiOptions.request.forceJsonOutput).toBe(false)
    expect(apiOptions.execution.useStream).toBe(false)
  })
})
