import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('Insight converters', () => {
  it('lives in the utility layer and preserves nested snake/camel conversion', async () => {
    expect(existsSync(resolve(process.cwd(), 'src/utils/insightConverters.ts'))).toBe(true)

    const { configFromApi, configToApi, toCamelCase, toSnakeCase } = await import('@/utils/insightConverters')

    expect(toSnakeCase({
      imageGen: {
        baseUrl: 'https://example.test',
        customLayers: [{ unitCount: 2 }],
      },
    })).toEqual({
      image_gen: {
        base_url: 'https://example.test',
        custom_layers: [{ unit_count: 2 }],
      },
    })

    expect(toCamelCase({
      image_gen: {
        base_url: 'https://example.test',
        custom_layers: [{ unit_count: 2 }],
      },
    })).toEqual({
      imageGen: {
        baseUrl: 'https://example.test',
        customLayers: [{ unitCount: 2 }],
      },
    })

    expect(configToApi({
      llm: {
        model: 'chat',
      },
      imageGen: {
        model: 'image',
      },
    })).toEqual({
      chat_llm: {
        model: 'chat',
      },
      image_gen: {
        model: 'image',
      },
    })

    expect(configFromApi({
      chat_llm: {
        model: 'chat',
      },
      image_gen: {
        model: 'image',
      },
    })).toEqual({
      llm: {
        model: 'chat',
      },
      imageGen: {
        model: 'image',
      },
    })
  })
})
