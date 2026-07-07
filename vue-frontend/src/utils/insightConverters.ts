const FIELD_MAPPINGS = {
  toApi: {
    llm: 'chat_llm',
  },
  toStore: {
    chatLlm: 'llm',
  },
}

function convertObjectKeys(value: unknown, convertKey: (key: string) => string): unknown {
  if (Array.isArray(value)) {
    return value.map(item => convertObjectKeys(item, convertKey))
  }

  if (value === null || typeof value !== 'object') {
    return value
  }

  const result: Record<string, unknown> = {}

  for (const [key, item] of Object.entries(value)) {
    result[convertKey(key)] = convertObjectKeys(item, convertKey)
  }

  return result
}

function camelToSnake(value: string): string {
  return value.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`)
}

function snakeToCamel(value: string): string {
  return value.replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase())
}

export function toSnakeCase(value: unknown): unknown {
  return convertObjectKeys(value, camelToSnake)
}

export function toCamelCase(value: unknown): unknown {
  return convertObjectKeys(value, snakeToCamel)
}

export function applyFieldMappings(
  source: Record<string, unknown>,
  direction: keyof typeof FIELD_MAPPINGS,
): Record<string, unknown> {
  const result = { ...source }

  for (const [from, to] of Object.entries(FIELD_MAPPINGS[direction])) {
    if (from in result) {
      result[to] = result[from]
      delete result[from]
    }
  }

  return result
}

export function configToApi<T extends Record<string, unknown>>(config: T): Record<string, unknown> {
  return applyFieldMappings(toSnakeCase(config) as Record<string, unknown>, 'toApi')
}

export function configFromApi<T extends Record<string, unknown>>(apiConfig: T): Record<string, unknown> {
  return applyFieldMappings(toCamelCase(apiConfig) as Record<string, unknown>, 'toStore')
}
