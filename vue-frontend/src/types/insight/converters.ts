/**
 * Manga Insight 类型转换器
 *
 * 提供 camelCase <-> snake_case 自动转换，消除手动字段映射代码。
 */

/**
 * 将 camelCase 对象转换为 snake_case
 */
export function toSnakeCase<T extends Record<string, unknown>>(obj: T): Record<string, unknown> {
  if (obj === null || obj === undefined) {
    return obj
  }

  if (Array.isArray(obj)) {
    return obj.map((item) =>
      typeof item === 'object' && item !== null ? toSnakeCase(item as Record<string, unknown>) : item
    ) as unknown as Record<string, unknown>
  }

  if (typeof obj !== 'object') {
    return obj as unknown as Record<string, unknown>
  }

  const result: Record<string, unknown> = {}

  for (const [key, value] of Object.entries(obj)) {
    const snakeKey = camelToSnake(key)

    if (value !== null && typeof value === 'object') {
      if (Array.isArray(value)) {
        result[snakeKey] = value.map((item) =>
          typeof item === 'object' && item !== null
            ? toSnakeCase(item as Record<string, unknown>)
            : item
        )
      } else {
        result[snakeKey] = toSnakeCase(value as Record<string, unknown>)
      }
    } else {
      result[snakeKey] = value
    }
  }

  return result
}

/**
 * 将 snake_case 对象转换为 camelCase
 */
export function toCamelCase<T extends Record<string, unknown>>(obj: T): Record<string, unknown> {
  if (obj === null || obj === undefined) {
    return obj
  }

  if (Array.isArray(obj)) {
    return obj.map((item) =>
      typeof item === 'object' && item !== null ? toCamelCase(item as Record<string, unknown>) : item
    ) as unknown as Record<string, unknown>
  }

  if (typeof obj !== 'object') {
    return obj as unknown as Record<string, unknown>
  }

  const result: Record<string, unknown> = {}

  for (const [key, value] of Object.entries(obj)) {
    const camelKey = snakeToCamel(key)

    if (value !== null && typeof value === 'object') {
      if (Array.isArray(value)) {
        result[camelKey] = value.map((item) =>
          typeof item === 'object' && item !== null
            ? toCamelCase(item as Record<string, unknown>)
            : item
        )
      } else {
        result[camelKey] = toCamelCase(value as Record<string, unknown>)
      }
    } else {
      result[camelKey] = value
    }
  }

  return result
}

/**
 * camelCase 转 snake_case
 */
function camelToSnake(str: string): string {
  return str.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`)
}

/**
 * snake_case 转 camelCase
 */
function snakeToCamel(str: string): string {
  return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase())
}

/**
 * 特殊字段映射（用于非标准转换）
 */
const FIELD_MAPPINGS: Record<string, string> = {
  // API 到 Store 的映射
  chat_llm: 'llm',
  image_gen: 'imageGen',
  // Store 到 API 的映射
  llm: 'chat_llm'
}

/**
 * 应用特殊字段映射
 */
export function applyFieldMappings(
  obj: Record<string, unknown>,
  direction: 'toApi' | 'toStore'
): Record<string, unknown> {
  const result = { ...obj }

  for (const [from, to] of Object.entries(FIELD_MAPPINGS)) {
    if (direction === 'toApi' && from in result && !from.includes('_')) {
      // Store -> API
      result[to] = result[from]
      delete result[from]
    } else if (direction === 'toStore' && from in result && from.includes('_')) {
      // API -> Store
      result[to] = result[from]
      delete result[from]
    }
  }

  return result
}

/**
 * 配置对象：Store -> API 格式
 */
export function configToApi<T extends Record<string, unknown>>(config: T): Record<string, unknown> {
  const snakeCase = toSnakeCase(config)
  return applyFieldMappings(snakeCase, 'toApi')
}

/**
 * 配置对象：API -> Store 格式
 */
export function configFromApi<T extends Record<string, unknown>>(apiConfig: T): Record<string, unknown> {
  const camelCase = toCamelCase(apiConfig)
  return applyFieldMappings(camelCase, 'toStore')
}
