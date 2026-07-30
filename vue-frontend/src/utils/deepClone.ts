export function deepClone<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map(item => deepClone(item)) as T
  }
  if (value instanceof Date) {
    return new Date(value.getTime()) as T
  }
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, child]) => [key, deepClone(child)])
    ) as T
  }
  return value
}
