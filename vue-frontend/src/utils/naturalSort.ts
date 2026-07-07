type NaturalSortPart = [isText: boolean, value: number | string]

export function naturalSortKey(path: string): NaturalSortPart[] {
  const normalizedPath = path.replace(/\\/g, '/')
  const parts: NaturalSortPart[] = []
  const digitPattern = /(\d+)/g
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = digitPattern.exec(normalizedPath)) !== null) {
    if (match.index > lastIndex) {
      const textPart = normalizedPath.slice(lastIndex, match.index).toLowerCase()
      if (textPart) {
        parts.push([true, textPart])
      }
    }

    parts.push([false, Number.parseInt(match[0], 10)])
    lastIndex = digitPattern.lastIndex
  }

  if (lastIndex < normalizedPath.length) {
    const textPart = normalizedPath.slice(lastIndex).toLowerCase()
    if (textPart) {
      parts.push([true, textPart])
    }
  }

  return parts
}

export function naturalSortCompare(a: string, b: string): number {
  const keyA = naturalSortKey(a)
  const keyB = naturalSortKey(b)
  const minLength = Math.min(keyA.length, keyB.length)

  for (let index = 0; index < minLength; index += 1) {
    const [isTextA, valueA] = keyA[index]!
    const [isTextB, valueB] = keyB[index]!

    if (isTextA !== isTextB) {
      return isTextA ? 1 : -1
    }

    if (valueA < valueB) return -1
    if (valueA > valueB) return 1
  }

  return keyA.length - keyB.length
}

export function naturalSort<T>(
  files: T[],
  getPath: (item: T) => string = item => String(item),
): T[] {
  return [...files].sort((a, b) => naturalSortCompare(getPath(a), getPath(b)))
}
