type SnapshotPredicate = () => boolean

function passesSnapshot(predicate?: SnapshotPredicate): boolean {
  return predicate ? predicate() : true
}

export function useLatestRequestGuard() {
  let currentRequestId = 0

  function next(): number {
    currentRequestId += 1
    return currentRequestId
  }

  function invalidate(): void {
    currentRequestId += 1
  }

  function isCurrent(requestId: number, predicate?: SnapshotPredicate): boolean {
    return requestId === currentRequestId && passesSnapshot(predicate)
  }

  return {
    next,
    invalidate,
    isCurrent,
  }
}

export function useKeyedLatestRequestGuard<Key>() {
  const currentRequestIds = new Map<Key, number>()

  function next(key: Key): number {
    const requestId = (currentRequestIds.get(key) || 0) + 1
    currentRequestIds.set(key, requestId)
    return requestId
  }

  function invalidate(key: Key): void {
    currentRequestIds.set(key, (currentRequestIds.get(key) || 0) + 1)
  }

  function isCurrent(key: Key, requestId: number, predicate?: SnapshotPredicate): boolean {
    return currentRequestIds.get(key) === requestId && passesSnapshot(predicate)
  }

  function forget(key: Key): void {
    currentRequestIds.delete(key)
  }

  return {
    next,
    invalidate,
    isCurrent,
    forget,
  }
}
