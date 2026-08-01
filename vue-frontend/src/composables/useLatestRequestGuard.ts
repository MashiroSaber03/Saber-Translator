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
