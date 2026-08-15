import { onBeforeUnmount, toValue, watch, type MaybeRefOrGetter } from 'vue'

let lockCount = 0
let previousBodyOverflow: string | null = null

function acquireBodyScrollLock(): void {
  if (lockCount === 0) {
    previousBodyOverflow = document.body.style.overflow
  }
  lockCount += 1
  document.body.style.overflow = 'hidden'
}

function releaseBodyScrollLock(): void {
  if (lockCount === 0) return
  lockCount -= 1
  if (lockCount === 0) {
    document.body.style.overflow = previousBodyOverflow ?? ''
    previousBodyOverflow = null
  }
}

export function useBodyScrollLock(locked: MaybeRefOrGetter<boolean>): void {
  let ownsLock = false

  const sync = (shouldLock: boolean) => {
    if (shouldLock === ownsLock) return
    if (shouldLock) acquireBodyScrollLock()
    else releaseBodyScrollLock()
    ownsLock = shouldLock
  }

  watch(
    () => Boolean(toValue(locked)),
    sync,
    { immediate: true },
  )

  onBeforeUnmount(() => sync(false))
}
