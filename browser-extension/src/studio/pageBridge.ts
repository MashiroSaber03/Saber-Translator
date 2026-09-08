import { onBeforeUnmount, ref, toRaw } from 'vue'
import type { StudioAction, StudioState } from './protocol'
export function usePageBridge() {
  const state = ref<StudioState | null>(null)
  let sequence = 0
  const pending = new Map<number, { resolve(value: unknown): void; reject(error: Error): void }>()
  function request<T = void>(action: StudioAction, payload?: unknown): Promise<T> {
    const id = ++sequence
    return new Promise<T>((resolve, reject) => {
      pending.set(id, { resolve: value => resolve(value as T), reject })
      window.parent.postMessage(
        { channel: 'saber:command', id, action, payload: toRaw(payload) },
        '*'
      )
    })
  }
  function receive(event: MessageEvent) {
    if (event.source !== window.parent) return
    if (event.data?.channel === 'saber:state') state.value = event.data.state
    if (event.data?.channel === 'saber:response') {
      const item = pending.get(event.data.id)
      if (!item) return
      pending.delete(event.data.id)
      if (event.data.ok) item.resolve(event.data.result)
      else item.reject(new Error(event.data.error))
    }
  }
  window.addEventListener('message', receive)
  if (window.parent !== window) notify('ready')
  onBeforeUnmount(() => {
    window.removeEventListener('message', receive)
    pending.clear()
  })
  function notify(action: StudioAction, payload?: unknown) {
    window.parent.postMessage({ channel: 'saber:command', action, payload }, '*')
  }
  return { state, request, notify }
}
