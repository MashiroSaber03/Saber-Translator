import { onBeforeUnmount, ref, toRaw } from 'vue'
import type { StudioAction, StudioState } from './protocol'
import type { DomDetectionResult } from '../types'
import { RequestFailure, saberRequest } from '../api'
export function usePageBridge() {
  const state = ref<StudioState | null>(null)
  let sequence = 0
  const pending = new Map<number, {
    resolve(value: unknown): void
    reject(error: Error): void
    domController?: AbortController
    domStarted?: boolean
  }>()
  function request<T = void>(action: StudioAction, payload?: unknown): Promise<T> {
    const id = ++sequence
    return new Promise<T>((resolve, reject) => {
      pending.set(id, {
        resolve: value => resolve(value as T), reject,
        ...(action === 'discover' && payload === 'dom-agent'
          ? { domController: new AbortController() } : {}),
      })
      window.parent.postMessage(
        { channel: 'saber:command', id, action, payload: toRaw(payload) },
        '*'
      )
    })
  }
  function receive(event: MessageEvent) {
    if (event.source !== window.parent) return
    if (event.data?.channel === 'saber:dom-detection') {
      const item = pending.get(event.data.id)
      // Only the DOM discovery initiated in this frame may invoke the model.
      if (!item?.domController || item.domStarted) return
      item.domStarted = true
      const reply = (response: object) => window.parent.postMessage({
        channel: 'saber:dom-detection-result', id: event.data.id, ...response,
      }, '*')
      // The extension document remains alive with its page. Model deadlines and
      // retries belong to the backend; a worker fetch or another timer can end early.
      void saberRequest<DomDetectionResult>('/dom-detection', {
        method: 'POST', body: JSON.stringify(event.data.payload), signal: item.domController.signal,
      }).then(
        result => reply({ ok: true, result }),
        error => reply({ ok: false, error: {
          code: error instanceof RequestFailure ? error.code : 'dom_agent_failed',
          message: error instanceof Error ? error.message : String(error),
          retryable: error instanceof RequestFailure ? error.retryable : true,
        } }),
      )
    }
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
    for (const item of pending.values()) item.domController?.abort()
    pending.clear()
  })
  function notify(action: StudioAction, payload?: unknown) {
    window.parent.postMessage({ channel: 'saber:command', action, payload }, '*')
  }
  return { state, request, notify }
}
