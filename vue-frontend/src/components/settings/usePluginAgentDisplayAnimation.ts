import { ref } from 'vue'

interface DisplayAnimationOptions {
  animate: boolean
  onTick: () => void | Promise<void>
}

function nextDisplaySlice(current: string, target: string): string {
  const step = Math.max(1, Math.ceil((target.length - current.length) / 6))
  return target.slice(0, current.length + step)
}

export function usePluginAgentDisplayAnimation(options: DisplayAnimationOptions) {
  const assistantMessageDisplayContent = ref<Record<string, string>>({})
  const assistantMessageDisplayTargets = ref<Record<string, string>>({})
  const assistantDisplayContent = ref<Record<string, string>>({})
  const assistantDisplayTargets = ref<Record<string, string>>({})
  const messageTimers = new Map<string, ReturnType<typeof setInterval>>()
  const streamTimers = new Map<string, ReturnType<typeof setInterval>>()

  function clear(): void {
    for (const timer of messageTimers.values()) clearInterval(timer)
    for (const timer of streamTimers.values()) clearInterval(timer)
    messageTimers.clear()
    streamTimers.clear()
    assistantMessageDisplayContent.value = {}
    assistantMessageDisplayTargets.value = {}
    assistantDisplayContent.value = {}
    assistantDisplayTargets.value = {}
  }

  function setMessageTarget(messageId: string, targetContent: string, animate: boolean): void {
    assistantMessageDisplayTargets.value = {
      ...assistantMessageDisplayTargets.value,
      [messageId]: targetContent,
    }

    if (!animate || !options.animate) {
      assistantMessageDisplayContent.value = {
        ...assistantMessageDisplayContent.value,
        [messageId]: targetContent,
      }
      const existingTimer = messageTimers.get(messageId)
      if (existingTimer) clearInterval(existingTimer)
      messageTimers.delete(messageId)
      return
    }

    if (!Object.prototype.hasOwnProperty.call(assistantMessageDisplayContent.value, messageId)) {
      assistantMessageDisplayContent.value = {
        ...assistantMessageDisplayContent.value,
        [messageId]: '',
      }
    }
    if (messageTimers.has(messageId)) return

    const tick = () => {
      const current = assistantMessageDisplayContent.value[messageId] || ''
      const target = assistantMessageDisplayTargets.value[messageId] || ''
      if (current === target) {
        const timer = messageTimers.get(messageId)
        if (timer) clearInterval(timer)
        messageTimers.delete(messageId)
        return
      }
      assistantMessageDisplayContent.value = {
        ...assistantMessageDisplayContent.value,
        [messageId]: nextDisplaySlice(current, target),
      }
      void options.onTick()
    }

    tick()
    messageTimers.set(messageId, setInterval(tick, 16))
  }

  function setStreamTarget(streamId: string, targetContent: string): void {
    assistantDisplayTargets.value = {
      ...assistantDisplayTargets.value,
      [streamId]: targetContent,
    }
    if (!options.animate) {
      assistantDisplayContent.value = {
        ...assistantDisplayContent.value,
        [streamId]: targetContent,
      }
      return
    }
    if (!Object.prototype.hasOwnProperty.call(assistantDisplayContent.value, streamId)) {
      assistantDisplayContent.value = {
        ...assistantDisplayContent.value,
        [streamId]: '',
      }
    }
    if (streamTimers.has(streamId)) return

    const tick = () => {
      const current = assistantDisplayContent.value[streamId] || ''
      const target = assistantDisplayTargets.value[streamId] || ''
      if (current === target) {
        const timer = streamTimers.get(streamId)
        if (timer) clearInterval(timer)
        streamTimers.delete(streamId)
        return
      }
      assistantDisplayContent.value = {
        ...assistantDisplayContent.value,
        [streamId]: nextDisplaySlice(current, target),
      }
      void options.onTick()
    }

    tick()
    streamTimers.set(streamId, setInterval(tick, 16))
  }

  return {
    assistantMessageDisplayContent,
    assistantMessageDisplayTargets,
    assistantDisplayContent,
    assistantDisplayTargets,
    clear,
    setMessageTarget,
    setStreamTarget,
  }
}
