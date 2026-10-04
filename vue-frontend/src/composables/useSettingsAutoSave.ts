import { nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { deepClone } from '@/utils/deepClone'
import { sameSettingValue } from '@/utils/providerSettings'

export function useSettingsAutoSave(options: {
  source: () => unknown
  ready: () => boolean
  save: () => Promise<void>
  onError: (error: unknown) => void
}) {
  const isSaving = ref(false)
  const hasUnsavedChanges = ref(false)
  let timer: ReturnType<typeof setTimeout> | null = null
  let savePromise: Promise<boolean> | null = null
  let previous: unknown = null

  function clearTimer(): void {
    if (timer !== null) clearTimeout(timer)
    timer = null
  }

  watch(
    () => options.ready() ? options.source() : null,
    value => {
      if (value === null) {
        previous = null
        clearTimer()
        return
      }
      const changed = previous !== null && !sameSettingValue(value, previous)
      previous = deepClone(value)
      if (!changed) return
      hasUnsavedChanges.value = true
      clearTimer()
      timer = setTimeout(() => void persistChanges(), 450)
    },
    { deep: true, flush: 'sync' },
  )

  async function persistChanges(): Promise<boolean> {
    clearTimer()
    if (savePromise) {
      const saved = await savePromise
      return saved ? persistChanges() : false
    }
    if (!options.ready() || !hasUnsavedChanges.value) return true

    hasUnsavedChanges.value = false
    isSaving.value = true
    savePromise = (async () => {
      try {
        await options.save()
        await nextTick()
        return true
      } catch (error) {
        hasUnsavedChanges.value = true
        options.onError(error)
        return false
      } finally {
        isSaving.value = false
      }
    })()
    let saved: boolean
    try {
      saved = await savePromise
    } finally {
      savePromise = null
    }
    return saved && hasUnsavedChanges.value ? persistChanges() : saved
  }

  onBeforeUnmount(() => void persistChanges())

  return { isSaving, hasUnsavedChanges, persistChanges }
}
