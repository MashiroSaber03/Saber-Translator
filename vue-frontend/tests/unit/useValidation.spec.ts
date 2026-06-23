import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import type { useValidation as useValidationFn } from '@/composables/useValidation'

describe('useValidation', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
  })

  it('setup reminder preference reads and writes stay quiet during normal user actions', async () => {
    localStorage.setItem('saber_translator_dismiss_setup_reminder', 'true')

    const { useValidation } = await import('@/composables/useValidation')
    const validation = useValidation()
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)

    try {
      validation.checkAndShowSetupReminder()
      expect(validation.showSetupReminder.value).toBe(false)

      validation.closeSetupReminder(true)
      expect(localStorage.getItem('saber_translator_dismiss_setup_reminder')).toBe('true')
      expect(validation.showSetupReminder.value).toBe(false)

      validation.resetSetupReminderDismiss()
      expect(localStorage.getItem('saber_translator_dismiss_setup_reminder')).toBeNull()
      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }
  })

  it('cancels delayed setup reminder when the owner unmounts', async () => {
    vi.useFakeTimers()
    const { useValidation } = await import('@/composables/useValidation')
    let validation: ReturnType<typeof useValidationFn> | null = null

    try {
      const Host = defineComponent({
        setup() {
          validation = useValidation()
          validation.initValidation()
          return () => h('div')
        },
      })

      const wrapper = mount(Host)
      wrapper.unmount()
      vi.advanceTimersByTime(500)

      expect(validation?.showSetupReminder.value).toBe(false)
    } finally {
      vi.useRealTimers()
    }
  })
})
