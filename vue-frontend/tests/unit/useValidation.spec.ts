import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
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

  it('exposes settings highlight state without querying the header DOM', async () => {
    vi.useFakeTimers()
    const { useValidation } = await import('@/composables/useValidation')
    const validation = useValidation()

    try {
      validation.highlightSettingsButton()

      expect(validation.isSettingsButtonHighlighted.value).toBe(true)

      vi.advanceTimersByTime(3000)

      expect(validation.isSettingsButtonHighlighted.value).toBe(false)
    } finally {
      vi.useRealTimers()
    }
  })

  it('keeps validation source comments focused on current behavior contracts', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/composables/useValidation.ts'), 'utf8')
    const firstTimeGuideSource = readFileSync(
      resolve(process.cwd(), 'src/components/translate/firstTimeGuideState.ts'),
      'utf8'
    )

    for (const staleNarration of [
      '翻译配置验证组合式函数',
      '// ===' + '=========================================================',
      '类型定义',
      '工具函数',
      'UI 交互函数',
      '返回',
      '@param',
      '@returns',
      '类型已在上方定义并导出',
    ]) {
      expect(source).not.toContain(staleNarration)
    }

    expect(source).toContain('@/components/translate/firstTimeGuideState')
    expect(source).not.toContain("const DISMISS_SETUP_REMINDER_KEY = 'saber_translator_dismiss_setup_reminder'")
    expect(source).not.toContain('localStorage.getItem(DISMISS_SETUP_REMINDER_KEY)')
    expect(source).not.toContain('localStorage.setItem(DISMISS_SETUP_REMINDER_KEY')
    expect(source).not.toContain('localStorage.removeItem(DISMISS_SETUP_REMINDER_KEY')
    expect(firstTimeGuideSource).toContain('Storage can be unavailable in restricted browser contexts')
  })

  it('keeps validation property provider coverage sourced from the manifest', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/property/validation.property.ts'), 'utf8')

    expect(source).toContain("import { AI_PROVIDER_MANIFEST } from '@/config/aiProviders'")
    expect(source).toContain("capabilities.includes('translation')")
    expect(source).toContain("capabilities.includes('hqTranslation')")

    for (const staleProviderList of [
      'providersRequiring' + 'ApiKey',
      'local' + 'Providers',
      'all' + 'Providers',
      'hq' + 'Providers',
      '生成器' + '定义',
      '// ===' + '=========================================================',
    ]) {
      expect(source).not.toContain(staleProviderList)
    }
  })
})
