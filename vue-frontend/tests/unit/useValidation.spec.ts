import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useSettingsStore } from '@/stores/settings'

describe('useValidation', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
  })

  it('exposes settings highlight state without querying the header DOM', async () => {
    vi.useFakeTimers()
    const { useValidation } = await import('@/composables/useValidation')
    const validation = useValidation()

    try {
      useSettingsStore().updateTranslationService({
        provider: 'deepseek',
        apiKey: '',
        modelName: '',
      })
      expect(validation.validateBeforeTranslation()).toBe(false)

      expect(validation.isSettingsButtonHighlighted.value).toBe(true)

      vi.advanceTimersByTime(3000)

      expect(validation.isSettingsButtonHighlighted.value).toBe(false)
    } finally {
      vi.useRealTimers()
    }
  })

  it('keeps validation source comments focused on current behavior contracts', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/composables/useValidation.ts'), 'utf8')
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

    expect(source).not.toContain('@/components/translate/firstTimeGuideState')
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
