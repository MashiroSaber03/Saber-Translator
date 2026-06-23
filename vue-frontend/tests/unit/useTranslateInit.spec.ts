import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent } from 'vue'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useTranslateInit } from '@/composables/useTranslateInit'
import { useImageStore } from '@/stores/imageStore'
import { useSessionStore } from '@/stores/sessionStore'
import { useSettingsStore } from '@/stores/settings'
import { getFontList, getPrompts, getTextboxPrompts } from '@/api/config'
import { cleanupGpu } from '@/api/system'
import { reloadTextStyleDefaultsFromBackend } from '@/defaults/textStyleDefaults'

const routeState = vi.hoisted(() => ({
  query: {} as Record<string, string | undefined>,
}))

vi.mock('vue-router', () => ({
  useRoute: () => routeState,
}))

vi.mock('@/api/config', () => ({
  getFontList: vi.fn(),
  getPrompts: vi.fn(),
  getTextboxPrompts: vi.fn(),
}))

vi.mock('@/api/system', () => ({
  cleanupGpu: vi.fn(),
}))

vi.mock('@/defaults/textStyleDefaults', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/defaults/textStyleDefaults')>()
  return {
    ...actual,
    reloadTextStyleDefaultsFromBackend: vi.fn(),
  }
})

vi.mock('@/utils/toast', () => ({
  showToast: vi.fn(),
}))

describe('useTranslateInit', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
    routeState.query = {}

    vi.mocked(reloadTextStyleDefaultsFromBackend).mockResolvedValue(undefined)
    vi.mocked(getFontList).mockResolvedValue({
      fonts: [
        {
          file_name: 'default.ttf',
          display_name: 'Default',
          path: '/fonts/default.ttf',
          is_default: true,
        },
      ],
    })
    vi.mocked(getPrompts).mockResolvedValue({
      prompt_names: ['Default'],
      default_prompt_content: '',
    })
    vi.mocked(getTextboxPrompts).mockResolvedValue({
      prompt_names: ['Default textbox'],
      default_prompt_content: '',
    })
    vi.mocked(cleanupGpu).mockResolvedValue({
      success: true,
      unloaded_models: ['ocr'],
      memory_allocated_mb: 0,
      memory_reserved_mb: 0,
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    window._isChangingFromSwitchImage = false
  })

  it('does not write routine console logs during successful initialization and image switching', async () => {
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {})
    const translateInit = useTranslateInit()
    const settingsStore = useSettingsStore()
    const sessionStore = useSessionStore()
    const imageStore = useImageStore()

    vi.spyOn(settingsStore, 'initSettings').mockImplementation(() => {})
    vi.spyOn(settingsStore, 'loadFromBackend').mockResolvedValue(true)
    vi.spyOn(sessionStore, 'clearContext').mockImplementation(() => {})

    await translateInit.initializeApp()
    imageStore.addImage('first.png', 'data:image/png;base64,first')
    imageStore.addImage('second.png', 'data:image/png;base64,second')
    translateInit.switchImage(1)
    await vi.advanceTimersByTimeAsync(100)

    expect(window._isChangingFromSwitchImage).toBe(false)
    expect(consoleLog).not.toHaveBeenCalled()
  })

  it('clears the image-switching flag when the owner unmounts', () => {
    const imageStore = useImageStore()
    imageStore.addImage('first.png', 'data:image/png;base64,first')
    imageStore.addImage('second.png', 'data:image/png;base64,second')

    let translateInit!: ReturnType<typeof useTranslateInit>
    const Harness = defineComponent({
      setup() {
        translateInit = useTranslateInit()
        return () => null
      },
    })

    const wrapper = mount(Harness)
    translateInit.switchImage(1)

    expect(window._isChangingFromSwitchImage).toBe(true)

    wrapper.unmount()
    expect(window._isChangingFromSwitchImage).toBe(false)
  })
})
