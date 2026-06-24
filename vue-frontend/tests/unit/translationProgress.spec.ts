import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import TranslationProgress from '@/components/translate/TranslationProgress.vue'

vi.mock('@/composables/useTranslationPipeline', () => ({
  useTranslation: () => ({
    progress: {
      value: {
        isInProgress: false,
        current: 0,
        total: 0,
        failed: 0,
      },
    },
  }),
}))

vi.mock('@/composables/translation/parallel', () => ({
  useParallelTranslation: () => ({
    progress: { value: null },
    isRunning: { value: false },
  }),
}))

describe('TranslationProgress', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('exposes normal progress values to assistive technology', () => {
    const wrapper = mount(TranslationProgress, {
      props: {
        progress: {
          isInProgress: true,
          current: 2,
          total: 5,
          failed: 1,
        },
      },
    })

    const progressbar = wrapper.get('[role="progressbar"]')
    expect(progressbar.attributes('aria-label')).toBe('翻译进度')
    expect(progressbar.attributes('aria-valuemin')).toBe('0')
    expect(progressbar.attributes('aria-valuemax')).toBe('100')
    expect(progressbar.attributes('aria-valuenow')).toBe('40')
  })

  it('clamps custom progress percentages to the progressbar range', () => {
    const wrapper = mount(TranslationProgress, {
      props: {
        progress: {
          isInProgress: true,
          current: 8,
          total: 5,
          failed: 0,
          percentage: 140,
        },
      },
    })

    const progressbar = wrapper.get('[role="progressbar"]')
    expect(progressbar.attributes('aria-valuenow')).toBe('100')
    expect(wrapper.get('.progress').attributes('style')).toContain('width: 100%;')
  })
})
