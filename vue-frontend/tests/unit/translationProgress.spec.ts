import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import TranslationProgress from '@/components/translate/TranslationProgress.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'

vi.mock('@/composables/useTranslationPipeline', () => ({
  useTranslation: () => ({
    progress: {
      value: {
        isInProgress: false,
        current: 0,
        total: 0,
        completed: 0,
        failed: 0,
      },
    },
  }),
}))

describe('TranslationProgress', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('exposes backend job progress to assistive technology', () => {
    const wrapper = mount(TranslationProgress, {
      props: {
        progress: {
          isInProgress: true,
          current: 2,
          total: 5,
          completed: 1,
          failed: 1,
          label: '后端正在处理',
        },
      },
    })

    const progressbar = wrapper.get('[role="progressbar"]')
    expect(progressbar.attributes('aria-label')).toBe('翻译进度')
    expect(progressbar.attributes('aria-valuenow')).toBe('40')
    expect(wrapper.text()).toContain('后端正在处理')
    expect(wrapper.text()).toContain('1 张失败')
  })

  it('clamps backend progress percentages to the progressbar range', () => {
    const wrapper = mount(TranslationProgress, {
      props: {
        progress: {
          isInProgress: true,
          current: 8,
          total: 5,
          completed: 8,
          failed: 0,
          percentage: 140,
        },
      },
    })

    expect(wrapper.getComponent(UiProgressBar).props('value')).toBe(100)
  })

  it('does not import or render the removed browser pool pipeline', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/TranslationProgress.vue'),
      'utf8',
    )

    expect(source).not.toContain('useParallelTranslation')
    expect(source).not.toContain('preSave')
    expect(source).not.toContain('translation-progress__pool')
    expect(source).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
  })

  it('uses component-container sizing and numeric font weights', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/TranslationProgress.vue'),
      'utf8',
    )
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(source).toContain('container: translation-progress / inline-size')
    expect(source).toContain('@container translation-progress')
    expect(styleBlock).not.toMatch(/font-weight:\s*(bold|normal)\b/)
  })
})
