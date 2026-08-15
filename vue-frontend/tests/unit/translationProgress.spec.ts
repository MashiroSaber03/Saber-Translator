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
        executionMode: 'sequential',
        pools: [],
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
          executionMode: 'sequential',
          pools: [],
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
          executionMode: 'sequential',
          pools: [],
        },
      },
    })

    expect(wrapper.getComponent(UiProgressBar).props('value')).toBe(100)
  })

  it('renders only backend-projected pool facts for parallel jobs', () => {
    const wrapper = mount(TranslationProgress, {
      props: {
        progress: {
          isInProgress: true,
          current: 1,
          total: 3,
          completed: 1,
          failed: 0,
          executionMode: 'parallel',
          pools: [{
            kind: 'ocr',
            total: 3,
            completed: 1,
            failed: 0,
            skipped: 0,
            cancelled: 0,
            waiting: 1,
            processing: 1,
            lockWaiting: true,
            current: [],
          }],
        },
      },
    })

    expect(wrapper.text()).toContain('文字识别')
    expect(wrapper.text()).toContain('完成 1 / 3')
    expect(wrapper.text()).toContain('处理中 1')
    expect(wrapper.text()).toContain('等待深度学习锁')
  })

  it('keeps non-zero progress visible for large jobs', () => {
    const wrapper = mount(TranslationProgress, {
      props: {
        progress: {
          isInProgress: true,
          current: 8,
          total: 2702,
          completed: 8,
          failed: 0,
          executionMode: 'parallel',
          pools: [{
            kind: 'translate',
            total: 2702,
            completed: 8,
            failed: 0,
            skipped: 0,
            cancelled: 0,
            waiting: 2693,
            processing: 1,
            lockWaiting: false,
            current: [],
          }],
        },
      },
    })

    const bars = wrapper.findAllComponents(UiProgressBar)
    expect(bars[0]?.props('value')).toBeCloseTo(8 / 2702 * 100)
    expect(bars[1]?.props('value')).toBeCloseTo(8 / 2702 * 100)
    expect(bars[0]?.props('value')).toBeGreaterThan(0)
  })

  it('stops progress animation while a backend task is paused', () => {
    const wrapper = mount(TranslationProgress, {
      props: {
        progress: {
          isInProgress: true,
          current: 2,
          total: 5,
          completed: 2,
          failed: 0,
          executionMode: 'parallel',
          status: 'paused',
          pools: [{
            kind: 'ocr',
            total: 5,
            completed: 2,
            failed: 0,
            skipped: 0,
            cancelled: 0,
            waiting: 2,
            processing: 1,
            lockWaiting: false,
            current: [],
          }],
        },
      },
    })

    const bars = wrapper.findAllComponents(UiProgressBar)
    expect(bars[0]?.props('animated')).toBe(false)
    expect(bars[1]?.props('animated')).toBe(false)
  })

  it('does not import the removed browser-owned pool pipeline', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/TranslationProgress.vue'),
      'utf8',
    )

    expect(source).not.toContain('useParallelTranslation')
    expect(source).not.toContain('preSave')
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
