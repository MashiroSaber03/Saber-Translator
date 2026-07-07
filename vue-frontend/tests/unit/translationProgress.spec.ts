import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import TranslationProgress from '@/components/translate/TranslationProgress.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import { useSettingsStore } from '@/stores/settings'

const parallelTranslationMock = vi.hoisted(() => ({
  progress: { value: null as unknown },
  isRunning: { value: false },
}))

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
  useParallelTranslation: () => parallelTranslationMock,
}))

describe('TranslationProgress', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    parallelTranslationMock.progress.value = null
    parallelTranslationMock.isRunning.value = false
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
    expect(wrapper.getComponent(UiProgressBar).props('value')).toBe(100)
    expect(wrapper.find('.progress').exists()).toBe(false)
  })

  it('maps parallel progress owner colors through semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/TranslationProgress.vue'),
      'utf8'
    )

    expect(source).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
  })

  it('does not warehouse one-off semantic aliases in the progress owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/TranslationProgress.vue'),
      'utf8'
    )

    for (const staleAlias of [
      '--translation-progress-completed-fill-start',
      '--translation-progress-completed-fill-end',
      '--translation-progress-processing-fill-start',
      '--translation-progress-processing-fill-end',
      '--translation-progress-presave-panel-start',
      '--translation-progress-presave-panel-end',
      '--translation-progress-label-text',
      '--translation-progress-failed-text',
    ]) {
      expect(source).not.toContain(staleAlias)
    }
  })

  it('keeps the progress owner free of unused DOM id hooks', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/TranslationProgress.vue'),
      'utf8'
    )

    expect(source).not.toContain('id="translationProgressBar"')
  })

  it('keeps multi-segment pool progress local while single parallel bars use UiProgressBar', () => {
    const settingsStore = useSettingsStore()
    settingsStore.settings.parallel.enabled = true
    parallelTranslationMock.isRunning.value = true
    parallelTranslationMock.progress.value = {
      totalCompleted: 2,
      totalFailed: 1,
      totalPages: 4,
      preSave: {
        isRunning: true,
        current: 1,
        total: 4,
      },
      pools: [
        {
          name: 'OCR',
          icon: 'scan-text',
          processing: true,
          completed: 1,
          waiting: 1,
          isWaitingLock: false,
        },
      ],
      save: {
        completed: 2,
        total: 4,
      },
    }

    const wrapper = mount(TranslationProgress, {
      props: {
        progress: {
          isInProgress: false,
          current: 0,
          total: 0,
          failed: 0,
        },
      },
    })

    const productProgressBars = wrapper.findAllComponents(UiProgressBar)
    expect(productProgressBars).toHaveLength(3)
    expect(productProgressBars.map((bar) => bar.props('label'))).toEqual([
      '预保存原始图片进度',
      '保存进度',
      '翻译总进度',
    ])
    expect(wrapper.find('.translation-progress__pool-bar').exists()).toBe(true)
  })

  it('uses component-container sizing for parallel pool rows', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/TranslationProgress.vue'),
      'utf8'
    )

    expect(source).toContain('container: translation-progress / inline-size')
    expect(source).toContain('@container translation-progress')
    expect(source).not.toContain('grid-template-columns: 80px 1fr 150px')
    expect(source).not.toContain('grid-template-columns: 70px 1fr 130px')
    expect(source).not.toContain('grid-template-columns: 60px 1fr 120px')
  })

  it('uses numeric font weights instead of keyword weight aliases', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/TranslationProgress.vue'),
      'utf8'
    )
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/font-weight:\s*(bold|normal)\b/)
  })

  it('keeps progress structure hooks owner-prefixed', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/TranslationProgress.vue'),
      'utf8'
    )

    for (const currentHook of [
      'translation-progress__parallel-header',
      'translation-progress__header-title',
      'translation-progress__presave-section',
      'translation-progress__presave-label',
      'translation-progress__pools-list',
      'translation-progress__pool-row',
      'translation-progress__pool-bar',
      'translation-progress__completed-segment',
      'translation-progress__processing-segment',
      'translation-progress__pool-stats',
      'translation-progress__completed-count',
      'translation-progress__total-count',
      'translation-progress__waiting-badge',
      'translation-progress__lock-indicator',
      'translation-progress__divider',
      'translation-progress__overall-section',
      'translation-progress__overall-label',
      'translation-progress__failed-text',
      'translation-progress__normal-label',
      'translation-progress__failed-count',
    ]) {
      expect(source).toContain(currentHook)
    }

    for (const oldHook of [
      'class="parallel-header"',
      'class="header-title"',
      'class="presave-section"',
      'class="presave-label"',
      'class="pools-list"',
      'class="pool-row"',
      'class="pool-progress-bar"',
      'class="progress-completed"',
      'class="progress-processing"',
      'class="pool-stats"',
      'class="completed-count"',
      'class="total-count"',
      'class="waiting-badge"',
      'class="lock-indicator"',
      'class="divider"',
      'class="overall-section"',
      'class="overall-label"',
      'class="failed-text"',
      'class="progress-bar-label"',
      'class="failed-count"',
    ]) {
      expect(source).not.toContain(oldHook)
    }
  })
})
