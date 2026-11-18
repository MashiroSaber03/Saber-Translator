// @vitest-environment jsdom
import { createApp, h, nextTick, reactive, type App } from 'vue'
import { afterEach, expect, it, vi } from 'vitest'
import SettingsView from './SettingsView.vue'
import { installedModelOptions, DETECTOR_MODEL_KEYS, OCR_MODEL_KEYS, INPAINT_MODEL_KEYS } from '../../../vue-frontend/src/utils/localModelAvailability'
import { allOcrEngineOptions } from '../../../vue-frontend/src/config/ocrChoices'
import type { PluginSettingsApi } from '../../../vue-frontend/src/types/browserExtensionSettings'

vi.mock('./ConnectionView.vue', () => ({ default: { render: () => null } }))
vi.mock('./TranslatorSettingsView.vue', () => ({ default: {
  props: ['installedModels'], setup: (props: { installedModels: string[] }) =>
    () => h('output', { 'data-models': props.installedModels.join(',') }),
} }))
vi.mock('./StyleSettingsView.vue', () => ({ default: {
  props: ['installedModels'], setup: (props: { installedModels: string[] }) =>
    () => h('output', { 'data-models': props.installedModels.join(',') }),
} }))
let app: App | undefined
afterEach(() => { app?.unmount(); vi.useRealTimers() })

it('refreshes both extension settings from one inventory and stops when closed', async () => {
  vi.useFakeTimers()
  document.body.innerHTML = '<div id="app"></div>'
  let installedModels: string[] = []
  const api = vi.fn(async () => ({ installedModels }))
  const props = reactive({ api: api as PluginSettingsApi, active: true, state: null, request: async () => {} })
  app = createApp({ render: () => h(SettingsView, props) })
  app.mount('#app')
  expect(api).not.toHaveBeenCalled()
  async function section(value: string) {
    ;(document.querySelector('[aria-label="配置分类"]') as HTMLButtonElement).click()
    await nextTick()
    ;(document.querySelector(`[data-value="${value}"]`) as HTMLButtonElement).click()
    await nextTick()
    await vi.advanceTimersByTimeAsync(0)
  }
  await section('ocr')
  expect(document.querySelector('output')?.getAttribute('data-models')).toBe('')
  installedModels = ['manga_ocr']
  await vi.advanceTimersByTimeAsync(5000)
  expect(document.querySelector('output')?.getAttribute('data-models')).toBe('manga_ocr')
  await section('style')
  expect([...document.querySelectorAll('output')].every(node => node.getAttribute('data-models') === 'manga_ocr')).toBe(true)
  expect(api).toHaveBeenCalledTimes(2)
  props.active = false
  await nextTick()
  await vi.advanceTimersByTimeAsync(10000)
  expect(api).toHaveBeenCalledTimes(2)
  installedModels = ['paddle_ocr']
  props.active = true
  await nextTick()
  await vi.advanceTimersByTimeAsync(0)
  expect([...document.querySelectorAll('output')].every(node => node.getAttribute('data-models') === 'paddle_ocr')).toBe(true)
})

it('uses the same dependencies as the translator and retains remote OCR and pure fill', () => {
  expect(installedModelOptions(allOcrEngineOptions, OCR_MODEL_KEYS, []).map(option => option.value)).toEqual(['baidu_ocr', 'ai_vision'])
  const detectors = [{ value: 'yolo', label: 'YOLO' }, { value: 'default', label: 'Default' }]
  expect(installedModelOptions(detectors, DETECTOR_MODEL_KEYS, ['detector_yolo'])).toEqual([])
  expect(installedModelOptions(detectors, DETECTOR_MODEL_KEYS, ['detector_yolo', 'detector_default'])).toEqual(detectors)
  const fills = [{ value: 'solid', label: '纯色' }, { value: 'lama_mpe', label: 'LAMA' }]
  expect(installedModelOptions(fills, INPAINT_MODEL_KEYS, [])).toEqual([fills[0]])
})
