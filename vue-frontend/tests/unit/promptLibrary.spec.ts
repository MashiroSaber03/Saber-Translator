import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, reactive } from 'vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const {
  updateTranslationServiceMock,
  updateAiVisionOcrMock,
  toastInfoMock,
  listV2PromptsMock,
  deletePromptMock,
  confirmProductActionMock,
} = vi.hoisted(() => ({
  updateTranslationServiceMock: vi.fn(),
  updateAiVisionOcrMock: vi.fn(),
  toastInfoMock: vi.fn(),
  listV2PromptsMock: vi.fn(),
  deletePromptMock: vi.fn(),
  confirmProductActionMock: vi.fn(),
}))

const settingsState = reactive({
  translation: {
    openaiOptions: {
      request: {
        forceJsonOutput: false,
      },
    },
  },
  aiVisionOcr: {
    openaiOptions: {
      request: {
        forceJsonOutput: false,
      },
    },
    promptMode: 'paddleocr_vl' as 'normal' | 'json' | 'paddleocr_vl',
  },
})

vi.mock('@/stores/settings', () => ({
  useSettingsStore: () => ({
    settings: settingsState,
    updateTranslationService: updateTranslationServiceMock,
    updateAiVisionOcr: updateAiVisionOcrMock,
  }),
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: toastInfoMock,
  }),
}))

vi.mock('@/api/v2/settings', () => ({
  listV2Prompts: listV2PromptsMock,
  createV2Prompt: vi.fn(),
  updateV2Prompt: vi.fn(),
  deleteV2Prompt: deletePromptMock,
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

import PromptLibrary from '@/components/settings/PromptLibrary.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiField from '@/components/ui/UiField.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'

function prompt(
  id: string,
  name: string,
  content = '提示词内容',
  isFactoryDefault = false,
) {
  return {
    id,
    name,
    content,
    type: 'translate',
    revision: 1,
    isFactoryDefault,
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve
    reject = promiseReject
  })
  return { promise, resolve, reject }
}

describe('PromptLibrary', () => {
  beforeEach(() => {
    updateTranslationServiceMock.mockReset()
    updateAiVisionOcrMock.mockReset()
    toastInfoMock.mockReset()
    listV2PromptsMock.mockReset()
    deletePromptMock.mockReset()
    confirmProductActionMock.mockReset()
    listV2PromptsMock.mockResolvedValue([])
    deletePromptMock.mockResolvedValue(undefined)
    confirmProductActionMock.mockResolvedValue(true)

    settingsState.translation.openaiOptions.request.forceJsonOutput = false
    settingsState.aiVisionOcr.openaiOptions.request.forceJsonOutput = false
    settingsState.aiVisionOcr.promptMode = 'paddleocr_vl'
  })

  it('uses fixed select primitives for prompt type and mode', async () => {
    const wrapper = mount(PromptLibrary)
    await flushPromises()

    const selects = wrapper.findAllComponents(UiSelect)
    expect(selects).toHaveLength(2)

    expect(selects[0]!.props('modelValue')).toBe('translate')
    expect(selects[0]!.props('options')).toEqual(expect.arrayContaining([
      expect.objectContaining({ value: 'translate' }),
      expect.objectContaining({ value: 'textbox' }),
      expect.objectContaining({ value: 'proofreading' }),
    ]))

    expect(selects[1]!.props('modelValue')).toBe('normal')
    expect(selects[1]!.props('options')).toEqual([
      { label: '普通模式', value: 'normal' },
      { label: 'JSON格式模式', value: 'json' },
    ])

    selects[1]!.vm.$emit('change', 'json')

    expect(updateTranslationServiceMock).toHaveBeenLastCalledWith({
      forceJsonOutput: true,
    })
    expect(toastInfoMock).toHaveBeenCalledWith('已切换到JSON格式模式')
  })

  it('preserves ai vision paddleocr_vl prompt mode instead of collapsing it to normal/json', async () => {
    const wrapper = mount(PromptLibrary)
    await flushPromises()

    const selects = wrapper.findAllComponents(UiSelect)
    expect(selects.length).toBeGreaterThanOrEqual(1)

    const typeSelect = selects[0]!
    typeSelect.vm.$emit('change', 'ai_vision_ocr')
    await flushPromises()

    const refreshedSelects = wrapper.findAllComponents(UiSelect)
    expect(refreshedSelects.length).toBeGreaterThanOrEqual(2)

    const modeSelect = refreshedSelects[1]!
    expect(modeSelect.props('modelValue')).toBe('paddleocr_vl')

    modeSelect.vm.$emit('change', 'paddleocr_vl')

    expect(updateAiVisionOcrMock).toHaveBeenLastCalledWith({
      forceJsonOutput: false,
      promptMode: 'paddleocr_vl',
    })
    expect(toastInfoMock).toHaveBeenCalledWith('已切换到OCR模型提示词模式')
  })

  it('uses separate controls for selecting, loading, and deleting prompt rows', async () => {
    listV2PromptsMock.mockResolvedValue([
      prompt('prompt-default', 'default', '提示词内容', true),
      prompt('prompt-custom', 'custom'),
    ])
    const wrapper = mount(PromptLibrary)
    await flushPromises()

    const promptRow = wrapper.find('.prompt-library__item')
    expect(promptRow.element.tagName).toBe('DIV')

    const selectButton = wrapper.find('.prompt-library__select-action')
    expect(selectButton.element.tagName).toBe('BUTTON')
    expect(selectButton.attributes('aria-label')).toBe('选择提示词：default')
    expect(selectButton.attributes('aria-pressed')).toBe('false')

    await selectButton.trigger('click')
    await flushPromises()

    expect(wrapper.getComponent(UiTextarea).props('modelValue')).toBe('提示词内容')
    expect(wrapper.find('.prompt-library__select-action').attributes('aria-pressed')).toBe('true')

    const loadButton = wrapper.find('.prompt-library__load-action')
    expect(loadButton.element.tagName).toBe('BUTTON')
    expect(loadButton.attributes('aria-label')).toBe('加载提示词：default')
    expect(loadButton.getComponent(UiIconButton).props()).toMatchObject({
      label: '加载提示词：default',
      variant: 'soft',
      size: 'sm',
    })

    const deleteButton = wrapper.find('.prompt-library__delete-action')
    expect(deleteButton.element.tagName).toBe('BUTTON')
    expect(deleteButton.attributes('aria-label')).toBe('删除提示词：default')
    expect(deleteButton.getComponent(UiIconButton).props()).toMatchObject({
      label: '删除提示词：default',
      variant: 'danger',
      size: 'sm',
    })
    expect(promptRow.findAllComponents(UiIconButton)).toHaveLength(2)
  })

  it('uses icon-button primitives instead of root button skins for prompt row actions', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PromptLibrary.vue'), 'utf8')
    const rootStyle = source.match(/\.prompt-library \{(?<body>[\s\S]*?)\n\}/)

    expect(source).toContain("import UiIconButton from '@/components/ui/UiIconButton.vue'")
    expect(source).toContain('class="prompt-library__list"')
    expect(source).toContain('class="prompt-library__item"')
    expect(source).toContain('class="prompt-library__select-action"')
    expect(source).toContain('class="prompt-library__name"')
    expect(source).toContain('class="prompt-library__actions"')
    expect(source).toContain('class="prompt-library__load-action"')
    expect(source).toContain('class="prompt-library__delete-action"')
    expect(source).not.toContain('class="prompt-list"')
    expect(source).not.toContain('class="prompt-item"')
    expect(source).not.toContain('class="prompt-select"')
    expect(source).not.toContain('class="prompt-name"')
    expect(source).not.toContain('class="prompt-actions"')
    expect(rootStyle?.groups?.body ?? '').not.toMatch(/--ui-button-/)
    expect(source).not.toContain('.prompt-library__delete-action:disabled')
  })

  it('does not assert shared icon-button primitives through internal class names', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/promptLibrary.spec.ts'), 'utf8')
    const buttonClassPrefix = 'ui-' + 'button--'
    const iconButtonClassPrefix = 'ui-' + 'icon-button--'

    expect(source).not.toContain(buttonClassPrefix)
    expect(source).not.toContain(iconButtonClassPrefix)
  })

  it('routes prompt list loading and empty states through product status feedback', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PromptLibrary.vue'), 'utf8')
    expect(source).not.toContain('loading-hint')
    expect(source).not.toContain('empty-hint')
    expect(source).toContain('ProductStatusBanner')

    const pendingPrompts = deferred<ReturnType<typeof prompt>[]>()
    listV2PromptsMock.mockReturnValueOnce(pendingPrompts.promise)

    const wrapper = mount(PromptLibrary)
    await nextTick()

    const loadingBanner = wrapper.findComponent(ProductStatusBanner)
    expect(loadingBanner.exists()).toBe(true)
    expect(loadingBanner.props()).toMatchObject({
      ariaLive: 'polite',
      iconName: 'refresh',
      title: '正在加载提示词',
      tone: 'neutral',
    })
    expect(wrapper.find('.loading-hint').exists()).toBe(false)

    pendingPrompts.resolve([])
    await flushPromises()

    const emptyBanner = wrapper.findComponent(ProductStatusBanner)
    expect(emptyBanner.exists()).toBe(true)
    expect(emptyBanner.props()).toMatchObject({
      iconName: 'file-text',
      title: '暂无保存的提示词',
      tone: 'neutral',
    })
    expect(wrapper.find('.empty-hint').exists()).toBe(false)
  })

  it('routes prompt library labels and save actions through typed settings primitives', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PromptLibrary.vue'), 'utf8')

    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('prompt-editor-actions')
    expect(source).toContain('ProductActionRow')

    const wrapper = mount(PromptLibrary)
    await flushPromises()

    const fields = wrapper.findAllComponents(UiField)
    const fieldByControlId = (controlId: string) =>
      fields.find(field => field.props('controlId') === controlId)

    expect(fieldByControlId('promptType')?.props('label')).toBe('提示词类型')
    expect(fieldByControlId('promptMode')?.props('label')).toBe('提示词模式')
    expect(fieldByControlId('promptMode')?.props('hint')).toBe('适用于普通翻译场景')
    expect(fieldByControlId('promptName')?.props('label')).toBe('提示词名称')
    expect(fieldByControlId('promptContent')?.props('label')).toBe('提示词内容')
    expect(wrapper.getComponent(UiTextarea).props('variant')).toBe('panel')
  })

  it('ignores stale prompt list responses after the prompt type changes', async () => {
    const translatePrompts = deferred<ReturnType<typeof prompt>[]>()
    const visionPrompts = deferred<ReturnType<typeof prompt>[]>()
    listV2PromptsMock.mockImplementation((type?: string) =>
      type === 'ai_vision_ocr' ? visionPrompts.promise : translatePrompts.promise
    )

    const wrapper = mount(PromptLibrary)

    const typeSelect = wrapper.findAllComponents(UiSelect)[0]!
    typeSelect.vm.$emit('change', 'ai_vision_ocr')

    visionPrompts.resolve([prompt('vision-current', 'vision-current')])
    await flushPromises()
    expect(wrapper.text()).toContain('vision-current')

    translatePrompts.resolve([prompt('translate-stale', 'translate-stale')])
    await flushPromises()
    expect(wrapper.text()).toContain('vision-current')
    expect(wrapper.text()).not.toContain('translate-stale')
  })

  it('confirms before deleting custom prompts', async () => {
    listV2PromptsMock.mockResolvedValue([
      prompt('prompt-custom', 'custom'),
    ])
    confirmProductActionMock.mockResolvedValueOnce(false)

    const wrapper = mount(PromptLibrary)
    await flushPromises()

    const deleteButton = wrapper.find('button[aria-label="删除提示词：custom"]')
    await deleteButton.trigger('click')
    await flushPromises()

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '删除提示词',
      message: '确定要删除提示词“custom”吗？此操作无法撤销。',
      confirmText: '删除',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(deletePromptMock).not.toHaveBeenCalled()

    confirmProductActionMock.mockResolvedValueOnce(true)
    await deleteButton.trigger('click')
    await flushPromises()

    expect(deletePromptMock).toHaveBeenCalledWith('prompt-custom')
  })
})
