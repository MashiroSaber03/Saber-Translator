import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, ref } from 'vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const {
  deletePromptFromLibraryMock,
  savePromptToLibraryMock,
  getDefaultPromptsMock,
  getPromptsLibraryMock,
  confirmProductActionMock,
  requestProductTextInputMock,
} = vi.hoisted(() => ({
  deletePromptFromLibraryMock: vi.fn(),
  savePromptToLibraryMock: vi.fn(),
  getDefaultPromptsMock: vi.fn(),
  getPromptsLibraryMock: vi.fn(),
  confirmProductActionMock: vi.fn(),
  requestProductTextInputMock: vi.fn(),
}))

vi.mock('@/api/insight', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/insight')>()
  return {
    ...actual,
    __v_isRef: false,
    PROMPT_METADATA: {
      batch_analysis: { label: '批量分析', hint: '批量分析提示词' },
    },
    deletePromptFromLibrary: deletePromptFromLibraryMock,
    getDefaultPrompts: getDefaultPromptsMock,
    getPromptsLibrary: getPromptsLibraryMock,
    importPromptsLibrary: vi.fn(),
    savePromptToLibrary: savePromptToLibraryMock,
  }
})

vi.mock('@/api/continuation', () => ({
  exportAsImages: vi.fn(),
  exportAsPdf: vi.fn(),
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

vi.mock('@/composables/useProductTextInput', () => ({
  requestProductTextInput: requestProductTextInputMock,
}))

import ExportPanel from '@/components/insight/continuation/ExportPanel.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChoiceCardGrid from '@/components/product/ProductChoiceCardGrid.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import PromptsSettingsTab from '@/components/insight/settings/PromptsSettingsTab.vue'
import InsightSettingsPanel from '@/components/insight/settings/InsightSettingsPanel.vue'
import ChatComposer from '@/components/insight/studio/preview/ChatComposer.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import type { PageContent } from '@/api/continuation'
import type { ContinuationState } from '@/composables/continuation/useContinuationState'

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(res => {
    resolve = res
  })
  return { promise, resolve }
}

function createContinuationPage(overrides: Partial<PageContent> = {}): PageContent {
  return {
    page_number: 1,
    continuity_text: '',
    story_text: '',
    dialogue_text: '',
    characters: [],
    final_prompt: '',
    image_url: '',
    previous_url: '',
    status: 'generated',
    ...overrides,
  }
}

function createContinuationState(overrides: Partial<ContinuationState> = {}): ContinuationState {
  return {
    isLoading: ref(false),
    isDataReady: ref(true),
    isSyncingAnalysis: ref(false),
    currentStep: ref(0),
    messageType: ref(''),
    errorMessage: ref(''),
    successMessage: ref(''),
    lastAnalysisSyncAt: ref(''),
    pageCount: ref(1),
    styleRefPages: ref(0),
    continuationDirection: ref(''),
    characters: ref([]),
    chapterScript: ref(null),
    pages: ref([createContinuationPage()]),
    imageRefreshKey: ref(0),
    isGeneratingPages: ref(false),
    initializeData: vi.fn(),
    syncAnalysisData: vi.fn(),
    resetState: vi.fn(),
    showMessage: vi.fn(),
    getCharacterImageUrl: vi.fn((characterName: string) => characterName),
    getFormImageUrl: vi.fn((imagePath: string) => imagePath),
    getGeneratedImageUrl: vi.fn((imagePath: string) => imagePath),
    ...overrides,
  }
}

describe('Insight card-like controls', () => {
  beforeEach(() => {
    const pinia = createPinia()
    setActivePinia(pinia)
    deletePromptFromLibraryMock.mockReset()
    savePromptToLibraryMock.mockReset()
    getDefaultPromptsMock.mockReset()
    getPromptsLibraryMock.mockReset()
    confirmProductActionMock.mockReset()
    requestProductTextInputMock.mockReset()
    confirmProductActionMock.mockResolvedValue(true)
    requestProductTextInputMock.mockResolvedValue('收藏提示词')
    getDefaultPromptsMock.mockResolvedValue({
      success: true,
      prompts: {
        batch_analysis: '默认提示词',
        chapter_summary: '',
        qa_response: '',
        segment_summary: '',
      },
    })
    getPromptsLibraryMock.mockResolvedValue({
      success: true,
      library: [
        {
          id: 'prompt-1',
          name: '战斗分析',
          type: 'batch_analysis',
          content: '分析战斗场面',
          created_at: '2026-05-21T10:00:00Z',
        },
      ],
    })
    savePromptToLibraryMock.mockResolvedValue({ success: true })
    deletePromptFromLibraryMock.mockResolvedValue({ success: true })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('keeps Insight business owner CSS on current semantic tokens without fallback chains', () => {
    const businessOwnerFiles = [
      'src/components/insight/ContinuationPanel.vue',
      'src/components/insight/continuation/PageDetailsPanel.vue',
      'src/components/insight/continuation/ImageGenerationPanel.vue',
      'src/components/insight/continuation/ScriptGenerationPanel.vue',
      'src/components/insight/continuation/OrthographicDialog.vue',
      'src/components/insight/continuation/ExportPanel.vue',
      'src/components/insight/settings/InsightSettingsPanel.vue',
      'src/components/insight/settings/BatchSettingsTab.vue',
      'src/components/insight/settings/PromptsSettingsTab.vue',
    ]

    for (const file of businessOwnerFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).not.toMatch(/var\(--color-border-muted,\s*var\(--color-border-(?:default|subtle)\)\)/)
      expect(source, file).not.toContain('var(--color-text-supporting, var(--color-text-secondary))')
    }
  })

  it('uses separate controls for loading and deleting saved prompts', async () => {
    const wrapper = mount(PromptsSettingsTab, {
      global: {
        plugins: [createPinia()],
      },
    })
    await flushPromises()

    const promptRow = wrapper.find('.prompts-settings-tab__saved-item')
    expect(promptRow.element.tagName).toBe('DIV')

    const loadButton = wrapper.find('.prompts-settings-tab__saved-load')
    expect(loadButton.element.tagName).toBe('BUTTON')
    expect(loadButton.attributes('aria-label')).toBe('加载提示词：战斗分析')

    await loadButton.trigger('click')
    expect(wrapper.emitted('showMessage')?.[0]).toEqual(['已加载提示词: 战斗分析', 'success'])

    const deleteButton = wrapper.get('[aria-label="删除提示词：战斗分析"]')
    expect(deleteButton.element.tagName).toBe('BUTTON')
    expect(deleteButton.attributes('aria-label')).toBe('删除提示词：战斗分析')
  })

  it('uses product confirmation for prompt reset and deletion', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const wrapper = mount(PromptsSettingsTab, {
      global: {
        plugins: [createPinia()],
      },
    })
    await flushPromises()

    const resetButton = wrapper.findAll('button').find(button => button.text().includes('重置'))
    expect(resetButton).toBeTruthy()
    await resetButton!.trigger('click')
    await flushPromises()

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '重置提示词',
      message: '确定要重置为默认提示词吗？当前编辑的内容将丢失。',
      confirmText: '重置',
      cancelText: '取消',
      tone: 'danger',
    })

    const deleteButton = wrapper.get('[aria-label="删除提示词：战斗分析"]')
    await deleteButton.trigger('click')
    await flushPromises()

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '删除提示词',
      message: '确定要删除这个提示词吗？',
      confirmText: '删除',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(confirmSpy).not.toHaveBeenCalled()
    expect(deletePromptFromLibraryMock).toHaveBeenCalledWith('prompt-1')
  })

  it('uses product text input when saving a prompt to the library', async () => {
    const promptSpy = vi.spyOn(window, 'prompt').mockReturnValue('收藏提示词')
    const wrapper = mount(PromptsSettingsTab, {
      global: {
        plugins: [createPinia()],
      },
    })
    await flushPromises()

    await wrapper.get('textarea').setValue('新的提示词内容')
    const saveButton = wrapper.findAll('button').find(button => button.text().includes('保存到库'))
    expect(saveButton).toBeTruthy()
    await saveButton!.trigger('click')
    await flushPromises()

    expect(requestProductTextInputMock).toHaveBeenCalledWith({
      title: '保存提示词',
      message: '请输入提示词名称：',
      placeholder: '提示词名称',
      confirmText: '保存',
      cancelText: '取消',
    })
    expect(promptSpy).not.toHaveBeenCalled()
    expect(savePromptToLibraryMock).toHaveBeenCalledWith(expect.objectContaining({
      name: '收藏提示词',
      content: '新的提示词内容',
      type: 'batch_analysis',
    }))
  })

  it('uses the fixed select primitive for prompt type selection', async () => {
    const wrapper = mount(PromptsSettingsTab, {
      global: {
        plugins: [createPinia()],
      },
    })
    await flushPromises()

    const promptTypeSelect = wrapper.getComponent(UiSelect)
    expect(promptTypeSelect.props('modelValue')).toBe('batch_analysis')
    expect(promptTypeSelect.props('options')).toEqual(expect.arrayContaining([
      expect.objectContaining({ value: 'qa_response', label: expect.stringContaining('问答') }),
    ]))
  })

  it('uses shared product primitives for prompt settings layout and saved prompts', async () => {
    const wrapper = mount(PromptsSettingsTab, {
      global: {
        plugins: [createPinia()],
      },
    })
    await flushPromises()

    expect(wrapper.getComponent(InsightSettingsPanel).props('description')).toContain('提示词模板')
    expect(wrapper.findAllComponents(UiField).length).toBeGreaterThanOrEqual(2)
    expect(wrapper.findAllComponents(ProductActionRow).length).toBeGreaterThanOrEqual(2)
    expect(wrapper.findComponent(ProductRecordCard).exists()).toBe(true)
    expect(wrapper.findComponent(ProductChipList).exists()).toBe(true)
    expect(wrapper.findComponent(UiIconButton).props('variant')).toBe('danger')
    expect(wrapper.find('.insight-settings-field').exists()).toBe(false)
    expect(wrapper.find('.form-hint').exists()).toBe(false)
    expect(wrapper.find('.prompt-type-badge').exists()).toBe(false)
    expect(wrapper.find('.button-icon-sm').exists()).toBe(false)
    expect(wrapper.find('.saved-prompt-delete').exists()).toBe(false)
  })

  it('uses product status feedback for saved prompt loading and empty states', async () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/settings/PromptsSettingsTab.vue'),
      'utf8',
    )
    expect(source).toContain('ProductStatusBanner')
    expect(source).not.toContain('placeholder-text')
    expect(source).not.toContain('loading-text')

    const pendingLibrary = deferred<{ success: true; library: [] }>()
    getPromptsLibraryMock.mockReturnValueOnce(pendingLibrary.promise)

    const wrapper = mount(PromptsSettingsTab, {
      global: {
        plugins: [createPinia()],
      },
    })
    await nextTick()
    await flushPromises()

    let banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props()).toMatchObject({
      ariaLive: 'polite',
      iconName: 'refresh',
      title: '正在加载提示词库',
      tone: 'neutral',
    })

    pendingLibrary.resolve({ success: true, library: [] })
    await flushPromises()

    banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props()).toMatchObject({
      iconName: 'file-text',
      title: '暂无保存的提示词',
      tone: 'neutral',
    })
  })

  it('uses typed panel textarea chrome for the prompt editor', async () => {
    const wrapper = mount(PromptsSettingsTab, {
      global: {
        plugins: [createPinia()],
      },
    })
    await flushPromises()

    expect(wrapper.get('textarea.prompts-settings-tab__editor').getComponent(UiTextarea).props()).toMatchObject({
      variant: 'panel',
      size: 'lg',
      rows: '12',
    })

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/settings/PromptsSettingsTab.vue'),
      'utf8',
    )
    expect(source).toContain('variant="panel"')
    expect(source).not.toMatch(/--ui-textarea-/)
  })

  it('keeps prompt settings hooks under the prompts-settings-tab owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/settings/PromptsSettingsTab.vue'),
      'utf8',
    )
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''
    const oldHooks = [
      'prompts-settings',
      'prompt-editor',
      'section-divider',
      'prompts-library-section',
      'library-header',
      'saved-prompts-list',
      'saved-prompt-item',
      'saved-prompt-card',
      'saved-prompt-load',
      'prompt-name',
    ]

    for (const hook of oldHooks) {
      const escapedHook = hook.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      expect(source).not.toMatch(new RegExp(`(?<![\\w-])${escapedHook}(?![\\w-])`))
    }
    expect(source).toContain('prompts-settings-tab__editor')
    expect(source).toContain('prompts-settings-tab__saved-load')
    expect(source).toContain('prompts-settings-tab__prompt-name')
    expect(styleBlock).not.toMatch(/\.prompts-settings\s+\./)
  })

  it('does not assert shared textarea primitives through internal class names', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/insightCardControls.spec.ts'), 'utf8')
    const textareaClassPrefix = 'ui-' + 'textarea--'

    expect(source).not.toContain(textareaClassPrefix)
  })

  it('keeps continuation panel fixtures typed to the current state contract', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/insightCardControls.spec.ts'), 'utf8')

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
  })

  it('keeps the Insight settings panel shell from skinning primitive controls', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/settings/InsightSettingsPanel.vue'),
      'utf8'
    )
    const rootStyle = source.match(/\.insight-settings-panel \{(?<body>[\s\S]*?)\n\}/)

    expect(rootStyle?.groups?.body ?? '').not.toMatch(/--ui-(input|select|button)-/)
  })

  it('opens prompt import through the component file-input owner', async () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/settings/PromptsSettingsTab.vue'),
      'utf8'
    )

    expect(source).not.toContain("document.getElementById('promptsFileInput')")
    expect(source).not.toContain('id="promptsFileInput"')
    expect(source).toContain('ref="promptsImportInput"')
    expect(source).toContain('@files-change="handlePromptsFileImport"')
    expect(source).not.toContain('@change="handlePromptsFileImport"')
    expect(source).not.toContain('target.files')
    expect(source).not.toContain("target.value = ''")

    const wrapper = mount(PromptsSettingsTab, {
      global: {
        plugins: [createPinia()],
      },
    })
    await flushPromises()

    const fileInput = wrapper.getComponent(UiFileInput)
    const clickSpy = vi
      .spyOn(fileInput.element as HTMLInputElement, 'click')
      .mockImplementation(() => undefined)
    const importButton = wrapper.findAll('button').find(button => button.text().includes('导入'))
    expect(importButton).toBeTruthy()

    await importButton!.trigger('click')

    expect(clickSpy).toHaveBeenCalledTimes(1)
  })

  it('uses a passive pending attachment card with an explicit remove button', async () => {
    const createObjectURLSpy = vi
      .spyOn(URL, 'createObjectURL')
      .mockReturnValue('blob:attachment-preview')
    const revokeObjectURLSpy = vi
      .spyOn(URL, 'revokeObjectURL')
      .mockImplementation(() => {})
    const wrapper = mount(ChatComposer, {
      props: {
        chatStreaming: false,
      },
    })
    const file = new File(['image'], 'scene.png', { type: 'image/png' })
    const input = wrapper.find('input[type="file"]')
    Object.defineProperty(input.element, 'files', {
      value: [file],
      configurable: true,
    })

    await input.trigger('change')

    expect(createObjectURLSpy).toHaveBeenCalledWith(file)
    expect(wrapper.find('.studio-chat-composer__pending-card').element.tagName).toBe('DIV')

    const removeButton = wrapper.find('.studio-chat-composer__pending-remove')
    expect(removeButton.element.tagName).toBe('BUTTON')
    expect(removeButton.attributes('aria-label')).toBe('移除附件：scene.png')
    expect(removeButton.getComponent(UiIcon).props('name')).toBe('x')
    expect(removeButton.text()).not.toContain('×')

    await removeButton.trigger('click')
    expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:attachment-preview')
  })

  it('uses product choice cards for export format selection', async () => {
    const state = createContinuationState()

    const wrapper = mount(ExportPanel, {
      props: {
        bookId: 'book-1',
        generatedCount: 1,
        state,
      },
    })

    const formatGrid = wrapper.getComponent(ProductChoiceCardGrid)
    expect(formatGrid.props('accessibilityLabel')).toBe('导出格式')
    expect(formatGrid.props('modelValue')).toBe('images')
    expect(formatGrid.props('items')).toEqual([
      { id: 'images', label: '图片 ZIP', description: '所有页面打包下载', iconName: 'image' },
      { id: 'pdf', label: 'PDF 文档', description: '方便阅读和分享', iconName: 'file-text' },
    ])

    formatGrid.vm.$emit('select', 'pdf')
    await wrapper.vm.$nextTick()

    expect(formatGrid.props('modelValue')).toBe('pdf')
  })

  it('does not cap the export step to a fixed narrow column', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/continuation/ExportPanel.vue'),
      'utf8'
    )

    expect(source).not.toContain('max-width: 600px')
  })

  it('does not override shared button primitive variables in the export panel root', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/continuation/ExportPanel.vue'),
      'utf8'
    )
    const rootStyle = source.match(/\.continuation-export-panel \{(?<body>[\s\S]*?)\n\}/)

    expect(rootStyle?.groups?.body ?? '').not.toMatch(/--ui-button-/)
    expect(source).toContain('class="continuation-export-panel"')
    expect(source).toContain('class="continuation-export-panel__summary-text"')
    expect(source).toContain('class="continuation-export-panel__summary-count"')
    expect(source).not.toContain('class="export-panel"')
    expect(source).not.toContain('.continuation-export-panel__summary p')
    expect(source).not.toContain('.continuation-export-panel__summary strong')
  })

  it('uses product action-row alignment for export controls', () => {
    const state = createContinuationState()

    const wrapper = mount(ExportPanel, {
      props: {
        bookId: 'book-1',
        generatedCount: 1,
        state,
      },
    })

    const actionRow = wrapper.getComponent(ProductActionRow)
    expect(actionRow.props('ariaLabel')).toBe('导出操作')
    expect(actionRow.props('justify')).toBe('center')
    expect(actionRow.classes()).not.toContain('export-actions')
  })

  it('uses product confirmation before clearing continuation data', async () => {
    const state = createContinuationState()
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)

    const wrapper = mount(ExportPanel, {
      props: {
        bookId: 'book-1',
        generatedCount: 1,
        state,
      },
    })

    const actionRow = wrapper.getComponent(ProductActionRow)
    expect(actionRow.props('ariaLabel')).toBe('导出操作')

    await wrapper.findAll('button').find(button => button.text().includes('清空并重新开始'))!.trigger('click')
    await flushPromises()

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '清空续写数据',
      message: '确定要清空所有续写数据并重新开始吗？此操作不可恢复。',
      confirmText: '清空',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(confirmSpy).not.toHaveBeenCalled()
    expect(wrapper.emitted('clear-and-restart')).toHaveLength(1)
  })
})
