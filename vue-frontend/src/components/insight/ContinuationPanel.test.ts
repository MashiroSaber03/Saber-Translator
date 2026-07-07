import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { nextTick, reactive, ref, type Ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductWizardSteps from '@/components/product/ProductWizardSteps.vue'
import ContinuationPanel from './ContinuationPanel.vue'

type ContinuationInsightStoreStub = {
  currentBookId: string
  dataRefreshKey: number
}

type ContinuationStateStub = ReturnType<typeof createStateStub>

type ContinuationCharacterManagementStub = Record<string, never>

type ContinuationImageGenerationStub = {
  isGenerating: Ref<boolean>
  generationProgress: Ref<number>
  batchGenerateImages: ReturnType<typeof vi.fn>
  regeneratePageImage: ReturnType<typeof vi.fn>
}

const mocks = vi.hoisted(() => ({
  insightStore: null as unknown as ContinuationInsightStoreStub,
  state: null as unknown as ContinuationStateStub,
  characterManagement: null as unknown as ContinuationCharacterManagementStub,
  imageGeneration: null as unknown as ContinuationImageGenerationStub,
  confirmProductAction: vi.fn(),
  clearContinuationData: vi.fn(),
  generateScriptWithRefs: vi.fn(),
  saveConfig: vi.fn(),
  saveScript: vi.fn(),
  savePages: vi.fn(),
}))

vi.mock('@/stores/insightStore', () => ({
  useInsightStore: () => mocks.insightStore,
}))

vi.mock('@/composables/continuation/useContinuationState', () => ({
  useContinuationState: () => mocks.state,
}))

vi.mock('@/composables/continuation/useCharacterManagement', () => ({
  useCharacterManagement: () => mocks.characterManagement,
}))

vi.mock('@/composables/continuation/useImageGeneration', () => ({
  useImageGeneration: () => mocks.imageGeneration,
}))

vi.mock('@/api/continuation', () => ({
  clearContinuationData: mocks.clearContinuationData,
  generateScriptWithRefs: mocks.generateScriptWithRefs,
  saveConfig: mocks.saveConfig,
  saveScript: mocks.saveScript,
  savePages: mocks.savePages,
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: mocks.confirmProductAction,
}))

function createStateStub(currentStep = 0) {
  return {
    isLoading: ref(false),
    isDataReady: ref(true),
    currentStep: ref(currentStep),
    messageType: ref<'success' | 'error' | 'info' | ''>(''),
    errorMessage: ref(''),
    successMessage: ref(''),
    pageCount: ref(10),
    styleRefPages: ref(3),
    continuationDirection: ref(''),
    characters: ref([
      {
        name: '主角',
        aliases: [],
        description: 'desc',
        forms: [],
        reference_image: '',
        enabled: true,
      },
    ]),
    chapterScript: ref(null),
    pages: ref([]),
    imageRefreshKey: ref(Date.now()),
    isGeneratingPages: ref(false),
    isSyncingAnalysis: ref(false),
    lastAnalysisSyncAt: ref(''),
    initializeData: vi.fn().mockResolvedValue(undefined),
    syncAnalysisData: vi.fn().mockResolvedValue(undefined),
    resetState: vi.fn().mockResolvedValue(undefined),
    showMessage: vi.fn(),
    getCharacterImageUrl: vi.fn().mockReturnValue(''),
    getFormImageUrl: vi.fn().mockReturnValue(''),
    getGeneratedImageUrl: vi.fn().mockReturnValue(''),
  }
}

const scriptPanelStub = {
  components: { UiButton },
  emits: ['generate', 'update-script', 'save-script', 'reset-script'],
  template: '<UiButton class="trigger-script-generate" @click="$emit(\'generate\', { referenceTokens: null, referenceImageCount: 5 })">generate</UiButton>',
}

const pageDetailsPanelStub = {
  emits: ['story-change'],
  template: '<button class="trigger-story-change" @click="$emit(\'story-change\', 1, \'story_text\', \'新剧情\')">story</button>',
}

const imagePanelPromptStub = {
  emits: ['prompt-change'],
  template: '<button class="trigger-prompt-change" @click="$emit(\'prompt-change\', 1, \'新提示词\')">prompt</button>',
}

function getButtonByText(wrapper: ReturnType<typeof mount>, text: string) {
  const button = wrapper.findAll('button').find(node => node.text().includes(text))
  expect(button).toBeTruthy()
  return button!
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve
  })
  return { promise, resolve }
}

describe('ContinuationPanel', () => {
  it('keeps its test doubles typed without any escape hatches', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/ContinuationPanel.test.ts'),
      'utf8',
    )

    expect(source).not.toContain('as ' + 'any')
  })

  beforeEach(() => {
    mocks.insightStore = reactive({
      currentBookId: 'book-1',
      dataRefreshKey: 0,
    })
    mocks.state = createStateStub()
    mocks.characterManagement = {}
    mocks.imageGeneration = {
      isGenerating: ref(false),
      generationProgress: ref(0),
      batchGenerateImages: vi.fn().mockResolvedValue(undefined),
      regeneratePageImage: vi.fn().mockResolvedValue(undefined),
    }
    mocks.confirmProductAction.mockReset().mockResolvedValue(true)
    mocks.clearContinuationData.mockReset().mockResolvedValue({ success: true })
    mocks.generateScriptWithRefs.mockReset().mockResolvedValue({
      success: true,
      script: {
        chapter_title: '新章节',
        page_count: 10,
        script_text: '新的脚本',
        generated_at: '2026-05-12T00:00:00',
      },
    })
    mocks.saveConfig.mockReset().mockResolvedValue({ success: true })
    mocks.saveScript.mockReset().mockResolvedValue({ success: true })
    mocks.savePages.mockReset().mockResolvedValue({ success: true })
  })

  it('re-initializes continuation data after clearing the workflow', async () => {
    const wrapper = mount(ContinuationPanel, {
      global: {
        stubs: {
          CharacterManagementPanel: true,
          ScriptGenerationPanel: scriptPanelStub,
          PageDetailsPanel: true,
          ImageGenerationPanel: true,
          ExportPanel: true,
        },
      },
    })

    expect(mocks.state.initializeData).toHaveBeenCalledTimes(1)

    await getButtonByText(wrapper, '清除数据重新开始').trigger('click')
    await nextTick()

    expect(mocks.clearContinuationData).toHaveBeenCalledWith('book-1')
    expect(mocks.state.resetState).toHaveBeenCalledTimes(1)
    expect(mocks.state.initializeData).toHaveBeenCalledTimes(2)
    expect(mocks.state.currentStep.value).toBe(0)
  })

  it('confirms before clearing continuation data from the settings step', async () => {
    mocks.confirmProductAction.mockResolvedValueOnce(false)
    const wrapper = mount(ContinuationPanel, {
      global: {
        stubs: {
          CharacterManagementPanel: true,
          ScriptGenerationPanel: scriptPanelStub,
          PageDetailsPanel: true,
          ImageGenerationPanel: true,
          ExportPanel: true,
        },
      },
    })

    await getButtonByText(wrapper, '清除数据重新开始').trigger('click')
    await nextTick()

    expect(mocks.confirmProductAction).toHaveBeenCalledWith({
      title: '清空续写数据',
      message: '确定要清空所有续写数据并重新开始吗？此操作不可恢复。',
      confirmText: '清空',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(mocks.clearContinuationData).not.toHaveBeenCalled()
  })

  it('persists continuation config when leaving the settings step', async () => {
    const wrapper = mount(ContinuationPanel, {
      global: {
        stubs: {
          CharacterManagementPanel: true,
          ScriptGenerationPanel: scriptPanelStub,
          PageDetailsPanel: true,
          ImageGenerationPanel: true,
          ExportPanel: true,
        },
      },
    })

    await getButtonByText(wrapper, '下一步：生成脚本').trigger('click')
    await nextTick()

    expect(mocks.saveConfig).toHaveBeenCalledWith('book-1', {
      page_count: 10,
      style_reference_pages: 3,
      continuation_direction: '',
    })
    expect(mocks.state.currentStep.value).toBe(1)
  })

  it('uses a four-step workflow without a standalone prompt step', async () => {
    const wrapper = mount(ContinuationPanel, {
      global: {
        stubs: {
          CharacterManagementPanel: true,
          ScriptGenerationPanel: scriptPanelStub,
          PageDetailsPanel: true,
          ImageGenerationPanel: true,
          ExportPanel: true,
        },
      },
    })

    const wizard = wrapper.getComponent(ProductWizardSteps)
    const stepNames = wizard.props('steps').map(step => step.label)
    expect(stepNames).toEqual(['角色设置', '生成脚本', '页面剧情', '图片生成/导出'])
    expect(wrapper.text()).not.toContain('提示词生成')
  })

  it('renders wizard steps as native buttons with current and disabled states', () => {
    const wrapper = mount(ContinuationPanel, {
      global: {
        stubs: {
          CharacterManagementPanel: true,
          ScriptGenerationPanel: scriptPanelStub,
          PageDetailsPanel: true,
          ImageGenerationPanel: true,
          ExportPanel: true,
        },
      },
    })

    const wizard = wrapper.getComponent(ProductWizardSteps)
    expect(wizard.props('activeIndex')).toBe(0)
    expect(wizard.props('steps').map(step => step.disabled)).toEqual([false, false, true, true])

    const stepButtons = wrapper.findAll('.product-wizard-steps__step')
    expect(stepButtons.map(step => step.text())).toEqual([
      '1角色设置',
      '2生成脚本',
      '3页面剧情',
      '4图片生成/导出',
    ])
    expect(stepButtons[0]?.attributes('aria-current')).toBe('step')
    expect(stepButtons[1]?.attributes('disabled')).toBeUndefined()
    expect(stepButtons[2]?.attributes('disabled')).toBeDefined()
    expect(stepButtons[3]?.attributes('disabled')).toBeDefined()
  })

  it('renders workflow navigation arrows through shared icons', () => {
    const wrapper = mount(ContinuationPanel, {
      global: {
        stubs: {
          CharacterManagementPanel: true,
          ScriptGenerationPanel: scriptPanelStub,
          PageDetailsPanel: true,
          ImageGenerationPanel: true,
          ExportPanel: true,
        },
      },
    })

    const nextButtons = wrapper.findAll('button').filter(button => button.text().includes('下一步'))
    const previousButtons = wrapper.findAll('button').filter(button => button.text().includes('上一步'))

    expect(nextButtons).toHaveLength(3)
    expect(previousButtons).toHaveLength(3)
    for (const button of nextButtons) {
      expect(button.text()).not.toContain('→')
      expect(button.findComponent(UiIcon).props('name')).toBe('chevron-right')
    }
    for (const button of previousButtons) {
      expect(button.text()).not.toContain('←')
      expect(button.findComponent(UiIcon).props('name')).toBe('chevron-left')
    }
  })

  it('renders settings controls through the shared field and form grid primitives', () => {
    const wrapper = mount(ContinuationPanel, {
      global: {
        stubs: {
          CharacterManagementPanel: true,
          ScriptGenerationPanel: scriptPanelStub,
          PageDetailsPanel: true,
          ImageGenerationPanel: true,
          ExportPanel: true,
        },
      },
    })

    expect(wrapper.findComponent(UiFormGrid).exists()).toBe(true)
    const fields = wrapper.findAllComponents(UiField)
    expect(fields.map(field => field.props('label'))).toEqual([
      '续写页数',
      '画风参考页数',
      '续写方向（可选）',
    ])
    expect(fields.map(field => field.props('variant'))).toEqual(['settings', 'settings', 'settings'])

    const numberFields = wrapper.findAllComponents(UiNumberField)
    expect(numberFields).toHaveLength(2)
    expect(numberFields[0]?.props()).toMatchObject({
      inputId: 'continuationPageCount',
      min: 5,
      max: 50,
      modelValue: 10,
    })
    expect(numberFields[1]?.props()).toMatchObject({
      inputId: 'continuationStyleRefPages',
      min: 1,
      max: 10,
      modelValue: 3,
    })
    expect(wrapper.getComponent(UiTextarea).props('variant')).toBe('panel')

    expect(wrapper.get('label[for="continuationPageCount"]').text()).toContain('续写页数')
    expect(wrapper.get('label[for="continuationDirection"]').text()).toContain('续写方向')
    expect(wrapper.find('.continuation-panel__field').exists()).toBe(false)
  })

  it('renders step actions through the shared product action row', () => {
    const wrapper = mount(ContinuationPanel, {
      global: {
        stubs: {
          CharacterManagementPanel: true,
          ScriptGenerationPanel: scriptPanelStub,
          PageDetailsPanel: true,
          ImageGenerationPanel: true,
          ExportPanel: true,
        },
      },
    })

    const actionRows = wrapper.findAllComponents(ProductActionRow)
    expect(actionRows).toHaveLength(4)
    expect(actionRows[0]?.props('justify')).toBe('between')
    expect(actionRows[0]?.props('divider')).toBe(true)
    expect(actionRows[3]?.props('justify')).toBe('start')
    expect(wrapper.find('.actions').exists()).toBe(false)
  })

  it('uses the shared product workspace scroll owner', () => {
    const wrapper = mount(ContinuationPanel, {
      global: {
        stubs: {
          CharacterManagementPanel: true,
          ScriptGenerationPanel: scriptPanelStub,
          PageDetailsPanel: true,
          ImageGenerationPanel: true,
          ExportPanel: true,
        },
      },
    })

    const workspace = wrapper.find('.product-workspace-panel--wizard')
    expect(workspace.exists()).toBe(true)
    expect(workspace.attributes('aria-label')).toBe('续写工作区')
    expect(wrapper.find('.product-workspace-panel__scroll > .continuation-panel').exists()).toBe(true)
  })

  it('keeps the wizard controls responsive inside the product scroll owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/ContinuationPanel.vue'),
      'utf8',
    )
    const wizardSource = readFileSync(
      resolve(process.cwd(), 'src/components/product/ProductWizardSteps.vue'),
      'utf8',
    )

    expect(source).toMatch(/\.continuation-panel__sync-bar\s*\{[\s\S]*flex-wrap:\s*wrap/)
    expect(source).toContain('class="continuation-panel__step-content"')
    expect(source).toContain('class="continuation-panel__step-panel"')
    expect(source).not.toMatch(/\.(?:analysis-sync-bar|analysis-sync-meta|analysis-sync-title|analysis-sync-status|analysis-sync-button|step-content|step-panel)\b/)
    expect(source).toContain('<ProductWizardSteps')
    expect(source).not.toContain('class="step-indicator"')
    expect(source).not.toContain('class="step"')
    expect(wizardSource).toMatch(/\.product-wizard-steps\s*\{[\s\S]*flex-wrap:\s*wrap/)
    expect(wizardSource).toMatch(/\.product-wizard-steps__step\s*\{[\s\S]*min-width:\s*0/)
    expect(wizardSource).toMatch(/\.product-wizard-steps__step\s*\{[\s\S]*flex:\s*1 1/)
    expect(wizardSource).toMatch(/\.product-wizard-steps__label\s*\{[\s\S]*overflow-wrap:\s*anywhere/)
    expect(source).toContain('@media (--breakpoint-sm-down)')
  })

  it('renders workflow messages through the shared product status banner', () => {
    mocks.state.successMessage.value = '脚本已保存'
    mocks.state.messageType.value = 'success'

    const wrapper = mount(ContinuationPanel, {
      global: {
        stubs: {
          CharacterManagementPanel: true,
          ScriptGenerationPanel: scriptPanelStub,
          PageDetailsPanel: true,
          ImageGenerationPanel: true,
          ExportPanel: true,
        },
      },
    })

    const banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props('tone')).toBe('success')
    expect(banner.props('ariaLive')).toBe('polite')
    expect(banner.text()).toContain('脚本已保存')
    expect(wrapper.find('.message').exists()).toBe(false)
  })

  it('surfaces config persistence failures after script generation', async () => {
    mocks.state = createStateStub(1)
    mocks.saveConfig.mockRejectedValue(new Error('配置保存失败'))

    const wrapper = mount(ContinuationPanel, {
      global: {
        stubs: {
          CharacterManagementPanel: true,
          ScriptGenerationPanel: scriptPanelStub,
          PageDetailsPanel: true,
          ImageGenerationPanel: true,
          ExportPanel: true,
        },
      },
    })

    await wrapper.find('.trigger-script-generate').trigger('click')
    await nextTick()

    expect(mocks.generateScriptWithRefs).toHaveBeenCalledWith('book-1', '', 10, undefined, 5)
    expect(mocks.state.showMessage).toHaveBeenCalledWith(
      expect.stringContaining('配置保存失败'),
      expect.any(String),
    )
  })

  it('provides a manual sync button and triggers manual sync without resetting continuation data', async () => {
    const wrapper = mount(ContinuationPanel, {
      global: {
        stubs: {
          CharacterManagementPanel: true,
          ScriptGenerationPanel: scriptPanelStub,
          PageDetailsPanel: true,
          ImageGenerationPanel: true,
          ExportPanel: true,
        },
      },
    })

    const syncButton = wrapper.find('.continuation-panel__sync-button')
    expect(syncButton.exists()).toBe(true)

    await syncButton.trigger('click')
    await nextTick()

    expect(mocks.state.syncAnalysisData).toHaveBeenCalledWith('manual')
    expect(mocks.state.resetState).not.toHaveBeenCalled()
  })

  it('auto-syncs continuation analysis data when the global insight refresh key changes', async () => {
    mount(ContinuationPanel, {
      global: {
        stubs: {
          CharacterManagementPanel: true,
          ScriptGenerationPanel: scriptPanelStub,
          PageDetailsPanel: true,
          ImageGenerationPanel: true,
          ExportPanel: true,
        },
      },
    })

    expect(mocks.state.syncAnalysisData).not.toHaveBeenCalled()

    mocks.insightStore.dataRefreshKey = Date.now()
    await nextTick()

    expect(mocks.state.syncAnalysisData).toHaveBeenCalledWith('auto')
  })

  it('clears pending story autosave when the panel unmounts', async () => {
    vi.useFakeTimers()
    mocks.state = createStateStub(2)
    mocks.state.pages.value = [
      {
        page_number: 1,
        continuity_text: '承接',
        story_text: '剧情',
        dialogue_text: '对白',
        characters: ['主角'],
        character_forms: [],
        final_prompt: '',
        image_url: '',
        previous_url: '',
        status: 'pending',
      },
    ]

    try {
      const wrapper = mount(ContinuationPanel, {
        global: {
          stubs: {
            CharacterManagementPanel: true,
            ScriptGenerationPanel: scriptPanelStub,
            PageDetailsPanel: pageDetailsPanelStub,
            ImageGenerationPanel: true,
            ExportPanel: true,
          },
        },
      })

      await wrapper.find('.trigger-story-change').trigger('click')
      wrapper.unmount()
      await vi.advanceTimersByTimeAsync(600)

      expect(mocks.savePages).not.toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
    }
  })

  it('cancels pending story autosave before clearing the workflow', async () => {
    vi.useFakeTimers()
    mocks.state = createStateStub(2)
    const page = {
      page_number: 1,
      continuity_text: '承接',
      story_text: '旧剧情',
      dialogue_text: '对白',
      characters: ['主角'],
      character_forms: [],
      final_prompt: '',
      image_url: '',
      previous_url: '',
      status: 'pending',
    }
    mocks.state.pages.value = [page]
    const clearRequest = deferred<{ success: boolean }>()
    mocks.clearContinuationData.mockReturnValueOnce(clearRequest.promise)

    try {
      const wrapper = mount(ContinuationPanel, {
        global: {
          stubs: {
            CharacterManagementPanel: true,
            ScriptGenerationPanel: scriptPanelStub,
            PageDetailsPanel: pageDetailsPanelStub,
            ImageGenerationPanel: true,
            ExportPanel: true,
          },
        },
      })

      await wrapper.find('.trigger-story-change').trigger('click')
      await getButtonByText(wrapper, '清除数据重新开始').trigger('click')
      await vi.advanceTimersByTimeAsync(600)

      expect(mocks.savePages).not.toHaveBeenCalled()

      clearRequest.resolve({ success: true })
      await nextTick()
    } finally {
      vi.useRealTimers()
    }
  })

  it('owns page-detail story edits and debounces persistence', async () => {
    vi.useFakeTimers()
    mocks.state = createStateStub(2)
    const page = {
      page_number: 1,
      continuity_text: '承接',
      story_text: '旧剧情',
      dialogue_text: '对白',
      characters: ['主角'],
      character_forms: [],
      final_prompt: '',
      image_url: '',
      previous_url: '',
      status: 'pending',
    }
    mocks.state.pages.value = [page]

    try {
      const wrapper = mount(ContinuationPanel, {
        global: {
          stubs: {
            CharacterManagementPanel: true,
            ScriptGenerationPanel: scriptPanelStub,
            PageDetailsPanel: pageDetailsPanelStub,
            ImageGenerationPanel: true,
            ExportPanel: true,
          },
        },
      })

      await wrapper.find('.trigger-story-change').trigger('click')

      expect(page.story_text).toBe('新剧情')
      expect(mocks.savePages).not.toHaveBeenCalled()

      await vi.advanceTimersByTimeAsync(600)

      expect(mocks.savePages).toHaveBeenCalledWith('book-1', [page])
    } finally {
      vi.useRealTimers()
    }
  })

  it('owns prompt edits emitted from the image generation panel and debounces persistence', async () => {
    vi.useFakeTimers()
    mocks.state = createStateStub(3)
    const page = {
      page_number: 1,
      continuity_text: '承接',
      story_text: '剧情',
      dialogue_text: '对白',
      characters: ['主角'],
      character_forms: [],
      final_prompt: '旧提示词',
      image_url: '',
      previous_url: '',
      status: 'pending',
    }
    mocks.state.pages.value = [page]

    try {
      const wrapper = mount(ContinuationPanel, {
        global: {
          stubs: {
            CharacterManagementPanel: true,
            ScriptGenerationPanel: scriptPanelStub,
            PageDetailsPanel: true,
            ImageGenerationPanel: imagePanelPromptStub,
            ExportPanel: true,
          },
        },
      })

      await wrapper.find('.trigger-prompt-change').trigger('click')

      expect(page.final_prompt).toBe('新提示词')
      expect(mocks.savePages).not.toHaveBeenCalled()

      await vi.advanceTimersByTimeAsync(600)

      expect(mocks.savePages).toHaveBeenCalledWith('book-1', [page])
    } finally {
      vi.useRealTimers()
    }
  })
})
