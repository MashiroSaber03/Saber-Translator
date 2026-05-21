import { mount } from '@vue/test-utils'
import { defineComponent, nextTick, reactive, ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ContinuationPanel from './ContinuationPanel.vue'

const mocks = vi.hoisted(() => ({
  insightStore: null as any,
  state: null as any,
  characterManagement: null as any,
  imageGeneration: null as any,
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
  ContinuationStateKey: Symbol('ContinuationStateKey'),
}))

vi.mock('@/composables/continuation/useCharacterManagement', () => ({
  useCharacterManagement: () => mocks.characterManagement,
  CharacterManagementKey: Symbol('CharacterManagementKey'),
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

const scriptPanelStub = defineComponent({
  emits: ['generate', 'update-script', 'save-script', 'reset-script'],
  template: '<button class="trigger-script-generate" @click="$emit(\'generate\', { referenceTokens: null, referenceImageCount: 5 })">generate</button>',
})

describe('ContinuationPanel', () => {
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

    await wrapper.find('button.btn.secondary.danger').trigger('click')
    await nextTick()

    expect(mocks.clearContinuationData).toHaveBeenCalledWith('book-1')
    expect(mocks.state.resetState).toHaveBeenCalledTimes(1)
    expect(mocks.state.initializeData).toHaveBeenCalledTimes(2)
    expect(mocks.state.currentStep.value).toBe(0)
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

    await wrapper.find('button.btn.primary').trigger('click')
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

    const stepNames = wrapper.findAll('.step-name').map(node => node.text())
    expect(stepNames).toEqual(['角色设置', '生成脚本', '页面剧情', '图片生成/导出'])
    expect(wrapper.text()).not.toContain('提示词生成')
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

    const syncButton = wrapper.find('.analysis-sync-button')
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
})
