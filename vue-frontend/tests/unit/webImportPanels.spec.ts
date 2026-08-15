import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { nextTick } from 'vue'
import WebImportLogsPanel from '@/components/translate/web-import/WebImportLogsPanel.vue'
import WebImportExtractBar from '@/components/translate/web-import/WebImportExtractBar.vue'
import WebImportFooterActions from '@/components/translate/web-import/WebImportFooterActions.vue'
import WebImportSettingsPanel from '@/components/translate/web-import/WebImportSettingsPanel.vue'
import WebImportBasicSettingsPanel from '@/components/translate/web-import/WebImportBasicSettingsPanel.vue'
import WebImportAdvancedSettingsPanel from '@/components/translate/web-import/WebImportAdvancedSettingsPanel.vue'
import WebImportPreprocessSettings from '@/components/translate/WebImportPreprocessSettings.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiPasswordField from '@/components/ui/UiPasswordField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductCollapsibleSection from '@/components/product/ProductCollapsibleSection.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import ProductLogPanel from '@/components/product/ProductLogPanel.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import type { WebImportSettings } from '@/types/webImport'

function createDraftSettings(): WebImportSettings {
  return {
    firecrawl: {
      apiKey: '',
    },
    agent: {
      provider: 'custom',
      apiKey: '',
      customBaseUrl: '',
      modelName: '',
      forceJsonOutput: true,
      useStream: false,
      maxRetries: 2,
      timeout: 60,
    },
    extraction: {
      prompt: '',
      maxIterations: 3,
    },
    download: {
      concurrency: 3,
      timeout: 30,
      retries: 2,
      delay: 0,
      useReferer: true,
    },
    advanced: {
      customCookie: '',
      customHeaders: '',
      bypassProxy: false,
    },
    imagePreprocess: {
      enabled: true,
      autoRotate: true,
      compression: {
        enabled: true,
        quality: 90,
        maxWidth: 0,
        maxHeight: 0,
      },
      formatConvert: {
        enabled: true,
        targetFormat: 'jpeg',
      },
    },
    ui: {
      showAgentLogs: true,
      autoImport: false,
    },
  }
}

function createSettingsActions() {
  return {
    setAgentApiKey: vi.fn(),
    setAgentBaseUrl: vi.fn(),
    setAgentForceJsonOutput: vi.fn(),
    setAgentMaxRetries: vi.fn(),
    setAgentModelName: vi.fn(),
    setAgentProvider: vi.fn(),
    setAgentTimeout: vi.fn(),
    setAgentUseStream: vi.fn(),
    setAutoImport: vi.fn(),
    setBypassProxy: vi.fn(),
    setCustomCookie: vi.fn(),
    setCustomHeaders: vi.fn(),
    setDownloadConcurrency: vi.fn(),
    setDownloadDelay: vi.fn(),
    setDownloadRetries: vi.fn(),
    setDownloadTimeout: vi.fn(),
    setDownloadUseReferer: vi.fn(),
    setExtractionMaxIterations: vi.fn(),
    setExtractionPrompt: vi.fn(),
    setFirecrawlApiKey: vi.fn(),
    setImageAutoRotate: vi.fn(),
    setImageCompressionEnabled: vi.fn(),
    setImageCompressionQuality: vi.fn(),
    setImageFormatConvertEnabled: vi.fn(),
    setImageMaxHeight: vi.fn(),
    setImageMaxWidth: vi.fn(),
    setImagePreprocessEnabled: vi.fn(),
    setImageTargetFormat: vi.fn(),
    setShowAgentLogs: vi.fn(),
  }
}

describe('WebImport panels', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('uses button semantics for settings disclosure', async () => {
    const wrapper = mount(WebImportSettingsPanel, {
      props: {
        activeSettingsTab: 'basic',
        agentProviderOptions: [],
        draftSettings: createDraftSettings(),
        hasUnsavedSettings: false,
        isFetchingModels: false,
        isSavingSettings: false,
        modelList: [],
        modelListOptions: [],
        providerRequiresApiKey: () => false,
        settingsExpanded: false,
        showCustomUrl: false,
        supportsFetchModels: false,
        testingAgent: false,
        testingFirecrawl: false,
        settingsActions: createSettingsActions(),
      },
    })

    const section = wrapper.getComponent(ProductCollapsibleSection)
    expect(section.props('expanded')).toBe(false)
    expect(section.props('title')).toBe('设置')
    expect(section.props('hint')).toBe('点击展开配置')

    await section.get('button').trigger('click')
    expect(wrapper.emitted('update:settingsExpanded')?.[0]).toEqual([true])
    expect(wrapper.find('.web-import-modal__settings-header').exists()).toBe(false)
  })

  it('uses typed settings field primitives for WebImport settings forms', async () => {
    const wrapper = mount(WebImportSettingsPanel, {
      props: {
        activeSettingsTab: 'basic',
        agentProviderOptions: [
          { label: 'OpenAI', value: 'openai' },
          { label: '自定义', value: 'custom' },
        ],
        draftSettings: createDraftSettings(),
        hasUnsavedSettings: false,
        isFetchingModels: false,
        isSavingSettings: false,
        modelList: ['gpt-4o-mini'],
        modelListOptions: [{ label: 'gpt-4o-mini', value: 'gpt-4o-mini' }],
        providerRequiresApiKey: () => true,
        settingsExpanded: true,
        showCustomUrl: true,
        supportsFetchModels: true,
        testingAgent: false,
        testingFirecrawl: false,
        settingsActions: createSettingsActions(),
      },
    })

    const settingsFields = wrapper
      .findAllComponents(UiField)
      .filter(field => field.props('variant') === 'settings')
    expect(settingsFields.length).toBeGreaterThanOrEqual(20)
    expect(wrapper.findAllComponents(UiNumberField).length).toBeGreaterThanOrEqual(10)
    expect(wrapper.find('.web-import-modal__form-row').exists()).toBe(false)
    expect(wrapper.find('.web-import-modal__form-label').exists()).toBe(false)
    expect(wrapper.find('.web-import-modal__form-grid').exists()).toBe(false)
    expect(wrapper.find('.web-import-basic-settings__number-input').exists()).toBe(false)
    expect(wrapper.find('.web-import-preprocess__number-input').exists()).toBe(false)
    expect(wrapper.findComponent(ProductSegmentedTabs).exists()).toBe(true)
    expect(wrapper.find('.web-import-modal__settings-tabs').exists()).toBe(false)

    const statusBanner = wrapper.getComponent(ProductStatusBanner)
    expect(statusBanner.props('tone')).toBe('success')
    expect(wrapper.text()).toContain('设置已同步')
    expect(wrapper.find('.web-import-modal__settings-actions').exists()).toBe(false)

    await wrapper.setProps({ hasUnsavedSettings: true })
    expect(wrapper.getComponent(ProductStatusBanner).props('tone')).toBe('warning')
    expect(wrapper.text()).toContain('有未保存的修改')

    const settingsActions = wrapper
      .findAllComponents(UiButton)
      .filter(button => ['取消修改', '保存设置'].includes(button.text()))
    expect(settingsActions.map(button => button.props('variant'))).toEqual(['secondary', 'primary'])

    const panelActions = wrapper
      .findAllComponents(UiButton)
      .filter(button =>
        ['测试连接', '获取模型', '测试 Agent 连接', '重置为默认'].includes(button.text())
      )
    expect(panelActions.map(button => button.props('variant'))).toEqual([
      'secondary',
      'secondary',
      'secondary',
      'secondary',
    ])
    expect(wrapper.find('.web-import-settings__reset-button').exists()).toBe(false)
  })

  it('locks the complete settings draft while its transaction is being saved', () => {
    const wrapper = mount(WebImportSettingsPanel, {
      props: {
        activeSettingsTab: 'basic',
        agentProviderOptions: [{ label: '自定义', value: 'custom' }],
        draftSettings: createDraftSettings(),
        hasUnsavedSettings: true,
        isFetchingModels: false,
        isSavingSettings: true,
        modelList: [],
        modelListOptions: [],
        providerRequiresApiKey: () => false,
        settingsActions: createSettingsActions(),
        settingsExpanded: true,
        showCustomUrl: true,
        supportsFetchModels: false,
        testingAgent: false,
        testingFirecrawl: false,
      },
    })

    const fieldset = wrapper.get('fieldset.web-import-settings__fieldset')
    expect(fieldset.attributes('disabled')).toBeDefined()
    expect(fieldset.attributes('aria-busy')).toBe('true')
    expect(wrapper.get('#webImportExtractionPrompt').element.matches(':disabled')).toBe(true)
  })

  it('binds WebImport select field labels to stable control ids', () => {
    const wrapper = mount(WebImportSettingsPanel, {
      props: {
        activeSettingsTab: 'basic',
        agentProviderOptions: [{ label: '自定义', value: 'custom' }],
        draftSettings: createDraftSettings(),
        hasUnsavedSettings: false,
        isFetchingModels: false,
        isSavingSettings: false,
        modelList: [],
        modelListOptions: [],
        providerRequiresApiKey: () => true,
        settingsExpanded: true,
        showCustomUrl: true,
        supportsFetchModels: false,
        testingAgent: false,
        testingFirecrawl: false,
        settingsActions: createSettingsActions(),
      },
    })

    const labels = wrapper.findAll('label')
    const providerLabel = labels.find(label => label.text() === '服务商')
    const targetFormatLabel = labels.find(label => label.text() === '目标格式')

    expect(providerLabel?.attributes('for')).toBe('webImportAgentProvider')
    expect(wrapper.get('#webImportAgentProvider').exists()).toBe(true)
    expect(targetFormatLabel?.attributes('for')).toBe('webImportImageTargetFormat')
    expect(wrapper.get('#webImportImageTargetFormat').exists()).toBe(true)
  })

  it('uses panel textarea variants for long-form WebImport settings', () => {
    const wrapper = mount(WebImportSettingsPanel, {
      props: {
        activeSettingsTab: 'basic',
        agentProviderOptions: [{ label: '自定义', value: 'custom' }],
        draftSettings: createDraftSettings(),
        hasUnsavedSettings: false,
        isFetchingModels: false,
        isSavingSettings: false,
        modelList: [],
        modelListOptions: [],
        providerRequiresApiKey: () => true,
        settingsExpanded: true,
        showCustomUrl: true,
        supportsFetchModels: false,
        testingAgent: false,
        testingFirecrawl: false,
        settingsActions: createSettingsActions(),
      },
    })

    const textareaVariants = new Map(
      wrapper
        .findAllComponents(UiTextarea)
        .map(textarea => [textarea.attributes('id'), textarea.props('variant')])
    )

    expect(textareaVariants.get('webImportExtractionPrompt')).toBe('panel')
    expect(textareaVariants.get('webImportCustomHeaders')).toBe('panel')
  })

  it('groups standalone WebImport Basic settings actions with product action rows', () => {
    const wrapper = mount(WebImportSettingsPanel, {
      props: {
        activeSettingsTab: 'basic',
        agentProviderOptions: [{ label: '自定义', value: 'custom' }],
        draftSettings: createDraftSettings(),
        hasUnsavedSettings: false,
        isFetchingModels: false,
        isSavingSettings: false,
        modelList: [],
        modelListOptions: [],
        providerRequiresApiKey: () => true,
        settingsExpanded: true,
        showCustomUrl: true,
        supportsFetchModels: false,
        testingAgent: false,
        testingFirecrawl: false,
        settingsActions: createSettingsActions(),
      },
    })

    const actionRows = wrapper.findAllComponents(ProductActionRow)
    expect(actionRows.map(row => row.props('ariaLabel'))).toEqual([
      'Firecrawl 操作',
      'AI Agent 操作',
      '提取提示词操作',
    ])
    expect(actionRows.every(row => row.props('justify') === 'start')).toBe(true)

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/web-import/WebImportBasicSettingsPanel.vue'),
      'utf8'
    )
    expect(source).not.toMatch(/<template #title>\s*提取设置\s*<UiButton/)
  })

  it('does not impose arbitrary upper bounds on WebImport execution settings', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/web-import/WebImportBasicSettingsPanel.vue'),
      'utf8'
    )

    for (const inputId of [
      'webImportMaxIterations',
      'webImportAgentMaxRetries',
      'webImportAgentTimeout',
      'webImportDownloadConcurrency',
      'webImportDownloadTimeout',
      'webImportDownloadRetries',
      'webImportDownloadDelay',
    ]) {
      const field = source.match(new RegExp(`input-id="${inputId}"[\\s\\S]*?/>`))?.[0]
      expect(field).toBeDefined()
      expect(field).not.toContain(':max=')
    }
  })

  it('keeps settings tab scrolling owned by the settings panel instead of the modal shell', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/web-import/WebImportSettingsPanel.vue'),
      'utf8'
    )

    expect(source).toContain('web-import-settings__tab-content')
    expect(source).not.toContain('web-import-modal__settings-tab-content')
    expect(source).not.toContain('max-height: 400px')
  })

  it('uses shared password fields for WebImport credential visibility controls', () => {
    const wrapper = mount(WebImportSettingsPanel, {
      props: {
        activeSettingsTab: 'basic',
        agentProviderOptions: [{ label: '自定义', value: 'custom' }],
        draftSettings: createDraftSettings(),
        hasUnsavedSettings: false,
        isFetchingModels: false,
        isSavingSettings: false,
        modelList: [],
        modelListOptions: [],
        providerRequiresApiKey: () => true,
        settingsActions: createSettingsActions(),
        settingsExpanded: true,
        showCustomUrl: true,
        supportsFetchModels: false,
        testingAgent: false,
        testingFirecrawl: false,
      },
    })

    const passwordFields = wrapper.findAllComponents(UiPasswordField)
    expect(passwordFields).toHaveLength(2)
    expect(passwordFields.map(field => field.props('showLabel'))).toEqual([
      '显示 Firecrawl API Key',
      '显示 AI Agent API Key',
    ])
  })

  it('keeps WebImport text fields on typed model updates instead of DOM event casts', () => {
    const basicSource = readFileSync(
      resolve(process.cwd(), 'src/components/translate/web-import/WebImportBasicSettingsPanel.vue'),
      'utf8'
    )
    const advancedSource = readFileSync(
      resolve(
        process.cwd(),
        'src/components/translate/web-import/WebImportAdvancedSettingsPanel.vue'
      ),
      'utf8'
    )

    expect(basicSource).not.toContain(
      '@input="settingsActions.setAgentBaseUrl(($event.target as HTMLInputElement).value)"'
    )
    expect(basicSource).not.toContain(
      '@input="settingsActions.setExtractionPrompt(($event.target as HTMLTextAreaElement).value)"'
    )
    expect(basicSource).toContain('AiProviderCredentialFields')
    expect(basicSource).toContain('@update:base-url="settingsActions.setAgentBaseUrl"')
    expect(basicSource).toContain('@update:model-value="settingsActions.setExtractionPrompt"')

    expect(advancedSource).not.toContain(
      '@input="settingsActions.setCustomCookie(($event.target as HTMLInputElement).value)"'
    )
    expect(advancedSource).not.toContain(
      '@input="settingsActions.setCustomHeaders(($event.target as HTMLTextAreaElement).value)"'
    )
    expect(advancedSource).toContain(
      '@update:model-value="value => settingsActions.setCustomCookie(String(value))"'
    )
    expect(advancedSource).toContain('@update:model-value="settingsActions.setCustomHeaders"')
  })

  it('delegates WebImport settings tabs to dedicated owner components', async () => {
    const wrapper = mount(WebImportSettingsPanel, {
      props: {
        activeSettingsTab: 'basic',
        agentProviderOptions: [{ label: '自定义', value: 'custom' }],
        draftSettings: createDraftSettings(),
        hasUnsavedSettings: false,
        isFetchingModels: false,
        isSavingSettings: false,
        modelList: [],
        modelListOptions: [],
        providerRequiresApiKey: () => false,
        settingsActions: createSettingsActions(),
        settingsExpanded: true,
        showCustomUrl: true,
        supportsFetchModels: false,
        testingAgent: false,
        testingFirecrawl: false,
      },
    })

    expect(wrapper.findComponent(WebImportBasicSettingsPanel).exists()).toBe(true)

    await wrapper.setProps({ activeSettingsTab: 'advanced' })

    expect(wrapper.findComponent(WebImportAdvancedSettingsPanel).exists()).toBe(true)
  })

  it('does not emit unknown WebImport settings tab ids from the shared tab primitive', async () => {
    const wrapper = mount(WebImportSettingsPanel, {
      props: {
        activeSettingsTab: 'basic',
        agentProviderOptions: [{ label: '自定义', value: 'custom' }],
        draftSettings: createDraftSettings(),
        hasUnsavedSettings: false,
        isFetchingModels: false,
        isSavingSettings: false,
        modelList: [],
        modelListOptions: [],
        providerRequiresApiKey: () => false,
        settingsActions: createSettingsActions(),
        settingsExpanded: true,
        showCustomUrl: true,
        supportsFetchModels: false,
        testingAgent: false,
        testingFirecrawl: false,
      },
    })

    await wrapper.getComponent(ProductSegmentedTabs).vm.$emit('update:activeTab', 'legacy-tab')

    expect(wrapper.emitted('update:activeSettingsTab')).toBeUndefined()

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/web-import/WebImportSettingsPanel.vue'),
      'utf8'
    )
    expect(source).not.toContain('tabId as SettingsTab')
  })

  it('uses product form sections for WebImport preprocess settings groups', () => {
    const wrapper = mount(WebImportSettingsPanel, {
      props: {
        activeSettingsTab: 'preprocess',
        agentProviderOptions: [{ label: '自定义', value: 'custom' }],
        draftSettings: createDraftSettings(),
        hasUnsavedSettings: false,
        isFetchingModels: false,
        isSavingSettings: false,
        modelList: [],
        modelListOptions: [],
        providerRequiresApiKey: () => false,
        settingsActions: createSettingsActions(),
        settingsExpanded: true,
        showCustomUrl: true,
        supportsFetchModels: false,
        testingAgent: false,
        testingFirecrawl: false,
      },
    })

    expect(wrapper.findAllComponents(ProductFormSection).map(section => section.text())).toEqual(
      expect.arrayContaining([
        expect.stringContaining('图片预处理'),
        expect.stringContaining('压缩设置'),
        expect.stringContaining('格式转换'),
      ])
    )

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/WebImportPreprocessSettings.vue'),
      'utf8'
    )
    expect(source).not.toContain('web-import-preprocess__subsection-title')
    expect(source).not.toMatch(/<h5\b/)
  })

  it('guards WebImport preprocess target format before dispatching typed actions', async () => {
    const settingsActions = createSettingsActions()
    const wrapper = mount(WebImportPreprocessSettings, {
      props: {
        draftSettings: createDraftSettings(),
        settingsActions,
      },
    })
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/WebImportPreprocessSettings.vue'),
      'utf8'
    )

    expect(source).toContain('function updateTargetFormat')
    expect(source).not.toContain('String(value) as WebImportSettings')

    const targetFormatSelect = wrapper.getComponent(UiSelect)
    targetFormatSelect.vm.$emit('update:modelValue', 'webp')
    targetFormatSelect.vm.$emit('update:modelValue', 'gif')
    await nextTick()

    expect(settingsActions.setImageTargetFormat).toHaveBeenCalledTimes(1)
    expect(settingsActions.setImageTargetFormat).toHaveBeenCalledWith('webp')
  })

  it('routes settings edits through typed panel actions instead of the store instance', async () => {
    const settingsActions = createSettingsActions()
    const wrapper = mount(WebImportSettingsPanel, {
      props: {
        activeSettingsTab: 'basic',
        agentProviderOptions: [{ label: '自定义', value: 'custom' }],
        draftSettings: createDraftSettings(),
        hasUnsavedSettings: false,
        isFetchingModels: false,
        isSavingSettings: false,
        modelList: [],
        modelListOptions: [],
        providerRequiresApiKey: () => false,
        settingsActions,
        settingsExpanded: true,
        showCustomUrl: true,
        supportsFetchModels: false,
        testingAgent: false,
        testingFirecrawl: false,
      },
    })

    await wrapper.get('#webImportFirecrawlApiKey').setValue('fc-test')

    expect(settingsActions.setFirecrawlApiKey).toHaveBeenCalledWith('fc-test')
    expect(wrapper.props()).not.toHaveProperty('webImportStore')
  })

  it('uses the shared model picker primitive for fetched agent models', () => {
    const settingsActions = createSettingsActions()
    const draftSettings = createDraftSettings()
    draftSettings.agent.modelName = 'gpt-4o-mini'

    const wrapper = mount(WebImportSettingsPanel, {
      props: {
        activeSettingsTab: 'basic',
        agentProviderOptions: [{ label: '自定义', value: 'custom' }],
        draftSettings,
        hasUnsavedSettings: false,
        isFetchingModels: false,
        isSavingSettings: false,
        modelList: ['gpt-4o-mini', 'gpt-4.1-mini'],
        modelListOptions: [
          { label: 'gpt-4o-mini', value: 'gpt-4o-mini' },
          { label: 'gpt-4.1-mini', value: 'gpt-4.1-mini' },
        ],
        providerRequiresApiKey: () => false,
        settingsActions,
        settingsExpanded: true,
        showCustomUrl: true,
        supportsFetchModels: true,
        testingAgent: false,
        testingFirecrawl: false,
      },
    })

    const modelPicker = wrapper.getComponent(UiModelPicker)
    expect(modelPicker.props('modelValue')).toBe('gpt-4o-mini')
    expect(modelPicker.props('options')).toEqual([
      { label: 'gpt-4o-mini', value: 'gpt-4o-mini' },
      { label: 'gpt-4.1-mini', value: 'gpt-4.1-mini' },
    ])
    expect(modelPicker.props('modelCount')).toBe(2)
    expect(modelPicker.props('showFetch')).toBe(true)

    modelPicker.vm.$emit('update:modelValue', 'gpt-4.1-mini')
    modelPicker.vm.$emit('change', 'gpt-4.1-mini')
    modelPicker.vm.$emit('fetch')

    expect(settingsActions.setAgentModelName).toHaveBeenCalledTimes(1)
    expect(settingsActions.setAgentModelName).toHaveBeenCalledWith('gpt-4.1-mini')
    expect(wrapper.emitted('fetch-models')).toHaveLength(1)

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/web-import/WebImportBasicSettingsPanel.vue'),
      'utf8'
    )
    expect(source).not.toContain(
      '@change="value => settingsActions.setAgentModelName(String(value))"'
    )
  })

  it('uses button semantics for logs disclosure', async () => {
    const wrapper = mount(WebImportLogsPanel, {
      props: {
        expanded: false,
        status: 'extracting',
        logs: [{ type: 'info', timestamp: '12:00', message: 'hello' }],
      },
    })

    const header = wrapper.get('button[aria-expanded]')
    expect(header.element.tagName).toBe('BUTTON')
    expect(header.attributes('aria-expanded')).toBe('false')
    const headerButton = wrapper.getComponent(UiButton)
    expect(headerButton.props('variant')).toBe('secondary')
    expect(headerButton.props('block')).toBe(true)
    expect(wrapper.findComponent(ProductLogPanel).exists()).toBe(true)
    expect(wrapper.find('.logs-header').exists()).toBe(false)

    await header.trigger('click')
    expect(wrapper.emitted('toggle')).toBeTruthy()
  })

  it('maps each WebImport log category to a distinct semantic tone', () => {
    const wrapper = mount(WebImportLogsPanel, {
      props: {
        expanded: true,
        status: 'idle',
        logs: [
          { type: 'info', timestamp: '12:00', message: 'info' },
          { type: 'tool_call', timestamp: '12:01', message: 'call' },
          { type: 'tool_result', timestamp: '12:02', message: 'result' },
          { type: 'thinking', timestamp: '12:03', message: 'thinking' },
          { type: 'error', timestamp: '12:04', message: 'error' },
        ],
      },
    })

    expect(wrapper.getComponent(ProductLogPanel).props('items')).toEqual([
      expect.objectContaining({ message: 'info', tone: 'info' }),
      expect.objectContaining({ message: 'call', tone: 'warning' }),
      expect.objectContaining({ message: 'result', tone: 'success' }),
      expect.objectContaining({ message: 'thinking', tone: 'accent' }),
      expect.objectContaining({ message: 'error', tone: 'danger' }),
    ])

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/product/ProductLogPanel.vue'),
      'utf8'
    )
    expect(source).toContain('product-log-panel__item--info')
    expect(source).toContain('color: var(--color-status-info)')
    expect(source).toContain('color: var(--color-status-success)')
  })

  it('uses product button variants for extract and footer actions', () => {
    const extractWrapper = mount(WebImportExtractBar, {
      props: {
        checkingSupport: false,
        galleryDLAvailable: true,
        galleryDLSupported: true,
        isProcessing: false,
        selectedEngine: 'auto',
        status: 'idle',
        urlInput: 'https://example.com/chapter',
      },
    })

    expect(extractWrapper.get('form').attributes('aria-label')).toBe('网页导入提取')
    expect(extractWrapper.findAllComponents(UiField).map(field => field.props('label'))).toEqual([
      '网页 URL',
      '提取引擎',
    ])
    const notice = extractWrapper
      .findAllComponents(ProductStatusBanner)
      .find(banner => banner.text().includes('请仅爬取'))
    expect(notice.props('tone')).toBe('warning')
    const extractActionRow = extractWrapper.getComponent(ProductActionRow)
    expect(extractActionRow.props('ariaLabel')).toBe('网页导入提取操作')
    expect(extractActionRow.props('justify')).toBe('start')
    expect(extractWrapper.getComponent(UiButton).props('variant')).toBe('primary')
    expect(extractWrapper.find('.web-import-extract-bar__submit').exists()).toBe(true)
    expect(extractWrapper.find('.extract-btn').exists()).toBe(false)
    expect(extractWrapper.find('.loading-spinner').exists()).toBe(false)

    const footerWrapper = mount(WebImportFooterActions, {
      props: {
        extractResult: {
          totalPages: 1,
          engine: 'gallery-dl',
          pages: [{ pageNumber: 1, imageUrl: 'https://example.com/1.jpg' }],
        },
        isProcessing: false,
        selectedCount: 1,
        status: 'idle',
      },
    })

    const buttons = footerWrapper.findAllComponents(UiButton)
    const actionRow = footerWrapper.getComponent(ProductActionRow)
    expect(actionRow.props('variant')).toBe('dialog')
    expect(actionRow.props('ariaLabel')).toBe('网页导入操作')
    expect(buttons.map(button => button.props('variant'))).toEqual(['secondary', 'primary'])
    expect(footerWrapper.find('.cancel-btn').exists()).toBe(false)
    expect(footerWrapper.find('.import-btn').exists()).toBe(false)
    expect(footerWrapper.find('.loading-spinner').exists()).toBe(false)
  })

  it('keeps the extract form responsive to modal width instead of fixed columns', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/web-import/WebImportExtractBar.vue'),
      'utf8'
    )

    expect(source).toContain('repeat(auto-fit, minmax(min(100%, 220px), 1fr))')
    expect(source).not.toContain('minmax(260px, 1fr) minmax(150px, 180px) auto')
  })

  it('guards extract engine updates before emitting typed engine values', async () => {
    const wrapper = mount(WebImportExtractBar, {
      props: {
        checkingSupport: false,
        galleryDLAvailable: true,
        galleryDLSupported: true,
        isProcessing: false,
        selectedEngine: 'auto',
        status: 'idle',
        urlInput: 'https://example.com/chapter',
      },
    })
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/web-import/WebImportExtractBar.vue'),
      'utf8'
    )

    expect(source).toContain('function updateSelectedEngine')
    expect(source).not.toContain('$event as WebImportEngine')

    const engineSelect = wrapper.getComponent(UiSelect)
    engineSelect.vm.$emit('update:modelValue', 'ai-agent')
    engineSelect.vm.$emit('update:modelValue', 'unknown-engine')
    await nextTick()

    expect(wrapper.emitted('update:selectedEngine')).toEqual([['ai-agent']])
  })

  it('renders Gallery-DL support feedback through product status banners', () => {
    const supportedWrapper = mount(WebImportExtractBar, {
      props: {
        checkingSupport: false,
        galleryDLAvailable: true,
        galleryDLSupported: true,
        isProcessing: false,
        selectedEngine: 'auto',
        status: 'idle',
        urlInput: 'https://example.com/chapter',
      },
    })

    const statusBanners = supportedWrapper.findAllComponents(ProductStatusBanner)
    expect(statusBanners.map(banner => banner.props('tone'))).toContain('success')
    expect(statusBanners.some(banner => banner.text().includes('支持 Gallery-DL 高速下载'))).toBe(
      true
    )
    expect(supportedWrapper.find('.engine-hint').exists()).toBe(false)
    expect(supportedWrapper.find('.hint-supported').exists()).toBe(false)

    const explicitAgentWrapper = mount(WebImportExtractBar, {
      props: {
        checkingSupport: false,
        galleryDLAvailable: true,
        galleryDLSupported: true,
        isProcessing: false,
        selectedEngine: 'ai-agent',
        status: 'idle',
        urlInput: 'https://example.com/chapter',
      },
    })
    expect(
      explicitAgentWrapper
        .findAllComponents(ProductStatusBanner)
        .some(
          banner =>
            banner.props('tone') === 'neutral' && banner.text().includes('将使用 AI Agent 模式')
        )
    ).toBe(true)

    const unavailableGalleryWrapper = mount(WebImportExtractBar, {
      props: {
        checkingSupport: false,
        galleryDLAvailable: false,
        galleryDLSupported: false,
        isProcessing: false,
        selectedEngine: 'gallery-dl',
        status: 'idle',
        urlInput: 'https://example.com/chapter',
      },
    })
    expect(
      unavailableGalleryWrapper
        .findAllComponents(ProductStatusBanner)
        .some(
          banner =>
            banner.props('tone') === 'warning' && banner.text().includes('无法使用 Gallery-DL')
        )
    ).toBe(true)

    const checkingWrapper = mount(WebImportExtractBar, {
      props: {
        checkingSupport: true,
        galleryDLAvailable: true,
        galleryDLSupported: false,
        isProcessing: false,
        selectedEngine: 'auto',
        status: 'idle',
        urlInput: 'https://example.com/chapter',
      },
    })
    expect(
      checkingWrapper
        .findAllComponents(ProductStatusBanner)
        .some(
          banner =>
            banner.props('tone') === 'info' && banner.text().includes('正在检查网站支持情况')
        )
    ).toBe(true)

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/web-import/WebImportExtractBar.vue'),
      'utf8'
    )
    expect(source).not.toContain('engine-hint')
    expect(source).not.toContain('hint-supported')
    expect(source).not.toContain('hint-unsupported')
  })

  it('focuses the source URL input through a typed request prop instead of exposing a child method', async () => {
    const wrapper = mount(WebImportExtractBar, {
      props: {
        checkingSupport: false,
        focusRequestId: 0,
        galleryDLAvailable: false,
        galleryDLSupported: false,
        isProcessing: false,
        selectedEngine: 'auto',
        status: 'idle',
        urlInput: '',
      },
    })
    expect(wrapper.getComponent(UiInput).exists()).toBe(true)
    const focusSpy = vi.spyOn(
      wrapper.get('#webImportSourceUrl').element as HTMLInputElement,
      'focus'
    )
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/web-import/WebImportExtractBar.vue'),
      'utf8'
    )

    await wrapper.setProps({ focusRequestId: 1 })
    await nextTick()
    await nextTick()

    expect(focusSpy).toHaveBeenCalledTimes(1)
    expect(source).not.toContain('defineExpose')
    expect(source).not.toContain('focusSourceUrlInput')
  })

  it('uses the shared spinner primitive for active extract and import states', () => {
    const extractWrapper = mount(WebImportExtractBar, {
      props: {
        checkingSupport: false,
        galleryDLAvailable: true,
        galleryDLSupported: false,
        isProcessing: true,
        selectedEngine: 'auto',
        status: 'extracting',
        urlInput: 'https://example.com/chapter',
      },
    })

    expect(extractWrapper.findComponent(UiSpinner).exists()).toBe(true)
    expect(extractWrapper.getComponent(UiSpinner).props('decorative')).toBe(true)
    expect(extractWrapper.find('.loading-spinner').exists()).toBe(false)

    const footerWrapper = mount(WebImportFooterActions, {
      props: {
        extractResult: {
          totalPages: 1,
          engine: 'gallery-dl',
          pages: [{ pageNumber: 1, imageUrl: 'https://example.com/1.jpg' }],
        },
        isProcessing: true,
        selectedCount: 1,
        status: 'downloading',
      },
    })

    expect(footerWrapper.findComponent(UiSpinner).exists()).toBe(true)
    expect(footerWrapper.getComponent(UiSpinner).props('decorative')).toBe(true)
    expect(footerWrapper.find('.loading-spinner').exists()).toBe(false)
  })
})
