import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia, type Pinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import DetectionSettings from '@/components/settings/DetectionSettings.vue'
import HqTranslationSettings from '@/components/settings/HqTranslationSettings.vue'
import OcrSettings from '@/components/settings/OcrSettings.vue'
import ProofreadingSettings from '@/components/settings/ProofreadingSettings.vue'
import UiCombobox from '@/components/ui/UiCombobox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { useSettingsStore } from '@/stores/settings'

const { fetchModelsMock } = vi.hoisted(() => ({
  fetchModelsMock: vi.fn(),
}))

vi.mock('@/api/v2/diagnostics', () => ({
  fetchModels: fetchModelsMock,
  testAiTranslateConnection: vi.fn(),
  testAiVisionOcrConnection: vi.fn(),
  testBaiduOcrConnection: vi.fn(),
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
  }),
}))

const globalMountOptions = (pinia: Pinia) => ({
  global: {
    plugins: [pinia],
    stubs: {
      OpenAIExtraBodyEditor: true,
      ParallelSettings: true,
      SavedPromptsPicker: true,
    },
  },
})

function mountedSelectOptionValues(wrapper: ReturnType<typeof mount>) {
  return wrapper.findAllComponents(UiSelect).map(select =>
    (select.props('options') || []).map((option: { value: string | number }) => option.value)
  )
}

function createDeferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolver) => {
    resolve = resolver
  })
  return { promise, resolve }
}

function findSelectByOptionValues(wrapper: ReturnType<typeof mount>, values: string[]) {
  return wrapper.findAllComponents(UiSelect).find(select => {
    const optionValues = (select.props('options') || []).map((option: { value: string | number }) => String(option.value))
    return values.every(value => optionValues.includes(value))
  })
}

function mountedComboboxOptionValues(wrapper: ReturnType<typeof mount>) {
  return wrapper.findAllComponents(UiCombobox).flatMap(combobox =>
    (combobox.props('options') || []).map((option: { value: string | number }) => String(option.value))
  )
}

describe('settings provider select contracts', () => {
  let pinia: Pinia

  beforeEach(() => {
    pinia = createPinia()
    setActivePinia(pinia)
    localStorage.clear()
    fetchModelsMock.mockReset()
  })

  it('uses fixed select primitives for model-provider fields outside dynamic model lists', () => {
    const store = useSettingsStore()
    store.setProofreadingEnabled(true)
    store.addProofreadingRound({
      name: '第一轮校对',
      provider: 'siliconflow',
      apiKey: '',
      modelName: '',
      customBaseUrl: '',
      openaiOptions: {
        request: {
          forceJsonOutput: false,
        },
        execution: {
          useStream: true,
          rpmLimit: 0,
          transportRetries: 1,
          businessRetries: 1,
        },
      },
      batchSize: 3,
      prompt: '校对提示词',
    })

    const hqWrapper = mount(HqTranslationSettings, globalMountOptions(pinia))
    const ocrWrapper = mount(OcrSettings, globalMountOptions(pinia))
    const proofreadingWrapper = mount(ProofreadingSettings, globalMountOptions(pinia))

    expect(mountedSelectOptionValues(hqWrapper)).toContainEqual(expect.arrayContaining(['siliconflow', 'custom']))
    expect(mountedSelectOptionValues(ocrWrapper)).toContainEqual(expect.arrayContaining(['gemini', 'custom']))
    expect(mountedSelectOptionValues(proofreadingWrapper)).toContainEqual(expect.arrayContaining(['siliconflow', 'custom']))
  })

  it('uses fixed select primitives for flat OCR enum fields', () => {
    const wrapper = mount(OcrSettings, globalMountOptions(pinia))
    const optionValues = mountedSelectOptionValues(wrapper)

    expect(optionValues).toContainEqual(expect.arrayContaining(['manga_ocr', 'ai_vision']))
    expect(optionValues).toContainEqual(expect.arrayContaining(['standard', 'high_precision']))
    expect(optionValues).toContainEqual(expect.arrayContaining(['auto_detect', 'JAP']))
    expect(optionValues).toContainEqual(expect.arrayContaining(['normal', 'json', 'paddleocr_vl']))
  })

  it('uses a fixed select primitive for the text detector field', () => {
    const wrapper = mount(DetectionSettings, globalMountOptions(pinia))
    const optionValues = mountedSelectOptionValues(wrapper)

    expect(optionValues).toContainEqual(expect.arrayContaining(['ctd', 'yolo', 'default']))
  })

  it('routes detection setting labels and hints through typed UiField props', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/DetectionSettings.vue'), 'utf8')

    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('class="ui-form-hint"')
    expect(source).toContain('label="检测器类型"')
    expect(source).toContain('control-id="settingsTextDetector"')

    const wrapper = mount(DetectionSettings, globalMountOptions(pinia))
    const fields = wrapper.findAllComponents(UiField)
    const fieldByLabel = (label: string) => fields.find(field => field.props('label') === label)

    expect(fieldByLabel('检测器类型')?.props('controlId')).toBe('settingsTextDetector')
    expect(fieldByLabel('最小文本框面积占比 (%)')?.props('hint')).toContain('极小文本框')
    expect(fieldByLabel('SaberYOLO 拆分阈值 (%)')?.props('hint')).toContain('默认 50%')
    expect(fieldByLabel('掩膜膨胀大小')?.props('hint')).toBe('掩膜膨胀像素数')
  })

  it('routes HQ translation labels and hints through typed UiField props', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/HqTranslationSettings.vue'), 'utf8')

    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('class="ui-form-hint"')

    const wrapper = mount(HqTranslationSettings, globalMountOptions(pinia))
    const fields = wrapper.findAllComponents(UiField)
    const fieldByLabel = (label: string) => fields.find(field => field.props('label') === label)

    expect(fieldByLabel('服务商')?.props('controlId')).toBe('settingsHqTranslateProvider')
    expect(fieldByLabel('API Key')?.props('controlId')).toBe('settingsHqApiKey')
    expect(fieldByLabel('Base URL')?.props('controlId')).toBe('settingsHqCustomBaseUrl')
    expect(fieldByLabel('模型名称')?.props('controlId')).toBe('settingsHqModelName')
    expect(fieldByLabel('批次大小')?.props('hint')).toBe('每批处理的图片数量 (推荐3-5张)')
    expect(fieldByLabel('RPM限制')?.props('hint')).toBe('每分钟请求数，0表示无限制')
    expect(fieldByLabel('重试次数')?.props('hint')).toBe('业务重试：空结果/结构解析失败')
    expect(fieldByLabel('传输重试')?.props('hint')).toBe('网络超时/429/5xx')

    expect(fields.map(field => field.props('hint')).filter(Boolean)).toEqual(expect.arrayContaining([
      '使用 response_format: json_object',
      '使用流式API调用',
    ]))
  })

  it('routes proofreading round labels and hints through typed UiField props', () => {
    const store = useSettingsStore()
    store.setProofreadingEnabled(true)
    store.addProofreadingRound({
      name: '第一轮校对',
      provider: 'siliconflow',
      apiKey: '',
      modelName: '',
      customBaseUrl: '',
      openaiOptions: {
        request: {
          forceJsonOutput: false,
        },
        execution: {
          useStream: true,
          rpmLimit: 0,
          transportRetries: 1,
          businessRetries: 1,
        },
      },
      batchSize: 3,
      prompt: '校对提示词',
    })
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/ProofreadingSettings.vue'), 'utf8')

    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('class="ui-form-hint"')

    const wrapper = mount(ProofreadingSettings, globalMountOptions(pinia))
    const fields = wrapper.findAllComponents(UiField)
    const fieldByControlId = (controlId: string) =>
      fields.find(field => field.props('controlId') === controlId)

    expect(fieldByControlId('settingsProofreadingMaxRetries')?.props('label')).toBe('全局重试次数')
    expect(fieldByControlId('proofreadingRound0Name')?.props('label')).toBe('轮次名称')
    expect(fieldByControlId('proofreadingRound0Provider')?.props('label')).toBe('服务商')
    expect(fieldByControlId('proofreadingRound0ApiKey')?.props('label')).toBe('API Key')
    expect(fieldByControlId('proofreadingRound0BaseUrl')?.props('label')).toBe('Base URL')
    expect(fieldByControlId('proofreadingRound0ModelName')?.props('label')).toBe('模型名称')
    expect(fieldByControlId('proofreadingRound0BatchSize')?.props('label')).toBe('批次大小')
    expect(fieldByControlId('proofreadingRound0RpmLimit')?.props('label')).toBe('RPM限制')
    expect(fieldByControlId('proofreadingRound0BusinessRetries')?.props('label')).toBe('业务重试')
    expect(fieldByControlId('proofreadingRound0TransportRetries')?.props('label')).toBe('传输重试')
    expect(fieldByControlId('proofreadingRound0Prompt')?.props('label')).toBe('校对提示词')

    expect(fields.map(field => field.props('hint')).filter(Boolean)).toEqual(expect.arrayContaining([
      '翻译完成后自动进行AI校对',
      '使用 response_format: json_object',
      '使用流式API调用，避免超时',
    ]))
  })

  it('routes OCR setting labels and hints through typed UiField props', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/OcrSettings.vue'), 'utf8')

    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('class="ui-form-hint"')
    expect(source).not.toContain('paddleocr-vl-lang-selector label')
    expect(source).not.toContain('.ocr-settings__prompt-language-field .ui-combobox')
    expect(source).toContain('class="ocr-settings__prompt-language-combobox"')
    expect(source).toContain('class="ocr-settings__prompt-mode-hint"')

    const store = useSettingsStore()
    store.updateHybridOcr({ enabled: true })
    store.updateAiVisionOcr({ promptMode: 'paddleocr_vl' })
    const wrapper = mount(OcrSettings, globalMountOptions(pinia))
    const fields = wrapper.findAllComponents(UiField)
    const fieldByControlId = (controlId: string) =>
      fields.find(field => field.props('controlId') === controlId)

    expect(fieldByControlId('settingsOcrEngine')?.props('label')).toBe('OCR引擎')
    expect(fieldByControlId('settingsSourceLanguage')?.props('label')).toBe('源语言')
    expect(fieldByControlId('settingsSourceLanguage')?.props('hint')).toContain('识别')
    expect(fieldByControlId('settingsHybridOcrEnabled')?.props('label')).toBe('启用混合OCR')
    expect(fieldByControlId('settingsHybridOcrEnabled')?.props('hint')).toContain('textline')
    expect(fieldByControlId('settingsHybridSecondaryOcr')?.props('label')).toBe('备用OCR')
    expect(fieldByControlId('settingsHybridThreshold')?.props('label')).toBe('混合阈值')
    expect(fieldByControlId('settingsPaddleOcrVlSourceLanguage')?.props('hint')).toBe('选择图像中的源语言，用于优化 OCR 识别效果')
    expect(fieldByControlId('settingsBaiduApiKey')?.props('label')).toBe('API Key')
    expect(fieldByControlId('settingsBaiduSecretKey')?.props('label')).toBe('Secret Key')
    expect(fieldByControlId('settingsBaiduVersion')?.props('label')).toBe('识别版本')
    expect(fieldByControlId('settingsBaiduSourceLanguage')?.props('label')).toBe('源语言')
    expect(fieldByControlId('settingsAiVisionProvider')?.props('label')).toBe('服务商')
    expect(fieldByControlId('settingsAiVisionApiKey')?.props('label')).toBe('API Key')
    expect(fieldByControlId('settingsCustomAiVisionBaseUrl')?.props('label')).toBe('Base URL')
    expect(fieldByControlId('settingsAiVisionModelName')?.props('label')).toBe('模型名称')
    expect(fieldByControlId('settingsAiVisionOcrPrompt')?.props('label')).toBe('OCR提示词')
    expect(fieldByControlId('settingsAiVisionPaddleOcrVlSourceLanguage')?.props('label')).toBe('源语言')
    expect(fieldByControlId('settingsRpmAiVisionOcr')?.props('hint')).toBe('0 表示无限制')
    expect(fieldByControlId('settingsAiVisionUseStream')?.props('hint')).toBe('使用流式请求并在终端输出流式日志')
    expect(fieldByControlId('settingsMinImageSize')?.props('hint')).toContain('VLM模型通常要求图片尺寸')
  })

  it('passes stable ids and accessible names into searchable combobox fields', () => {
    const files = [
      'src/components/settings/OcrSettings.vue',
      'src/components/settings/TextStyleDefaultsSettings.vue',
      'src/components/translate/settings-sidebar/TextStyleSection.vue',
      'src/components/product/ProductBookSelector.vue',
    ]

    for (const file of files) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      const comboboxTags = source.match(/<UiCombobox[\s\S]*?>/g) ?? []
      expect(comboboxTags.length, `${file} should render searchable comboboxes`).toBeGreaterThan(0)

      for (const tag of comboboxTags) {
        expect(tag, `${file} combobox should expose a stable trigger id`).toMatch(/input-id=|:input-id=/)
        expect(tag, `${file} combobox should expose an accessible name`).toMatch(/aria-label=|:aria-label=/)
      }
    }
  })

  it('uses the shared password field for API key visibility controls', () => {
    const files = [
      'src/components/settings/HqTranslationSettings.vue',
      'src/components/settings/OcrSettings.vue',
      'src/components/settings/ProofreadingSettings.vue',
      'src/components/settings/TranslationSettings.vue',
      'src/components/translate/web-import/WebImportBasicSettingsPanel.vue',
    ]

    for (const file of files) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source, `${file} should render credentials through the shared provider field`).toContain('AiProviderCredentialFields')
      expect(source, `${file} should not keep local password toggle classes`).not.toContain('password-toggle')
      expect(source, `${file} should not keep local secure-input classes`).not.toContain('secure-input')
    }

    const credentialSource = readFileSync(
      resolve(process.cwd(), 'src/components/settings/AiProviderCredentialFields.vue'),
      'utf8',
    )
    expect(credentialSource).toContain('UiPasswordField')
  })

  it('uses product action rows for settings prompt controls instead of local reset and format skins', () => {
    const promptFiles = [
      'src/components/settings/HqTranslationSettings.vue',
      'src/components/settings/OcrSettings.vue',
      'src/components/settings/ProofreadingSettings.vue',
      'src/components/settings/TranslationSettings.vue',
    ]

    for (const file of promptFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source, `${file} should use ProductActionRow for prompt actions`).toContain('ProductActionRow')
      expect(source, `${file} should not keep local prompt format skins`).not.toContain('prompt-format-selector')
      expect(source, `${file} should not keep local reset button skins`).not.toContain('reset-btn')
      expect(source, `${file} should not keep HQ reset skins`).not.toContain('hq-reset-prompt-btn')
    }

    const savedPromptsPicker = readFileSync(
      resolve(process.cwd(), 'src/components/settings/SavedPromptsPicker.vue'),
      'utf8',
    )
    expect(savedPromptsPicker).toContain('ProductChipList')
    expect(savedPromptsPicker).not.toContain('prompt-chip')
    expect(savedPromptsPicker).not.toContain('prompts-chips-container')
    expect(savedPromptsPicker).not.toContain('picker-label')
  })

  it('uses panel textarea variants for settings prompt editors', () => {
    const promptFiles = [
      'src/components/settings/HqTranslationSettings.vue',
      'src/components/settings/OcrSettings.vue',
      'src/components/settings/ProofreadingSettings.vue',
      'src/components/settings/TranslationSettings.vue',
    ]

    for (const file of promptFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      const textareaTags = source.match(/<UiTextarea[\s\S]*?>/g) ?? []
      expect(textareaTags.length, `${file} should render prompt textareas`).toBeGreaterThan(0)
      for (const tag of textareaTags) {
        expect(tag, `${file} prompt textareas should use the product panel variant`).toContain('variant="panel"')
      }
    }
  })

  it('keeps HQ and proofreading settings on shared button primitives without owner button skins', () => {
    const files = [
      'src/components/settings/HqTranslationSettings.vue',
      'src/components/settings/ProofreadingSettings.vue',
    ]

    for (const file of files) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source, `${file} should express actions through UiButton`).toContain('<UiButton')
      expect(source, `${file} should not override shared button tokens`).not.toContain('--ui-button-')
    }

    const proofreadingSource = readFileSync(
      resolve(process.cwd(), 'src/components/settings/ProofreadingSettings.vue'),
      'utf8',
    )
    expect(proofreadingSource).toContain('variant="danger"')
    expect(proofreadingSource).toContain('class="proofreading-settings__add-round-action"')
    expect(proofreadingSource).toContain('class="proofreading-settings__round-delete-action"')
    expect(proofreadingSource).toContain('<UiIcon name="plus"')
    expect(proofreadingSource).not.toContain('+ 添加轮次')
    expect(proofreadingSource).not.toContain('class="proofreading-round"')
    expect(proofreadingSource).not.toContain('class="round-header"')
    expect(proofreadingSource).not.toContain('class="round-title"')
    expect(proofreadingSource).not.toContain('class="round-content"')
  })

  it('groups standalone settings connection tests with product action rows', () => {
    const translationSettings = readFileSync(
      resolve(process.cwd(), 'src/components/settings/TranslationSettings.vue'),
      'utf8',
    )
    const ocrSettings = readFileSync(
      resolve(process.cwd(), 'src/components/settings/OcrSettings.vue'),
      'utf8',
    )

    expect(translationSettings).toContain('aria-label="本地翻译连接测试"')
    expect(translationSettings).toContain('aria-label="云端翻译连接测试"')
    expect(translationSettings).not.toMatch(/<UiField v-show="isLocalProvider" variant="settings">\s*<UiButton/)
    expect(translationSettings).not.toMatch(/<UiField v-show="!isLocalProvider" variant="settings">\s*<UiButton/)

    expect(ocrSettings).toContain('aria-label="百度 OCR 操作"')
    expect(ocrSettings).toContain('aria-label="AI 视觉 OCR 操作"')
    expect(ocrSettings).not.toMatch(/<UiButton variant="secondary" block @click="testBaiduOcr"/)
    expect(ocrSettings).not.toMatch(/<UiButton variant="secondary" block @click="testAiVisionOcr"/)
  })

  it('routes OCR source language updates through the settings store action', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/OcrSettings.vue'), 'utf8')

    expect(source).toContain('settingsStore.setSourceLanguage')
    expect(source).not.toContain('settings.value.sourceLanguage =')
  })

  it('uses primary model-fetch emphasis in settings pages without changing embedded model pickers', () => {
    const settingsFiles = [
      'src/components/settings/TranslationSettings.vue',
      'src/components/settings/OcrSettings.vue',
      'src/components/settings/HqTranslationSettings.vue',
      'src/components/settings/ProofreadingSettings.vue',
    ]
    const insightSettingsFiles = [
      'src/components/insight/settings/VlmSettingsTab.vue',
      'src/components/insight/settings/LlmSettingsTab.vue',
      'src/components/insight/settings/EmbeddingSettingsTab.vue',
      'src/components/insight/settings/RerankerSettingsTab.vue',
      'src/components/insight/settings/ImageGenSettingsTab.vue',
    ]

    for (const file of settingsFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      const modelPickerTags = source.match(/<UiModelPicker[\s\S]*?>/g) ?? []
      expect(modelPickerTags.length, `${file} should render model picker fields`).toBeGreaterThan(0)
      for (const tag of modelPickerTags) {
        if (tag.includes(':show-fetch="false"') || tag.includes('show-fetch="false"')) {
          continue
        }
        expect(tag, `${file} should use primary model-fetch emphasis`).toContain('fetch-variant="primary"')
      }
    }

    const insightModelProviderSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/settings/InsightModelProviderSection.vue'),
      'utf8',
    )
    expect(insightModelProviderSource).toContain('<UiModelPicker')
    expect(insightModelProviderSource).toContain(':fetch-variant="fetchVariant"')

    for (const file of insightSettingsFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      const modelProviderTags = source.match(/<InsightModelProviderSection[\s\S]*?>/g) ?? []
      expect(modelProviderTags.length, `${file} should render model fields through InsightModelProviderSection`)
        .toBeGreaterThan(0)
      for (const tag of modelProviderTags) {
        if (tag.includes(':show-fetch="false"') || tag.includes('show-fetch="false"')) {
          continue
        }
        expect(tag, `${file} should use primary model-fetch emphasis`).toContain('fetch-variant="primary"')
      }
    }

    const embeddedFiles = [
      'src/components/translate/web-import/WebImportBasicSettingsPanel.vue',
      'src/components/settings/PluginAgentModal.vue',
    ]

    for (const file of embeddedFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source).not.toContain('fetch-variant="primary"')
    }
  })

  it('ignores stale HQ translation model-list responses after provider changes', async () => {
    const store = useSettingsStore()
    store.settings.hqTranslation.provider = 'siliconflow'
    store.settings.hqTranslation.apiKey = 'model-key'

    const pendingModels = createDeferred<{ models: Array<{ id: string; name: string }> }>()
    fetchModelsMock.mockReturnValueOnce(pendingModels.promise)

    const wrapper = mount(HqTranslationSettings, globalMountOptions(pinia))
    wrapper.getComponent(UiModelPicker).vm.$emit('fetch')
    expect(fetchModelsMock).toHaveBeenCalledWith('siliconflow', 'model-key', '', 'hq')

    const providerSelect = findSelectByOptionValues(wrapper, ['siliconflow', 'deepseek'])
    expect(providerSelect).toBeTruthy()
    providerSelect!.vm.$emit('change', 'deepseek')

    pendingModels.resolve({
      models: [{ id: 'stale-hq-model', name: 'Stale HQ Model' }],
    })
    await flushPromises()

    expect(mountedComboboxOptionValues(wrapper)).not.toContain('stale-hq-model')
  })

  it('ignores stale AI Vision OCR model-list responses after provider changes', async () => {
    const store = useSettingsStore()
    store.settings.ocrEngine = 'ai_vision'
    store.settings.aiVisionOcr.provider = 'siliconflow'
    store.settings.aiVisionOcr.apiKey = 'model-key'

    const pendingModels = createDeferred<{ models: Array<{ id: string; name: string }> }>()
    fetchModelsMock.mockReturnValueOnce(pendingModels.promise)

    const wrapper = mount(OcrSettings, globalMountOptions(pinia))
    wrapper.getComponent(UiModelPicker).vm.$emit('fetch')
    expect(fetchModelsMock).toHaveBeenCalledWith('siliconflow', 'model-key', '', 'ai_vision_ocr')

    const providerSelect = findSelectByOptionValues(wrapper, ['siliconflow', 'gemini'])
    expect(providerSelect).toBeTruthy()
    providerSelect!.vm.$emit('change', 'gemini')

    pendingModels.resolve({
      models: [{ id: 'stale-vision-model', name: 'Stale Vision Model' }],
    })
    await flushPromises()

    expect(mountedComboboxOptionValues(wrapper)).not.toContain('stale-vision-model')
  })

  it('ignores stale proofreading round model-list responses after round provider changes', async () => {
    const store = useSettingsStore()
    store.setProofreadingEnabled(true)
    store.addProofreadingRound({
      name: '第一轮校对',
      provider: 'siliconflow',
      apiKey: 'model-key',
      modelName: '',
      customBaseUrl: '',
      openaiOptions: {
        request: {
          forceJsonOutput: false,
        },
        execution: {
          useStream: true,
          rpmLimit: 0,
          transportRetries: 1,
          businessRetries: 1,
        },
      },
      batchSize: 3,
      prompt: '校对提示词',
    })

    const pendingModels = createDeferred<{ models: Array<{ id: string; name: string }> }>()
    fetchModelsMock.mockReturnValueOnce(pendingModels.promise)

    const wrapper = mount(ProofreadingSettings, globalMountOptions(pinia))
    wrapper.getComponent(UiModelPicker).vm.$emit('fetch')
    expect(fetchModelsMock).toHaveBeenCalledWith('siliconflow', 'model-key', '', 'proofreading_0')

    const providerSelect = findSelectByOptionValues(wrapper, ['siliconflow', 'deepseek'])
    expect(providerSelect).toBeTruthy()
    providerSelect!.vm.$emit('update:modelValue', 'deepseek')
    providerSelect!.vm.$emit('change', 'deepseek')

    pendingModels.resolve({
      models: [{ id: 'stale-proofreading-model', name: 'Stale Proofreading Model' }],
    })
    await flushPromises()

    expect(mountedComboboxOptionValues(wrapper)).not.toContain('stale-proofreading-model')
  })
})
