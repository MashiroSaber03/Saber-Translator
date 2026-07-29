import { computed, getCurrentInstance, onUnmounted, ref } from 'vue'
import {
  getProviderDisplayName as getProviderDisplayNameFromManifest,
  isLocalProviderId,
  normalizeProviderId,
  providerRequiresApiKey,
  providerRequiresBaseUrl
} from '@/config/aiProviders'
import { useSettingsStore } from '@/stores/settings'
import { useToast } from '@/utils/toast'
import type { TranslationProvider, HqTranslationProvider, ProofreadingRound } from '@/types/settings'
import { isSupportedHybridOcrCombo } from '@/utils/hybridOcr'
import {
  dismissFirstTimeGuide,
  resetFirstTimeGuideDismissal,
  shouldShowFirstTimeGuide,
} from '@/components/translate/firstTimeGuideState'

const OCR_ENGINE_DISPLAY_NAMES: Record<string, string> = {
  manga_ocr: 'MangaOCR',
  paddle_ocr: 'PaddleOCR',
  paddleocr_vl: 'PaddleOCR-VL',
  baidu_ocr: '百度OCR',
  ai_vision: 'AI视觉OCR',
  '48px_ocr': '48px OCR'
}

export interface ValidationResult {
  valid: boolean
  message: string
  missingItems?: string[]
}

export type ValidationType = 'normal' | 'hq' | 'proofread' | 'ocr'

export interface ValidationOptions {
  proofreadingRounds?: ProofreadingRound[]
}

export function useValidation() {
  const settingsStore = useSettingsStore()
  const toast = useToast()

  const showSetupReminder = ref(false)
  const isSettingsButtonHighlighted = ref(false)
  let setupReminderTimer: ReturnType<typeof setTimeout> | null = null
  let highlightTimer: ReturnType<typeof setTimeout> | null = null

  const isSetupReminderDismissed = computed(() => {
    return !shouldShowFirstTimeGuide()
  })

  function getProviderDisplayName(provider: string): string {
    return getProviderDisplayNameFromManifest(provider)
  }

  function requiresApiKey(provider: TranslationProvider): boolean {
    return providerRequiresApiKey(provider)
  }

  function isLocalProvider(provider: TranslationProvider): boolean {
    return isLocalProviderId(provider)
  }

  function requiresBaseUrl(provider: TranslationProvider): boolean {
    return providerRequiresBaseUrl(provider)
  }

  function hqRequiresBaseUrl(provider: HqTranslationProvider): boolean {
    return providerRequiresBaseUrl(provider)
  }

  function getOcrEngineDisplayName(engine: string): string {
    return OCR_ENGINE_DISPLAY_NAMES[engine] || engine
  }

  function hasStoredCredential(domain: string, provider: string): boolean {
    const normalized = normalizeProviderId(provider)
    return settingsStore.hasCredential(domain, provider)
      || (normalized !== provider && settingsStore.hasCredential(domain, normalized))
  }

  function hasUsableApiKey(
    apiKey: string | undefined,
    domain: string,
    provider: string,
  ): boolean {
    return Boolean(apiKey?.trim()) || hasStoredCredential(domain, provider)
  }

  function validateOcrConfig(): ValidationResult {
    const settings = settingsStore.settings
    const engine = settings.ocrEngine
    const baiduOcr = settings.baiduOcr
    const aiVisionOcr = settings.aiVisionOcr
    const missingItems: string[] = []

    if (!engine) {
      return {
        valid: false,
        message: '请先在顶部设置菜单中选择 OCR 引擎',
        missingItems: ['OCR 引擎']
      }
    }

    const validateEngineConfig = (ocrEngine: string, prefix = '') => {
      if (ocrEngine === 'baidu_ocr') {
        const apiKey = baiduOcr?.apiKey?.trim() ?? ''
        const secretKey = baiduOcr?.secretKey?.trim() ?? ''
        const isReplacingCredential = Boolean(apiKey || secretKey)
        const hasStoredBaiduCredential = hasStoredCredential('ocr', 'baidu')
        if (!apiKey && (isReplacingCredential || !hasStoredBaiduCredential)) {
          missingItems.push(`${prefix}百度OCR 的 API Key`)
        }
        if (!secretKey && (isReplacingCredential || !hasStoredBaiduCredential)) {
          missingItems.push(`${prefix}百度OCR 的 Secret Key`)
        }
      }

      if (ocrEngine === 'ai_vision') {
        if (!aiVisionOcr?.provider) {
          missingItems.push(`${prefix}AI视觉OCR 的服务商`)
        }
        if (
          aiVisionOcr?.provider &&
          providerRequiresApiKey(normalizeProviderId(aiVisionOcr.provider)) &&
          !hasUsableApiKey(
            aiVisionOcr?.apiKey,
            'ai_vision_ocr',
            aiVisionOcr.provider,
          )
        ) {
          missingItems.push(`${prefix}AI视觉OCR 的 API Key`)
        }
        if (!aiVisionOcr?.modelName || aiVisionOcr.modelName.trim() === '') {
          missingItems.push(`${prefix}AI视觉OCR 的模型名称`)
        }
        if (
          normalizeProviderId(aiVisionOcr?.provider) === 'custom' &&
          (!aiVisionOcr?.customBaseUrl || aiVisionOcr.customBaseUrl.trim() === '')
        ) {
          missingItems.push(`${prefix}AI视觉OCR 的自定义 Base URL`)
        }
      }
    }

    validateEngineConfig(engine)

    if (settings.hybridOcr?.enabled) {
      if (!isSupportedHybridOcrCombo(engine, settings.hybridOcr.secondaryEngine)) {
        return {
          valid: false,
          message: '请先在顶部设置菜单中选择 MangaOCR / 48px OCR 组合',
          missingItems: ['混合OCR引擎组合']
        }
      }
      validateEngineConfig(settings.hybridOcr.secondaryEngine, '备用')
    }

    if (missingItems.length > 0) {
      return {
        valid: false,
        message: `请先在顶部设置菜单中填写 ${missingItems[0]}`,
        missingItems
      }
    }

    return { valid: true, message: '' }
  }

  function validateTranslationConfig(): ValidationResult {
    const { translation } = settingsStore.settings
    const { provider, apiKey, modelName, customBaseUrl } = translation
    const missingItems: string[] = []

    if (!provider) {
      return {
        valid: false,
        message: '请先在顶部设置菜单中选择翻译服务商',
        missingItems: ['翻译服务商']
      }
    }

    if (requiresApiKey(provider)) {
      if (!hasUsableApiKey(apiKey, 'translation', provider)) {
        missingItems.push(`${getProviderDisplayName(provider)} 的 API Key`)
      }
    }

    if (!modelName || modelName.trim() === '') {
      if (isLocalProvider(provider) || requiresApiKey(provider)) {
        missingItems.push(`${getProviderDisplayName(provider)} 的模型名称`)
      }
    }

    if (requiresBaseUrl(provider)) {
      if (!customBaseUrl || customBaseUrl.trim() === '') {
        missingItems.push('自定义 OpenAI 服务的 Base URL')
      }
    }

    if (missingItems.length > 0) {
      return {
        valid: false,
        message: `请先在顶部设置菜单中填写 ${missingItems[0]}`,
        missingItems
      }
    }

    return { valid: true, message: '' }
  }

  function validateHqTranslationConfig(): ValidationResult {
    const { hqTranslation } = settingsStore.settings
    const { provider, apiKey, modelName, customBaseUrl } = hqTranslation
    const missingItems: string[] = []

    if (!provider) {
      return {
        valid: false,
        message: '请先在顶部设置菜单中选择高质量翻译的服务商',
        missingItems: ['高质量翻译服务商']
      }
    }

    if (
      providerRequiresApiKey(provider)
      && !hasUsableApiKey(apiKey, 'hq', provider)
    ) {
      missingItems.push('高质量翻译的 API Key')
    }

    if (!modelName || modelName.trim() === '') {
      missingItems.push('高质量翻译的模型名称')
    }

    if (hqRequiresBaseUrl(provider)) {
      if (!customBaseUrl || customBaseUrl.trim() === '') {
        missingItems.push('高质量翻译的 Base URL')
      }
    }

    if (missingItems.length > 0) {
      return {
        valid: false,
        message: `请先在顶部设置菜单中填写 ${missingItems[0]}`,
        missingItems
      }
    }

    return { valid: true, message: '' }
  }

  function validateProofreadingConfig(proofreadingRounds?: ProofreadingRound[]): ValidationResult {
    const rounds = proofreadingRounds || settingsStore.settings.proofreading.rounds
    const missingItems: string[] = []

    if (!rounds || rounds.length === 0) {
      return {
        valid: false,
        message: '请先在顶部设置菜单中添加至少一个校对轮次',
        missingItems: ['校对轮次']
      }
    }

    for (let i = 0; i < rounds.length; i++) {
      const round = rounds[i]
      if (!round) continue

      const roundName = round.name || `轮次${i + 1}`

      if (!round.provider) {
        missingItems.push(`校对 ${roundName} 的服务商`)
      }

      if (
        providerRequiresApiKey(round.provider)
        && !hasUsableApiKey(round.apiKey, `proofreading_${i}`, round.provider)
      ) {
        missingItems.push(`校对 ${roundName} 的 API Key`)
      }

      if (!round.modelName || round.modelName.trim() === '') {
        missingItems.push(`校对 ${roundName} 的模型名称`)
      }

      if (missingItems.length > 0) {
        return {
          valid: false,
          message: `请先在顶部设置菜单中为 ${missingItems[0]}`,
          missingItems
        }
      }
    }

    return { valid: true, message: '' }
  }

  function validateBeforeTranslation(
    type: ValidationType = 'normal',
    options: ValidationOptions = {}
  ): boolean {
    let result: ValidationResult

    switch (type) {
      case 'normal':
        result = validateTranslationConfig()
        break
      case 'hq':
        result = validateHqTranslationConfig()
        break
      case 'proofread':
        result = validateProofreadingConfig(options.proofreadingRounds)
        break
      case 'ocr':
        result = validateOcrConfig()
        break
      default:
        result = validateTranslationConfig()
    }

    if (!result.valid) {
      toast.error(result.message)
      highlightSettingsButton()
      return false
    }

    return true
  }

  function validateFullTranslationConfig(): boolean {
    const ocrResult = validateOcrConfig()
    if (!ocrResult.valid) {
      toast.error(ocrResult.message)
      highlightSettingsButton()
      return false
    }

    const translationResult = validateTranslationConfig()
    if (!translationResult.valid) {
      toast.error(translationResult.message)
      highlightSettingsButton()
      return false
    }

    return true
  }

  function clearSetupReminderTimer(): void {
    if (setupReminderTimer) {
      clearTimeout(setupReminderTimer)
      setupReminderTimer = null
    }
  }

  function clearHighlightTimer(): void {
    if (highlightTimer) {
      clearTimeout(highlightTimer)
      highlightTimer = null
    }
    isSettingsButtonHighlighted.value = false
  }

  function highlightSettingsButton(): void {
    clearHighlightTimer()
    isSettingsButtonHighlighted.value = true

    highlightTimer = setTimeout(() => {
      clearHighlightTimer()
    }, 3000)
  }

  function checkAndShowSetupReminder(): void {
    if (isSetupReminderDismissed.value) {
      return
    }
    showSetupReminder.value = true
  }

  function closeSetupReminder(shouldDismiss: boolean = false): void {
    if (shouldDismiss) {
      dismissFirstTimeGuide()
    }
    showSetupReminder.value = false
  }

  function resetSetupReminderDismiss(): void {
    resetFirstTimeGuideDismissal()
  }

  function initValidation(): void {
    clearSetupReminderTimer()
    setupReminderTimer = setTimeout(() => {
      setupReminderTimer = null
      checkAndShowSetupReminder()
    }, 500)
  }

  if (getCurrentInstance()) {
    onUnmounted(() => {
      clearSetupReminderTimer()
      clearHighlightTimer()
    })
  }

  return {
    showSetupReminder,
    isSettingsButtonHighlighted,

    isSetupReminderDismissed,

    validateOcrConfig,
    validateTranslationConfig,
    validateHqTranslationConfig,
    validateProofreadingConfig,
    validateBeforeTranslation,
    validateFullTranslationConfig,

    highlightSettingsButton,
    checkAndShowSetupReminder,
    closeSetupReminder,
    resetSetupReminderDismiss,
    initValidation,

    getProviderDisplayName,
    getOcrEngineDisplayName,
    requiresApiKey,
    isLocalProvider,
    requiresBaseUrl
  }
}
