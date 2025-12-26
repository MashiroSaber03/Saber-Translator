/**
 * 翻译配置验证组合式函数
 * 用于验证翻译前的配置完整性，包括普通翻译、高质量翻译和AI校对
 * 
 * 功能：
 * - 配置完整性检查
 * - 缺失项提示
 * - 引导用户完成配置
 * - 设置按钮高亮引导动画
 */

import { ref, computed } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'
import { useToast } from '@/utils/toast'
import type { TranslationProvider, HqTranslationProvider, ProofreadingRound } from '@/types/settings'

// ============================================================
// 常量定义
// ============================================================

/** 本地存储键名：是否已关闭设置提醒 */
const DISMISS_SETUP_REMINDER_KEY = 'saber_translator_dismiss_setup_reminder'

/** 需要 API Key 的服务商列表 */
const PROVIDERS_REQUIRING_API_KEY: TranslationProvider[] = [
  'siliconflow',
  'deepseek',
  'volcano',
  'caiyun',
  'baidu_translate',
  'youdao_translate',
  'gemini',
  'custom_openai'
]

/** OCR 引擎显示名称映射 */
const OCR_ENGINE_DISPLAY_NAMES: Record<string, string> = {
  manga_ocr: 'MangaOCR',
  paddle_ocr: 'PaddleOCR',
  baidu_ocr: '百度OCR',
  ai_vision: 'AI视觉OCR'
}

/** 本地服务商（需要模型名称，但不需要 API Key） */
const LOCAL_PROVIDERS: TranslationProvider[] = ['ollama', 'sakura']

/** 需要自定义 Base URL 的服务商 */
const PROVIDERS_REQUIRING_BASE_URL: TranslationProvider[] = ['custom_openai']

/** 高质量翻译需要自定义 Base URL 的服务商 */
const HQ_PROVIDERS_REQUIRING_BASE_URL: HqTranslationProvider[] = ['custom_openai']

/** 服务商显示名称映射 */
const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
  siliconflow: 'SiliconFlow',
  deepseek: 'DeepSeek',
  volcano: '火山引擎',
  caiyun: '彩云小译',
  baidu_translate: '百度翻译',
  youdao_translate: '有道翻译',
  gemini: 'Google Gemini',
  custom_openai: '自定义 OpenAI',
  ollama: 'Ollama',
  sakura: 'Sakura'
}

// ============================================================
// 类型定义
// ============================================================

/** 验证结果接口 */
export interface ValidationResult {
  /** 验证是否通过 */
  valid: boolean
  /** 错误消息（验证失败时） */
  message: string
  /** 缺失的配置项列表 */
  missingItems?: string[]
}

/** 验证类型 */
export type ValidationType = 'normal' | 'hq' | 'proofread' | 'ocr'

/** 验证选项 */
export interface ValidationOptions {
  /** 校对轮次配置（用于校对验证） */
  proofreadingRounds?: ProofreadingRound[]
}

// ============================================================
// 组合式函数
// ============================================================

/**
 * 翻译配置验证组合式函数
 */
export function useValidation() {
  const settingsStore = useSettingsStore()
  const toast = useToast()

  // ============================================================
  // 状态
  // ============================================================

  /** 是否显示设置提醒弹窗 */
  const showSetupReminder = ref(false)

  /** 设置按钮是否正在高亮 */
  const isSettingsButtonHighlighted = ref(false)

  // ============================================================
  // 计算属性
  // ============================================================

  /** 是否已关闭设置提醒 */
  const isSetupReminderDismissed = computed(() => {
    try {
      return localStorage.getItem(DISMISS_SETUP_REMINDER_KEY) === 'true'
    } catch {
      return false
    }
  })

  // ============================================================
  // 工具函数
  // ============================================================

  /**
   * 获取服务商显示名称
   * @param provider - 服务商标识
   * @returns 服务商显示名称
   */
  function getProviderDisplayName(provider: string): string {
    return PROVIDER_DISPLAY_NAMES[provider] || provider
  }

  /**
   * 检查服务商是否需要 API Key
   * @param provider - 服务商标识
   * @returns 是否需要 API Key
   */
  function requiresApiKey(provider: TranslationProvider): boolean {
    return PROVIDERS_REQUIRING_API_KEY.includes(provider)
  }

  /**
   * 检查服务商是否为本地服务商
   * @param provider - 服务商标识
   * @returns 是否为本地服务商
   */
  function isLocalProvider(provider: TranslationProvider): boolean {
    return LOCAL_PROVIDERS.includes(provider)
  }

  /**
   * 检查服务商是否需要自定义 Base URL
   * @param provider - 服务商标识
   * @returns 是否需要自定义 Base URL
   */
  function requiresBaseUrl(provider: TranslationProvider): boolean {
    return PROVIDERS_REQUIRING_BASE_URL.includes(provider)
  }

  /**
   * 检查高质量翻译服务商是否需要自定义 Base URL
   * @param provider - 服务商标识
   * @returns 是否需要自定义 Base URL
   */
  function hqRequiresBaseUrl(provider: HqTranslationProvider): boolean {
    return HQ_PROVIDERS_REQUIRING_BASE_URL.includes(provider)
  }

  // ============================================================
  // 验证函数
  // ============================================================

  /**
   * 获取 OCR 引擎显示名称
   * @param engine - OCR 引擎标识
   * @returns OCR 引擎显示名称
   */
  function getOcrEngineDisplayName(engine: string): string {
    return OCR_ENGINE_DISPLAY_NAMES[engine] || engine
  }

  /**
   * 验证 OCR 配置
   * @returns 验证结果
   */
  function validateOcrConfig(): ValidationResult {
    const settings = settingsStore.settings
    const engine = settings.ocrEngine
    const baiduOcr = settings.baiduOcr
    const aiVisionOcr = settings.aiVisionOcr
    const missingItems: string[] = []

    // 检查 OCR 引擎是否已选择
    if (!engine) {
      return {
        valid: false,
        message: '请先在顶部 ⚙️ 设置菜单中选择 OCR 引擎',
        missingItems: ['OCR 引擎']
      }
    }

    // 检查百度 OCR 配置
    if (engine === 'baidu_ocr') {
      if (!baiduOcr?.apiKey || baiduOcr.apiKey.trim() === '') {
        missingItems.push('百度OCR 的 API Key')
      }
      if (!baiduOcr?.secretKey || baiduOcr.secretKey.trim() === '') {
        missingItems.push('百度OCR 的 Secret Key')
      }
    }

    // 检查 AI 视觉 OCR 配置
    if (engine === 'ai_vision') {
      if (!aiVisionOcr?.provider) {
        missingItems.push('AI视觉OCR 的服务商')
      }
      if (!aiVisionOcr?.apiKey || aiVisionOcr.apiKey.trim() === '') {
        missingItems.push('AI视觉OCR 的 API Key')
      }
      if (!aiVisionOcr?.modelName || aiVisionOcr.modelName.trim() === '') {
        missingItems.push('AI视觉OCR 的模型名称')
      }
      // 检查自定义服务商的 Base URL
      if (aiVisionOcr?.provider === 'custom' && (!aiVisionOcr?.customBaseUrl || aiVisionOcr.customBaseUrl.trim() === '')) {
        missingItems.push('AI视觉OCR 的自定义 Base URL')
      }
    }

    if (missingItems.length > 0) {
      return {
        valid: false,
        message: `请先在顶部 ⚙️ 设置菜单中填写 ${missingItems[0]}`,
        missingItems
      }
    }

    return { valid: true, message: '' }
  }

  /**
   * 验证普通翻译配置
   * @returns 验证结果
   */
  function validateTranslationConfig(): ValidationResult {
    const { translation } = settingsStore.settings
    const { provider, apiKey, modelName, customBaseUrl } = translation
    const missingItems: string[] = []

    // 检查服务商是否已选择
    if (!provider) {
      return {
        valid: false,
        message: '请先在顶部 ⚙️ 设置菜单中选择翻译服务商',
        missingItems: ['翻译服务商']
      }
    }

    // 检查需要 API Key 的服务商
    if (requiresApiKey(provider)) {
      if (!apiKey || apiKey.trim() === '') {
        missingItems.push(`${getProviderDisplayName(provider)} 的 API Key`)
      }
    }

    // 检查模型名称
    if (!modelName || modelName.trim() === '') {
      if (isLocalProvider(provider) || requiresApiKey(provider)) {
        missingItems.push(`${getProviderDisplayName(provider)} 的模型名称`)
      }
    }

    // 检查自定义 Base URL
    if (requiresBaseUrl(provider)) {
      if (!customBaseUrl || customBaseUrl.trim() === '') {
        missingItems.push('自定义 OpenAI 服务的 Base URL')
      }
    }

    if (missingItems.length > 0) {
      return {
        valid: false,
        message: `请先在顶部 ⚙️ 设置菜单中填写 ${missingItems[0]}`,
        missingItems
      }
    }

    return { valid: true, message: '' }
  }

  /**
   * 验证高质量翻译配置
   * @returns 验证结果
   */
  function validateHqTranslationConfig(): ValidationResult {
    const { hqTranslation } = settingsStore.settings
    const { provider, apiKey, modelName, customBaseUrl } = hqTranslation
    const missingItems: string[] = []

    // 检查服务商是否已选择
    if (!provider) {
      return {
        valid: false,
        message: '请先在顶部 ⚙️ 设置菜单中选择高质量翻译的服务商',
        missingItems: ['高质量翻译服务商']
      }
    }

    // 检查 API Key
    if (!apiKey || apiKey.trim() === '') {
      missingItems.push('高质量翻译的 API Key')
    }

    // 检查模型名称
    if (!modelName || modelName.trim() === '') {
      missingItems.push('高质量翻译的模型名称')
    }

    // 检查自定义 Base URL
    if (hqRequiresBaseUrl(provider)) {
      if (!customBaseUrl || customBaseUrl.trim() === '') {
        missingItems.push('高质量翻译的 Base URL')
      }
    }

    if (missingItems.length > 0) {
      return {
        valid: false,
        message: `请先在顶部 ⚙️ 设置菜单中填写 ${missingItems[0]}`,
        missingItems
      }
    }

    return { valid: true, message: '' }
  }

  /**
   * 验证 AI 校对配置
   * @param proofreadingRounds - 校对轮次配置（可选，默认从 store 获取）
   * @returns 验证结果
   */
  function validateProofreadingConfig(proofreadingRounds?: ProofreadingRound[]): ValidationResult {
    const rounds = proofreadingRounds || settingsStore.settings.proofreading.rounds
    const missingItems: string[] = []

    // 检查是否有校对轮次
    if (!rounds || rounds.length === 0) {
      return {
        valid: false,
        message: '请先在顶部 ⚙️ 设置菜单中添加至少一个校对轮次',
        missingItems: ['校对轮次']
      }
    }

    // 检查每个轮次的配置
    for (let i = 0; i < rounds.length; i++) {
      const round = rounds[i]
      if (!round) continue

      const roundName = round.name || `轮次${i + 1}`

      if (!round.provider) {
        missingItems.push(`校对 ${roundName} 的服务商`)
      }

      if (!round.apiKey || round.apiKey.trim() === '') {
        missingItems.push(`校对 ${roundName} 的 API Key`)
      }

      if (!round.modelName || round.modelName.trim() === '') {
        missingItems.push(`校对 ${roundName} 的模型名称`)
      }

      // 如果有缺失项，返回第一个错误
      if (missingItems.length > 0) {
        return {
          valid: false,
          message: `请先在顶部 ⚙️ 设置菜单中为 ${missingItems[0]}`,
          missingItems
        }
      }
    }

    return { valid: true, message: '' }
  }

  /**
   * 翻译前验证配置
   * 验证失败时显示错误消息并高亮设置按钮
   * @param type - 验证类型：'normal' | 'hq' | 'proofread' | 'ocr'
   * @param options - 额外选项
   * @returns 验证是否通过
   */
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
      // 显示错误消息
      toast.error(result.message)
      // 高亮设置按钮
      highlightSettingsButton()
      return false
    }

    return true
  }

  /**
   * 验证完整的翻译流程配置（OCR + 翻译）
   * @returns 验证是否通过
   */
  function validateFullTranslationConfig(): boolean {
    // 先验证 OCR 配置
    const ocrResult = validateOcrConfig()
    if (!ocrResult.valid) {
      toast.error(ocrResult.message)
      highlightSettingsButton()
      return false
    }

    // 再验证翻译配置
    const translationResult = validateTranslationConfig()
    if (!translationResult.valid) {
      toast.error(translationResult.message)
      highlightSettingsButton()
      return false
    }

    return true
  }

  // ============================================================
  // UI 交互函数
  // ============================================================

  /**
   * 高亮设置按钮以引导用户
   * 添加脉冲动画和发光效果
   */
  function highlightSettingsButton(): void {
    const settingsBtn = document.getElementById('openSettingsBtn')
    if (!settingsBtn) return

    isSettingsButtonHighlighted.value = true

    // 添加动画效果
    settingsBtn.style.animation = 'settingsBtnPulse 0.5s ease-in-out 3'
    settingsBtn.style.boxShadow = '0 0 10px var(--primary-color, #4a90d9)'

    // 3秒后移除效果
    setTimeout(() => {
      settingsBtn.style.animation = ''
      settingsBtn.style.boxShadow = ''
      isSettingsButtonHighlighted.value = false
    }, 3000)
  }

  /**
   * 检查并显示设置提醒弹窗
   * 如果用户未选择"不再显示"，则显示提醒
   */
  function checkAndShowSetupReminder(): void {
    if (isSetupReminderDismissed.value) {
      console.log('用户已选择不再显示设置提醒')
      return
    }
    showSetupReminder.value = true
  }

  /**
   * 关闭设置提醒弹窗
   * @param shouldDismiss - 是否永久关闭（不再显示）
   */
  function closeSetupReminder(shouldDismiss: boolean = false): void {
    if (shouldDismiss) {
      try {
        localStorage.setItem(DISMISS_SETUP_REMINDER_KEY, 'true')
        console.log('用户选择永久关闭设置提醒')
      } catch (error) {
        console.error('保存设置提醒状态失败:', error)
      }
    }
    showSetupReminder.value = false
  }

  /**
   * 重置"不再显示"状态
   * 用于测试或用户主动重置
   */
  function resetSetupReminderDismiss(): void {
    try {
      localStorage.removeItem(DISMISS_SETUP_REMINDER_KEY)
      console.log('设置提醒状态已重置')
    } catch (error) {
      console.error('重置设置提醒状态失败:', error)
    }
  }

  /**
   * 初始化验证器
   * 页面加载时延迟显示设置提醒弹窗
   */
  function initValidation(): void {
    setTimeout(() => {
      checkAndShowSetupReminder()
    }, 500)
  }

  // ============================================================
  // 返回
  // ============================================================

  return {
    // 状态
    showSetupReminder,
    isSettingsButtonHighlighted,

    // 计算属性
    isSetupReminderDismissed,

    // 验证函数
    validateOcrConfig,
    validateTranslationConfig,
    validateHqTranslationConfig,
    validateProofreadingConfig,
    validateBeforeTranslation,
    validateFullTranslationConfig,

    // UI 交互函数
    highlightSettingsButton,
    checkAndShowSetupReminder,
    closeSetupReminder,
    resetSetupReminderDismiss,
    initValidation,

    // 工具函数
    getProviderDisplayName,
    getOcrEngineDisplayName,
    requiresApiKey,
    isLocalProvider,
    requiresBaseUrl
  }
}

// 类型已在上方定义并导出
