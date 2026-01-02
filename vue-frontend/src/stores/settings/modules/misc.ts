/**
 * 更多设置模块
 * 对应设置模态窗的 "更多" Tab
 * 包含 PDF 处理、调试设置、文字样式等杂项设置
 */

import { computed, type Ref } from 'vue'
import type {
  TranslationSettings,
  TranslationSettingsUpdates,
  TextStyleSettings,
  PdfProcessingMethod
} from '@/types/settings'

/**
 * 创建更多设置模块
 */
export function useMiscSettings(
  settings: Ref<TranslationSettings>,
  saveToStorage: () => void
) {
  // ============================================================
  // 计算属性
  // ============================================================

  /** 当前文字样式设置 */
  const textStyle = computed(() => settings.value.textStyle)

  // ============================================================
  // 通用设置更新方法
  // ============================================================

  /**
   * 更新翻译设置
   * @param updates - 要更新的设置
   */
  function updateSettings(updates: TranslationSettingsUpdates): void {
    Object.assign(settings.value, updates)
    saveToStorage()
  }

  /**
   * 更新文字样式设置
   * @param updates - 要更新的样式
   */
  function updateTextStyle(updates: Partial<TextStyleSettings>): void {
    Object.assign(settings.value.textStyle, updates)
    saveToStorage()
  }

  // ============================================================
  // PDF处理和调试设置方法
  // ============================================================

  /**
   * 设置PDF处理方式
   * @param method - 处理方式
   */
  function setPdfProcessingMethod(method: PdfProcessingMethod): void {
    settings.value.pdfProcessingMethod = method
    saveToStorage()
    console.log(`PDF处理方式已设置为: ${method}`)
  }

  /**
   * 设置检测框调试开关
   * @param show - 是否显示
   */
  function setShowDetectionDebug(show: boolean): void {
    settings.value.showDetectionDebug = show
    saveToStorage()
  }

  return {
    // 计算属性
    textStyle,

    // 方法
    updateSettings,
    updateTextStyle,
    setPdfProcessingMethod,
    setShowDetectionDebug
  }
}
