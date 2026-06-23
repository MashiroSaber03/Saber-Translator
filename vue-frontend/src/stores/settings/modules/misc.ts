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
  }

  /**
   * 设置检测框调试开关
   * @param show - 是否显示
   */
  function setShowDetectionDebug(show: boolean): void {
    settings.value.showDetectionDebug = show
    saveToStorage()
  }

  /**
   * 设置书架模式自动保存开关
   * @param enabled - 是否启用
   */
  function setAutoSaveInBookshelfMode(enabled: boolean): void {
    settings.value.autoSaveInBookshelfMode = enabled
    saveToStorage()
  }

  /**
   * 设置消除文字模式是否同时执行OCR
   * @param enabled - 是否启用
   */
  function setRemoveTextWithOcr(enabled: boolean): void {
    settings.value.removeTextWithOcr = enabled
    saveToStorage()
  }

  /**
   * 设置详细日志开关（全局）
   * @param enabled - 是否启用
   */
  function setEnableVerboseLogs(enabled: boolean): void {
    settings.value.enableVerboseLogs = enabled
    saveToStorage()
  }

  /**
   * 设置LAMA修复是否禁用自动缩放
   * @param disabled - 是否禁用缩放
   */
  function setLamaDisableResize(disabled: boolean): void {
    settings.value.lamaDisableResize = disabled
    saveToStorage()
  }

  return {
    // 计算属性
    textStyle,

    // 方法
    updateSettings,
    updateTextStyle,
    setPdfProcessingMethod,
    setShowDetectionDebug,
    setAutoSaveInBookshelfMode,
    setRemoveTextWithOcr,
    setEnableVerboseLogs,
    setLamaDisableResize
  }
}

