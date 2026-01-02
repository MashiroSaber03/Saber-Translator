/**
 * 检测设置模块
 * 对应设置模态窗的 "检测设置" Tab
 */

import { type Ref } from 'vue'
import type {
  TranslationSettings,
  TextDetector,
  BoxExpandSettings,
  PreciseMaskSettings
} from '@/types/settings'

/**
 * 创建检测设置模块
 */
export function useDetectionSettings(
  settings: Ref<TranslationSettings>,
  saveToStorage: () => void
) {
  // ============================================================
  // 检测设置方法
  // ============================================================

  /**
   * 设置文本检测器
   * @param detector - 检测器类型
   */
  function setTextDetector(detector: TextDetector): void {
    settings.value.textDetector = detector
    saveToStorage()
    console.log(`文本检测器已设置为: ${detector}`)
  }

  /**
   * 更新文本框扩展参数
   * @param updates - 要更新的参数
   */
  function updateBoxExpand(updates: Partial<BoxExpandSettings>): void {
    Object.assign(settings.value.boxExpand, updates)
    saveToStorage()
  }

  /**
   * 更新精确文字掩膜设置
   * @param updates - 要更新的设置
   */
  function updatePreciseMask(updates: Partial<PreciseMaskSettings>): void {
    Object.assign(settings.value.preciseMask, updates)
    saveToStorage()
  }

  return {
    // 方法
    setTextDetector,
    updateBoxExpand,
    updatePreciseMask
  }
}
