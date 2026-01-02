/**
 * 提示词管理设置模块
 * 对应设置模态窗的 "提示词管理" Tab
 * 
 * 注意：setTranslatePrompt 和 setTranslatePromptMode 在 translation.ts 模块中
 * 本模块只包含文本框提示词相关方法
 */

import { type Ref } from 'vue'
import type { TranslationSettings } from '@/types/settings'

/**
 * 创建提示词管理设置模块
 */
export function usePromptsSettings(
  settings: Ref<TranslationSettings>,
  saveToStorage: () => void
) {
  // ============================================================
  // 文本框提示词设置方法
  // ============================================================

  /**
   * 设置文本框提示词
   * @param prompt - 提示词内容
   */
  function setTextboxPrompt(prompt: string): void {
    settings.value.textboxPrompt = prompt
    saveToStorage()
  }

  /**
   * 切换文本框提示词启用状态
   * @param enabled - 是否启用
   */
  function setUseTextboxPrompt(enabled: boolean): void {
    settings.value.useTextboxPrompt = enabled
    saveToStorage()
  }

  return {
    // 方法
    setTextboxPrompt,
    setUseTextboxPrompt
  }
}
