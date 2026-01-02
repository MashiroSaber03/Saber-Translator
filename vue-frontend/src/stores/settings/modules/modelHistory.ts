/**
 * 模型历史记录模块
 * 管理模型使用历史和自定义字号预设
 */

import { ref } from 'vue'
import { STORAGE_KEY_MODEL_HISTORY } from '@/constants'

/**
 * 创建模型历史记录模块
 */
export function useModelHistory() {
  // ============================================================
  // 状态定义
  // ============================================================

  /** 模型使用历史记录（按服务商分组） */
  const modelHistory = ref<Record<string, string[]>>({})

  /** 自定义字号预设 */
  const customFontPresets = ref<number[]>([])

  // ============================================================
  // 模型历史记录方法
  // ============================================================

  /**
   * 添加模型到历史记录
   * @param provider - 服务商名称
   * @param modelName - 模型名称
   */
  function addModelToHistory(provider: string, modelName: string): void {
    if (!provider || !modelName) return

    if (!modelHistory.value[provider]) {
      modelHistory.value[provider] = []
    }

    // 如果已存在，先移除
    const index = modelHistory.value[provider].indexOf(modelName)
    if (index !== -1) {
      modelHistory.value[provider].splice(index, 1)
    }

    // 添加到开头（最近使用的在前）
    modelHistory.value[provider].unshift(modelName)

    // 限制历史记录数量（最多保留20个）
    if (modelHistory.value[provider].length > 20) {
      modelHistory.value[provider] = modelHistory.value[provider].slice(0, 20)
    }

    saveModelHistoryToStorage()
    console.log(`[Settings] 添加模型到历史: ${provider} -> ${modelName}`)
  }

  /**
   * 获取服务商的模型历史记录
   * @param provider - 服务商名称
   * @returns 模型名称列表
   */
  function getModelHistory(provider: string): string[] {
    return modelHistory.value[provider] || []
  }

  /**
   * 清除服务商的模型历史记录
   * @param provider - 服务商名称
   */
  function clearModelHistory(provider: string): void {
    if (modelHistory.value[provider]) {
      delete modelHistory.value[provider]
      saveModelHistoryToStorage()
      console.log(`[Settings] 清除模型历史: ${provider}`)
    }
  }

  // ============================================================
  // 自定义字号预设方法
  // ============================================================

  /**
   * 添加自定义字号预设
   * @param fontSize - 字号
   */
  function addCustomFontPreset(fontSize: number): void {
    if (!customFontPresets.value.includes(fontSize)) {
      customFontPresets.value.push(fontSize)
      customFontPresets.value.sort((a, b) => a - b)
      saveCustomFontPresetsToStorage()
      console.log(`已添加自定义字号预设: ${fontSize}`)
    }
  }

  /**
   * 删除自定义字号预设
   * @param fontSize - 字号
   */
  function removeCustomFontPreset(fontSize: number): void {
    const index = customFontPresets.value.indexOf(fontSize)
    if (index !== -1) {
      customFontPresets.value.splice(index, 1)
      saveCustomFontPresetsToStorage()
      console.log(`已删除自定义字号预设: ${fontSize}`)
    }
  }

  // ============================================================
  // localStorage 持久化方法
  // ============================================================

  /**
   * 保存模型历史记录到 localStorage
   */
  function saveModelHistoryToStorage(): void {
    try {
      const data = JSON.stringify(modelHistory.value)
      localStorage.setItem(STORAGE_KEY_MODEL_HISTORY, data)
    } catch (error) {
      console.error('保存模型历史记录失败:', error)
    }
  }

  /**
   * 从 localStorage 加载模型历史记录
   */
  function loadModelHistoryFromStorage(): void {
    try {
      const data = localStorage.getItem(STORAGE_KEY_MODEL_HISTORY)
      if (data) {
        modelHistory.value = JSON.parse(data)
        console.log('已从 localStorage 加载模型历史记录')
      }
    } catch (error) {
      console.error('加载模型历史记录失败:', error)
    }
  }

  /**
   * 保存自定义字号预设到 localStorage
   */
  function saveCustomFontPresetsToStorage(): void {
    try {
      const data = JSON.stringify(customFontPresets.value)
      localStorage.setItem('customFontSizePresets', data)
    } catch (error) {
      console.error('保存自定义字号预设失败:', error)
    }
  }

  /**
   * 从 localStorage 加载自定义字号预设
   */
  function loadCustomFontPresetsFromStorage(): void {
    try {
      const data = localStorage.getItem('customFontSizePresets')
      if (data) {
        customFontPresets.value = JSON.parse(data)
      }
    } catch (error) {
      console.error('加载自定义字号预设失败:', error)
    }
  }

  return {
    // 状态
    modelHistory,
    customFontPresets,

    // 模型历史方法
    addModelToHistory,
    getModelHistory,
    clearModelHistory,

    // 自定义字号预设方法
    addCustomFontPreset,
    removeCustomFontPreset,

    // 持久化方法
    saveModelHistoryToStorage,
    loadModelHistoryFromStorage,
    saveCustomFontPresetsToStorage,
    loadCustomFontPresetsFromStorage
  }
}
