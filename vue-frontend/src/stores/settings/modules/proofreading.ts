/**
 * AI 校对设置模块
 * 对应设置模态窗的 "AI校对" Tab
 */

import { computed, type Ref } from 'vue'
import type {
  TranslationSettings,
  ProofreadingRound
} from '@/types/settings'

/**
 * 创建 AI 校对设置模块
 */
export function useProofreadingSettings(
  settings: Ref<TranslationSettings>,
  saveToStorage: () => void
) {
  // ============================================================
  // 计算属性
  // ============================================================

  /** AI校对是否启用 */
  const isProofreadingEnabled = computed(() => settings.value.proofreading.enabled)

  // ============================================================
  // AI校对设置方法
  // ============================================================

  /**
   * 设置AI校对启用状态
   * @param enabled - 是否启用
   */
  function setProofreadingEnabled(enabled: boolean): void {
    settings.value.proofreading.enabled = enabled
    saveToStorage()
    console.log(`AI校对已${enabled ? '启用' : '禁用'}`)
  }

  /**
   * 添加校对轮次
   * @param round - 校对轮次配置
   */
  function addProofreadingRound(round: ProofreadingRound): void {
    round.rpmLimit = round.openaiOptions.execution.rpmLimit
    round.maxRetries = round.openaiOptions.execution.maxRetries
    round.forceJsonOutput = round.openaiOptions.request.forceJsonOutput
    round.useStream = round.openaiOptions.execution.useStream
    settings.value.proofreading.rounds.push(round)
    saveToStorage()
    console.log(`已添加校对轮次: ${round.name}`)
  }

  /**
   * 更新校对轮次
   * @param index - 轮次索引
   * @param updates - 要更新的配置
   */
  function updateProofreadingRound(index: number, updates: Partial<ProofreadingRound>): void {
    if (index >= 0 && index < settings.value.proofreading.rounds.length) {
      const round = settings.value.proofreading.rounds[index]
      if (round) {
        Object.assign(round, updates)
        if (updates.rpmLimit !== undefined) round.openaiOptions.execution.rpmLimit = updates.rpmLimit
        if (updates.maxRetries !== undefined) round.openaiOptions.execution.maxRetries = updates.maxRetries
        if (updates.forceJsonOutput !== undefined) round.openaiOptions.request.forceJsonOutput = updates.forceJsonOutput
        if (updates.useStream !== undefined) round.openaiOptions.execution.useStream = updates.useStream
        round.rpmLimit = round.openaiOptions.execution.rpmLimit
        round.maxRetries = round.openaiOptions.execution.maxRetries
        round.forceJsonOutput = round.openaiOptions.request.forceJsonOutput
        round.useStream = round.openaiOptions.execution.useStream
        saveToStorage()
      }
    }
  }

  /**
   * 删除校对轮次
   * @param index - 轮次索引
   */
  function removeProofreadingRound(index: number): void {
    if (index >= 0 && index < settings.value.proofreading.rounds.length) {
      const removed = settings.value.proofreading.rounds.splice(index, 1)
      saveToStorage()
      console.log(`已删除校对轮次: ${removed[0]?.name}`)
    }
  }

  /**
   * 设置校对重试次数
   * @param maxRetries - 最大重试次数
   */
  function setProofreadingMaxRetries(maxRetries: number): void {
    settings.value.proofreading.maxRetries = maxRetries
    saveToStorage()
  }

  return {
    // 计算属性
    isProofreadingEnabled,

    // 方法
    setProofreadingEnabled,
    addProofreadingRound,
    updateProofreadingRound,
    removeProofreadingRound,
    setProofreadingMaxRetries
  }
}
