import { computed, type Ref } from 'vue'
import type {
  TranslationSettings,
  ProofreadingRound
} from '@/types/settings'
import {
  applyOpenAiOptionsPatch,
  omitOpenAiOptionsPatchFields,
  type OpenAiOptionsPatch,
} from '@/utils/openaiOptions'

export function useProofreadingSettings(
  settings: Ref<TranslationSettings>,
  saveToStorage: () => void
) {
  type ProofreadingRoundUiUpdates = Partial<ProofreadingRound> & OpenAiOptionsPatch
  const isProofreadingEnabled = computed(() => settings.value.proofreading.enabled)

  function setProofreadingEnabled(enabled: boolean): void {
    settings.value.proofreading.enabled = enabled
    saveToStorage()
  }

  function addProofreadingRound(round: ProofreadingRound): void {
    settings.value.proofreading.rounds.push(round)
    saveToStorage()
  }

  function updateProofreadingRound(index: number, updates: ProofreadingRoundUiUpdates): void {
    if (index >= 0 && index < settings.value.proofreading.rounds.length) {
      const round = settings.value.proofreading.rounds[index]
      if (round) {
        Object.assign(round, omitOpenAiOptionsPatchFields(updates))
        applyOpenAiOptionsPatch(round.openaiOptions, updates)
        saveToStorage()
      }
    }
  }

  function removeProofreadingRound(index: number): void {
    if (index >= 0 && index < settings.value.proofreading.rounds.length) {
      settings.value.proofreading.rounds.splice(index, 1)
      saveToStorage()
    }
  }

  function setProofreadingMaxRetries(maxRetries: number): void {
    settings.value.proofreading.maxRetries = maxRetries
    saveToStorage()
  }

  return {
    isProofreadingEnabled,
    setProofreadingEnabled,
    addProofreadingRound,
    updateProofreadingRound,
    removeProofreadingRound,
    setProofreadingMaxRetries
  }
}
