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
) {
  type ProofreadingRoundUiUpdates = Partial<ProofreadingRound> & OpenAiOptionsPatch
  const isProofreadingEnabled = computed(() => settings.value.proofreading.enabled)

  function setProofreadingEnabled(enabled: boolean): void {
    settings.value.proofreading.enabled = enabled
  }

  function addProofreadingRound(round: ProofreadingRound): void {
    settings.value.proofreading.rounds.push(round)
  }

  function updateProofreadingRound(index: number, updates: ProofreadingRoundUiUpdates): void {
    if (index >= 0 && index < settings.value.proofreading.rounds.length) {
      const round = settings.value.proofreading.rounds[index]
      if (round) {
        Object.assign(round, omitOpenAiOptionsPatchFields(updates))
        applyOpenAiOptionsPatch(round.openaiOptions, updates)
      }
    }
  }

  function removeProofreadingRound(index: number): void {
    if (index >= 0 && index < settings.value.proofreading.rounds.length) {
      settings.value.proofreading.rounds.splice(index, 1)
    }
  }

  function setProofreadingMaxRetries(maxRetries: number): void {
    settings.value.proofreading.maxRetries = maxRetries
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
