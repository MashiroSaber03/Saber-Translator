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
import { isProofreadingRoundId } from '../proofreadingIdentity'

export function useProofreadingSettings(
  settings: Ref<TranslationSettings>,
) {
  type ProofreadingRoundUiUpdates = Partial<Omit<ProofreadingRound, 'id'>> & OpenAiOptionsPatch
  const isProofreadingEnabled = computed(() => settings.value.proofreading.enabled)

  function setProofreadingEnabled(enabled: boolean): void {
    settings.value.proofreading.enabled = enabled
  }

  function addProofreadingRound(round: ProofreadingRound): void {
    if (!isProofreadingRoundId(round.id)) {
      throw new Error('校对轮次 ID 无效')
    }
    if (settings.value.proofreading.rounds.some(existing => existing.id === round.id)) {
      throw new Error('校对轮次 ID 重复')
    }
    settings.value.proofreading.rounds.push(round)
  }

  function updateProofreadingRound(index: number, updates: ProofreadingRoundUiUpdates): void {
    if (index >= 0 && index < settings.value.proofreading.rounds.length) {
      const round = settings.value.proofreading.rounds[index]
      if (round) {
        const safeUpdates = omitOpenAiOptionsPatchFields(updates)
        delete (safeUpdates as Partial<ProofreadingRound>).id
        Object.assign(round, safeUpdates)
        applyOpenAiOptionsPatch(round.openaiOptions, updates)
      }
    }
  }

  function removeProofreadingRound(index: number): void {
    if (index >= 0 && index < settings.value.proofreading.rounds.length) {
      settings.value.proofreading.rounds.splice(index, 1)
    }
  }

  return {
    isProofreadingEnabled,
    setProofreadingEnabled,
    addProofreadingRound,
    updateProofreadingRound,
    removeProofreadingRound
  }
}
