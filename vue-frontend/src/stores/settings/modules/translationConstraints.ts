/**
 * 术语表 / 禁翻表设置模块
 */

import { computed, type Ref } from 'vue'
import type { TranslationSettings } from '@/types/settings'
import type {
  GlossaryEntry,
  GlossarySettings,
  NonTranslateEntry,
  NonTranslateSettings,
} from '@/types/translationConstraints'

export function useTranslationConstraintsSettings(
  settings: Ref<TranslationSettings>,
  saveToStorage: () => void,
) {
  const glossary = computed(() => settings.value.glossary)
  const nonTranslate = computed(() => settings.value.nonTranslate)

  function setGlossaryEnabled(enabled: boolean): void {
    settings.value.glossary.enabled = enabled
    saveToStorage()
  }

  function setGlossaryEntries(entries: GlossaryEntry[]): void {
    settings.value.glossary.entries = entries
    saveToStorage()
  }

  function updateGlossary(updates: Partial<GlossarySettings>): void {
    settings.value.glossary = {
      ...settings.value.glossary,
      ...updates,
    }
    saveToStorage()
  }

  function setNonTranslateEnabled(enabled: boolean): void {
    settings.value.nonTranslate.enabled = enabled
    saveToStorage()
  }

  function setNonTranslateEntries(entries: NonTranslateEntry[]): void {
    settings.value.nonTranslate.entries = entries
    saveToStorage()
  }

  function updateNonTranslate(updates: Partial<NonTranslateSettings>): void {
    settings.value.nonTranslate = {
      ...settings.value.nonTranslate,
      ...updates,
    }
    saveToStorage()
  }

  return {
    glossary,
    nonTranslate,
    setGlossaryEnabled,
    setGlossaryEntries,
    updateGlossary,
    setNonTranslateEnabled,
    setNonTranslateEntries,
    updateNonTranslate,
  }
}
