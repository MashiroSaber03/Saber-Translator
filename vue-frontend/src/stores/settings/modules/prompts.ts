import { type Ref } from 'vue'
import type { TranslationSettings } from '@/types/settings'

export function usePromptsSettings(
  settings: Ref<TranslationSettings>,
  saveToStorage: () => void
) {
  function setTextboxPrompt(prompt: string): void {
    settings.value.textboxPrompt = prompt
    saveToStorage()
  }

  function setUseTextboxPrompt(enabled: boolean): void {
    settings.value.useTextboxPrompt = enabled
    saveToStorage()
  }

  return {
    setTextboxPrompt,
    setUseTextboxPrompt
  }
}
