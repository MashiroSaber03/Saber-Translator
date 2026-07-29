import { type Ref } from 'vue'
import type { TranslationSettings } from '@/types/settings'

export function usePromptsSettings(
  settings: Ref<TranslationSettings>,
) {
  function setTextboxPrompt(prompt: string): void {
    settings.value.textboxPrompt = prompt
  }

  function setUseTextboxPrompt(enabled: boolean): void {
    settings.value.useTextboxPrompt = enabled
  }

  return {
    setTextboxPrompt,
    setUseTextboxPrompt
  }
}
