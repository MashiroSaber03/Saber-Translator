import { computed, type Ref } from 'vue'
import type {
  TranslationSettings,
  TranslationSettingsUpdates,
  TextStyleSettings
} from '@/types/settings'

export function useMiscSettings(
  settings: Ref<TranslationSettings>,
) {
  const textStyle = computed(() => settings.value.textStyle)

  function updateSettings(updates: TranslationSettingsUpdates): void {
    Object.assign(settings.value, updates)
  }

  function updateTextStyle(updates: Partial<TextStyleSettings>): void {
    Object.assign(settings.value.textStyle, updates)
  }

  function setShowDetectionDebug(show: boolean): void {
    settings.value.showDetectionDebug = show
  }

  function setRemoveTextWithOcr(enabled: boolean): void {
    settings.value.removeTextWithOcr = enabled
  }

  function setEnableVerboseLogs(enabled: boolean): void {
    settings.value.enableVerboseLogs = enabled
  }

  function setLamaDisableResize(disabled: boolean): void {
    settings.value.lamaDisableResize = disabled
  }

  return {
    textStyle,
    updateSettings,
    updateTextStyle,
    setShowDetectionDebug,
    setRemoveTextWithOcr,
    setEnableVerboseLogs,
    setLamaDisableResize
  }
}

