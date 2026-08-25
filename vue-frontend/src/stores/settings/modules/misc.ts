import { computed, type Ref } from 'vue'
import type {
  ParallelSettings,
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

  function updateParallel(updates: Partial<ParallelSettings>): void {
    Object.assign(settings.value.parallel, updates)
  }

  function setRemoveTextWithOcr(enabled: boolean): void {
    settings.value.removeTextWithOcr = enabled
  }

  function setCompressVisionImages(enabled: boolean): void {
    settings.value.compressVisionImages = enabled
  }

  function setLamaDisableResize(disabled: boolean): void {
    settings.value.lamaDisableResize = disabled
  }

  return {
    textStyle,
    updateSettings,
    updateTextStyle,
    updateParallel,
    setShowDetectionDebug,
    setRemoveTextWithOcr,
    setCompressVisionImages,
    setLamaDisableResize
  }
}

