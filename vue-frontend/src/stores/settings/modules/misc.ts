import { computed, type Ref } from 'vue'
import type {
  TranslationSettings,
  TranslationSettingsUpdates,
  TextStyleSettings,
  PdfProcessingMethod
} from '@/types/settings'

export function useMiscSettings(
  settings: Ref<TranslationSettings>,
  saveToStorage: () => void
) {
  const textStyle = computed(() => settings.value.textStyle)

  function updateSettings(updates: TranslationSettingsUpdates): void {
    Object.assign(settings.value, updates)
    saveToStorage()
  }

  function updateTextStyle(updates: Partial<TextStyleSettings>): void {
    Object.assign(settings.value.textStyle, updates)
    saveToStorage()
  }

  function setPdfProcessingMethod(method: PdfProcessingMethod): void {
    settings.value.pdfProcessingMethod = method
    saveToStorage()
  }

  function setShowDetectionDebug(show: boolean): void {
    settings.value.showDetectionDebug = show
    saveToStorage()
  }

  function setAutoSaveInBookshelfMode(enabled: boolean): void {
    settings.value.autoSaveInBookshelfMode = enabled
    saveToStorage()
  }

  function setRemoveTextWithOcr(enabled: boolean): void {
    settings.value.removeTextWithOcr = enabled
    saveToStorage()
  }

  function setEnableVerboseLogs(enabled: boolean): void {
    settings.value.enableVerboseLogs = enabled
    saveToStorage()
  }

  function setLamaDisableResize(disabled: boolean): void {
    settings.value.lamaDisableResize = disabled
    saveToStorage()
  }

  return {
    textStyle,
    updateSettings,
    updateTextStyle,
    setPdfProcessingMethod,
    setShowDetectionDebug,
    setAutoSaveInBookshelfMode,
    setRemoveTextWithOcr,
    setEnableVerboseLogs,
    setLamaDisableResize
  }
}

