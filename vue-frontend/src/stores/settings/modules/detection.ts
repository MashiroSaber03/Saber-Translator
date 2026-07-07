import { type Ref } from 'vue'
import type {
  TranslationSettings,
  TextDetector,
  BoxExpandSettings,
  PreciseMaskSettings
} from '@/types/settings'

export function useDetectionSettings(
  settings: Ref<TranslationSettings>,
  saveToStorage: () => void
) {
  function setTextDetector(detector: TextDetector): void {
    settings.value.textDetector = detector
    saveToStorage()
  }

  function setMinTextBlockAreaPercent(percent: number): void {
    settings.value.minTextBlockAreaPercent = percent
    saveToStorage()
  }

  function setEnableAuxYoloDetection(enabled: boolean): void {
    settings.value.enableAuxYoloDetection = enabled
    saveToStorage()
  }

  function setAuxYoloConfThreshold(threshold: number): void {
    settings.value.auxYoloConfThreshold = threshold
    saveToStorage()
  }

  function setAuxYoloOverlapThreshold(threshold: number): void {
    settings.value.auxYoloOverlapThreshold = threshold
    saveToStorage()
  }

  function setEnableSaberYoloRefine(enabled: boolean): void {
    settings.value.enableSaberYoloRefine = enabled
    saveToStorage()
  }

  function setSaberYoloRefineOverlapThreshold(threshold: number): void {
    settings.value.saberYoloRefineOverlapThreshold = threshold
    saveToStorage()
  }

  function updateBoxExpand(updates: Partial<BoxExpandSettings>): void {
    Object.assign(settings.value.boxExpand, updates)
    saveToStorage()
  }

  function updatePreciseMask(updates: Partial<PreciseMaskSettings>): void {
    Object.assign(settings.value.preciseMask, updates)
    saveToStorage()
  }

  return {
    setTextDetector,
    setMinTextBlockAreaPercent,
    setEnableAuxYoloDetection,
    setAuxYoloConfThreshold,
    setAuxYoloOverlapThreshold,
    setEnableSaberYoloRefine,
    setSaberYoloRefineOverlapThreshold,
    updateBoxExpand,
    updatePreciseMask
  }
}
