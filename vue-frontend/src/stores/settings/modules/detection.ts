import { type Ref } from 'vue'
import type {
  TranslationSettings,
  TextDetector,
  BoxExpandSettings,
  PreciseMaskSettings
} from '@/types/settings'

export function useDetectionSettings(
  settings: Ref<TranslationSettings>,
) {
  function setTextDetector(detector: TextDetector): void {
    settings.value.textDetector = detector
  }

  function setMinTextBlockAreaPercent(percent: number): void {
    settings.value.minTextBlockAreaPercent = percent
  }

  function setEnableAuxYoloDetection(enabled: boolean): void {
    settings.value.enableAuxYoloDetection = enabled
  }

  function setAuxYoloConfThreshold(threshold: number): void {
    settings.value.auxYoloConfThreshold = threshold
  }

  function setAuxYoloOverlapThreshold(threshold: number): void {
    settings.value.auxYoloOverlapThreshold = threshold
  }

  function setEnableSaberYoloRefine(enabled: boolean): void {
    settings.value.enableSaberYoloRefine = enabled
  }

  function setSaberYoloRefineOverlapThreshold(threshold: number): void {
    settings.value.saberYoloRefineOverlapThreshold = threshold
  }

  function updateBoxExpand(updates: Partial<BoxExpandSettings>): void {
    Object.assign(settings.value.boxExpand, updates)
  }

  function updatePreciseMask(updates: Partial<PreciseMaskSettings>): void {
    Object.assign(settings.value.preciseMask, updates)
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
