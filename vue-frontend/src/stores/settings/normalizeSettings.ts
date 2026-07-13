import {
  DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE,
  DEFAULT_HQ_TRANSLATION_MAX_RETRIES,
  DEFAULT_PROOFREADING_MAX_RETRIES,
  DEFAULT_RPM_AI_VISION_OCR,
  DEFAULT_RPM_TRANSLATION,
  DEFAULT_TRANSLATION_MAX_RETRIES,
} from '@/constants'
import type { TextDetector, TranslationSettings } from '@/types/settings'
import { normalizeHybridOcrConfig } from '@/utils/hybridOcr'
import { normalizeOpenAiOptions } from '@/utils/openaiOptions'

import { createDefaultSettings } from './defaults'

function normalizeTextDetector(detector: unknown): TextDetector {
  return detector === 'ctd' || detector === 'yolo' || detector === 'default'
    ? detector
    : 'default'
}

function numberOrFallback(value: unknown, fallback: number): number {
  if (value === undefined || value === null || value === '') return fallback
  const parsed = Number(value)
  return Number.isNaN(parsed) ? fallback : parsed
}

export function normalizeSettings(settings: TranslationSettings): void {
  const defaults = createDefaultSettings()
  settings.textDetector = normalizeTextDetector(settings.textDetector)

  settings.boxExpand.ratio = numberOrFallback(settings.boxExpand.ratio, defaults.boxExpand.ratio)
  settings.boxExpand.top = numberOrFallback(settings.boxExpand.top, defaults.boxExpand.top)
  settings.boxExpand.bottom = numberOrFallback(settings.boxExpand.bottom, defaults.boxExpand.bottom)
  settings.boxExpand.left = numberOrFallback(settings.boxExpand.left, defaults.boxExpand.left)
  settings.boxExpand.right = numberOrFallback(settings.boxExpand.right, defaults.boxExpand.right)

  settings.preciseMask.dilateSize = numberOrFallback(
    settings.preciseMask.dilateSize,
    defaults.preciseMask.dilateSize,
  )
  settings.preciseMask.boxExpandRatio = numberOrFallback(
    settings.preciseMask.boxExpandRatio,
    defaults.preciseMask.boxExpandRatio,
  )

  settings.saberYoloRefineOverlapThreshold = numberOrFallback(
    settings.saberYoloRefineOverlapThreshold,
    defaults.saberYoloRefineOverlapThreshold,
  )
  settings.minTextBlockAreaPercent = Math.max(
    0,
    numberOrFallback(settings.minTextBlockAreaPercent, defaults.minTextBlockAreaPercent),
  )
  settings.enableAuxYoloDetection = Boolean(settings.enableAuxYoloDetection)
  settings.auxYoloConfThreshold = numberOrFallback(
    settings.auxYoloConfThreshold,
    defaults.auxYoloConfThreshold,
  )
  settings.auxYoloOverlapThreshold = numberOrFallback(
    settings.auxYoloOverlapThreshold,
    defaults.auxYoloOverlapThreshold,
  )

  settings.translation.openaiOptions = normalizeOpenAiOptions(settings.translation.openaiOptions, {
    execution: {
      useStream: true,
      rpmLimit: DEFAULT_RPM_TRANSLATION,
      transportRetries: 1,
      businessRetries: DEFAULT_TRANSLATION_MAX_RETRIES,
    },
  })
  settings.hqTranslation.batchSize = numberOrFallback(
    settings.hqTranslation.batchSize,
    defaults.hqTranslation.batchSize,
  )
  settings.hqTranslation.openaiOptions = normalizeOpenAiOptions(settings.hqTranslation.openaiOptions, {
    execution: {
      useStream: true,
      rpmLimit: 7,
      transportRetries: 3,
      businessRetries: DEFAULT_HQ_TRANSLATION_MAX_RETRIES,
    },
  })
  settings.pluginAgent.openaiOptions = normalizeOpenAiOptions(settings.pluginAgent.openaiOptions, {
    execution: {
      useStream: true,
      rpmLimit: 0,
      transportRetries: 10,
      businessRetries: 10,
    },
  })
  settings.aiVisionOcr.openaiOptions = normalizeOpenAiOptions(settings.aiVisionOcr.openaiOptions, {
    execution: {
      useStream: false,
      rpmLimit: DEFAULT_RPM_AI_VISION_OCR,
      transportRetries: 1,
      businessRetries: DEFAULT_TRANSLATION_MAX_RETRIES,
    },
  })
  settings.aiVisionOcr.openaiOptions.request.forceJsonOutput = settings.aiVisionOcr.promptMode === 'json'
  settings.aiVisionOcr.minImageSize = numberOrFallback(
    settings.aiVisionOcr.minImageSize,
    DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE,
  )

  const normalizedHybrid = normalizeHybridOcrConfig(
    settings.ocrEngine,
    {
      ...settings.hybridOcr,
      enabled: Boolean(settings.hybridOcr.enabled),
    },
    { preferRecommendedOrder: Boolean(settings.hybridOcr.enabled) },
  )
  settings.ocrEngine = normalizedHybrid.primaryEngine
  settings.hybridOcr = normalizedHybrid.hybrid

  settings.proofreading.maxRetries = numberOrFallback(
    settings.proofreading.maxRetries,
    DEFAULT_PROOFREADING_MAX_RETRIES,
  )
  settings.proofreading.rounds = settings.proofreading.rounds.map(round => ({
    ...round,
    openaiOptions: normalizeOpenAiOptions(round.openaiOptions, {
      execution: {
        useStream: true,
        rpmLimit: 7,
        transportRetries: 1,
        businessRetries: DEFAULT_HQ_TRANSLATION_MAX_RETRIES,
      },
    }),
  }))
  settings.settingsSchemaVersion = 3
}
