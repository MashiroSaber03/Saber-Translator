import type { HybridOcrEngine, HybridOcrSettings, OcrEngine } from '@/types/settings'

export const SUPPORTED_HYBRID_OCR_ENGINES: HybridOcrEngine[] = ['48px_ocr', 'manga_ocr']
export const DEFAULT_HYBRID_OCR_THRESHOLD = 0.2
export const RECOMMENDED_HYBRID_PRIMARY_ENGINE: HybridOcrEngine = '48px_ocr'
export const RECOMMENDED_HYBRID_SECONDARY_ENGINE: HybridOcrEngine = 'manga_ocr'

export function isSupportedHybridOcrEngine(engine: unknown): engine is HybridOcrEngine {
  return engine === '48px_ocr' || engine === 'manga_ocr'
}

export function isSupportedHybridOcrCombo(
  primaryEngine: unknown,
  secondaryEngine: unknown
): primaryEngine is HybridOcrEngine {
  return (
    isSupportedHybridOcrEngine(primaryEngine) &&
    isSupportedHybridOcrEngine(secondaryEngine) &&
    primaryEngine !== secondaryEngine
  )
}

export function getHybridCounterpartEngine(primaryEngine: HybridOcrEngine): HybridOcrEngine {
  return primaryEngine === '48px_ocr' ? 'manga_ocr' : '48px_ocr'
}

function normalizeHybridThreshold(value: unknown): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) {
    return DEFAULT_HYBRID_OCR_THRESHOLD
  }
  return Math.max(0, Math.min(1, parsed))
}

export function normalizeHybridOcrConfig(
  primaryEngine: OcrEngine,
  hybrid: HybridOcrSettings | (Partial<HybridOcrSettings> & Record<string, unknown>),
  options?: {
    preferRecommendedOrder?: boolean
  }
): { primaryEngine: OcrEngine; hybrid: HybridOcrSettings } {
  const legacyHybrid = hybrid as Record<string, unknown>
  const confidenceThreshold = normalizeHybridThreshold(
    legacyHybrid.threshold48px
      ?? legacyHybrid.thresholdMangaOcr
      ?? legacyHybrid.thresholdPaddleOcr
      ?? hybrid.confidenceThreshold
  )

  if (!hybrid.enabled) {
    const normalizedSecondary = isSupportedHybridOcrEngine(primaryEngine)
      ? getHybridCounterpartEngine(primaryEngine)
      : RECOMMENDED_HYBRID_SECONDARY_ENGINE

    return {
      primaryEngine,
      hybrid: {
        enabled: false,
        secondaryEngine: normalizedSecondary,
        confidenceThreshold
      }
    }
  }

  if (options?.preferRecommendedOrder && !isSupportedHybridOcrCombo(primaryEngine, hybrid.secondaryEngine)) {
    return {
      primaryEngine: RECOMMENDED_HYBRID_PRIMARY_ENGINE,
      hybrid: {
        enabled: true,
        secondaryEngine: RECOMMENDED_HYBRID_SECONDARY_ENGINE,
        confidenceThreshold
      }
    }
  }

  let normalizedPrimary = primaryEngine
  if (!isSupportedHybridOcrEngine(normalizedPrimary)) {
    normalizedPrimary = RECOMMENDED_HYBRID_PRIMARY_ENGINE
  }

  let normalizedSecondary = hybrid.secondaryEngine
  if (!isSupportedHybridOcrCombo(normalizedPrimary, normalizedSecondary)) {
    normalizedSecondary = getHybridCounterpartEngine(normalizedPrimary)
  }

  return {
    primaryEngine: normalizedPrimary,
    hybrid: {
      enabled: true,
      secondaryEngine: normalizedSecondary,
      confidenceThreshold
    }
  }
}
