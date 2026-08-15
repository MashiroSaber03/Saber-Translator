import type { HybridOcrEngine, HybridOcrSettings, OcrEngine } from '@/types/settings'

export const SUPPORTED_HYBRID_OCR_ENGINES: HybridOcrEngine[] = ['48px_ocr', 'manga_ocr']
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

export function applyHybridOcrRules(
  primaryEngine: OcrEngine,
  hybrid: HybridOcrSettings,
  options?: {
    preferRecommendedOrder?: boolean
  }
): { primaryEngine: OcrEngine; hybrid: HybridOcrSettings } {
  if (!hybrid.enabled) {
    const normalizedSecondary = isSupportedHybridOcrEngine(primaryEngine)
      ? getHybridCounterpartEngine(primaryEngine)
      : RECOMMENDED_HYBRID_SECONDARY_ENGINE

    return {
      primaryEngine,
      hybrid: {
        enabled: false,
        secondaryEngine: normalizedSecondary,
        confidenceThreshold: hybrid.confidenceThreshold,
      }
    }
  }

  if (options?.preferRecommendedOrder) {
    return {
      primaryEngine: RECOMMENDED_HYBRID_PRIMARY_ENGINE,
      hybrid: {
        enabled: true,
        secondaryEngine: RECOMMENDED_HYBRID_SECONDARY_ENGINE,
        confidenceThreshold: hybrid.confidenceThreshold,
      }
    }
  }

  let normalizedPrimary = primaryEngine
  if (!isSupportedHybridOcrEngine(normalizedPrimary)) {
    normalizedPrimary = RECOMMENDED_HYBRID_PRIMARY_ENGINE
  }

  const configuredSecondary = hybrid.secondaryEngine
  const normalizedSecondary = (
    isSupportedHybridOcrEngine(configuredSecondary)
    && configuredSecondary !== normalizedPrimary
  )
    ? configuredSecondary
    : getHybridCounterpartEngine(normalizedPrimary)

  return {
    primaryEngine: normalizedPrimary,
    hybrid: {
      enabled: true,
      secondaryEngine: normalizedSecondary,
      confidenceThreshold: hybrid.confidenceThreshold,
    }
  }
}
