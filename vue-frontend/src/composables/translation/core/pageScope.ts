import type { PipelineConfig } from './types'
import type { ImageData as AppImageData } from '@/types/image'
import { pageSelectionToPageIndexes } from '@/utils/pageSelection'

export interface PipelineImageSelection {
  image: AppImageData
  index: number
}

export function resolvePipelinePageIndexes(
  config: PipelineConfig,
  totalImages: number,
  currentIndex: number,
  failedIndices: number[],
): number[] {
  if (totalImages <= 0) {
    return []
  }

  if (config.scope === 'current') {
    return currentIndex >= 0 && currentIndex < totalImages ? [currentIndex] : []
  }

  if (config.scope === 'failed') {
    return failedIndices.filter((index) => index >= 0 && index < totalImages)
  }

  if (config.scope === 'selection' && config.pageSelection) {
    return pageSelectionToPageIndexes(config.pageSelection.pages)
      .filter((index) => index >= 0 && index < totalImages)
  }

  return Array.from({ length: totalImages }, (_, index) => index)
}

export function resolvePipelineImageSelection(
  config: PipelineConfig,
  images: readonly (AppImageData | undefined)[],
  currentIndex: number,
  failedIndices: number[],
): PipelineImageSelection[] {
  return resolvePipelinePageIndexes(config, images.length, currentIndex, failedIndices)
    .map((index) => ({ image: images[index], index }))
    .filter((item): item is PipelineImageSelection => item.image !== undefined)
}
