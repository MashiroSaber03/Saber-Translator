import type { PublicModelKey } from '../api/v2/auth'
import type { UiSelectOption } from '../components/ui/selectTypes'

export const OCR_MODEL_KEYS: Partial<Record<string, PublicModelKey>> = {
  manga_ocr: 'manga_ocr', '48px_ocr': 'ocr_48px',
  paddle_ocr: 'paddle_ocr', paddleocr_vl: 'paddleocr_vl',
}
export const DETECTOR_MODEL_KEYS: Partial<Record<string, PublicModelKey>> = {
  default: 'detector_default', ctd: 'detector_ctd', yolo: 'detector_yolo',
}
export const INPAINT_MODEL_KEYS: Partial<Record<string, PublicModelKey>> = {
  lama_mpe: 'lama_mpe', litelama: 'litelama', lama_manga: 'lama_manga',
}

export function localModelInstalled(model: PublicModelKey, installed: readonly string[]): boolean {
  // CTD/YOLO provide boxes; their precise masks still come from Default.
  if (model === 'detector_ctd' || model === 'detector_yolo') {
    return installed.includes(model) && installed.includes('detector_default')
  }
  return installed.includes(model === 'aux_ysg_yolo' ? 'detector_yolo' : model)
}

export function installedModelOptions<T extends UiSelectOption>(
  options: T[], keys: Partial<Record<string, PublicModelKey>>, installed: readonly string[],
): T[] {
  return options.filter(option => {
    const model = keys[String(option.value)]
    return !model || localModelInstalled(model, installed)
  })
}
