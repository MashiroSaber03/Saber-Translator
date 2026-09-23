import type { UiColorSwatchOption } from '@/components/ui/UiColorSwatchGroup.vue'

export interface ReaderSettings {
  imageWidth: number
  imageGap: number
  bgColor: string
  layout: ReaderLayout
  fits: Record<ReaderLayout, ReaderFit>
  direction: 'ltr' | 'rtl'
  progress: 'normal' | 'pages' | 'hidden'
}

export type ReaderLayout = 'single' | 'double' | 'vertical' | 'horizontal'
export type ReaderFit = 'screen' | 'width' | 'height' | 'original'
export const READER_LAYOUTS: { value: ReaderLayout; label: string }[] = [
  { value: 'single', label: '单页' },
  { value: 'double', label: '双页' },
  { value: 'vertical', label: '纵向连续' },
  { value: 'horizontal', label: '横向连续' },
]
export const READER_FITS: { value: ReaderFit; label: string }[] = [
  { value: 'screen', label: '适应屏幕' },
  { value: 'width', label: '适应宽度' },
  { value: 'height', label: '适应高度' },
  { value: 'original', label: '原始尺寸' },
]

export const READER_SETTINGS_KEY = 'readerSettings'
export const DEFAULT_READER_SETTINGS: ReaderSettings = {
  imageWidth: 100,
  imageGap: 8,
  bgColor: '#1a1a2e',
  layout: 'vertical',
  fits: { single: 'screen', double: 'screen', vertical: 'width', horizontal: 'height' },
  direction: 'ltr',
  progress: 'normal',
}

export const READER_BG_COLOR_PRESETS: UiColorSwatchOption[] = [
  { value: '#1a1a2e', label: '深蓝' },
  { value: '#ffffff', label: '白色' },
  { value: '#f5f5dc', label: '米色' },
  { value: '#2d2d2d', label: '深灰' },
]

const readerBgColorValues = new Set(READER_BG_COLOR_PRESETS.map(preset => preset.value))

function isNumberInRange(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max
}

function isReaderSettings(value: unknown): value is ReaderSettings {
  if (!value || typeof value !== 'object') return false

  const candidate = value as Partial<ReaderSettings>
  return (
    isNumberInRange(candidate.imageWidth, 50, 100) &&
    isNumberInRange(candidate.imageGap, 0, 50) &&
    typeof candidate.bgColor === 'string' &&
    readerBgColorValues.has(candidate.bgColor)
  )
}

export function parseReaderSettingsPayload(payload: string | null): ReaderSettings | null {
  if (!payload) return null
  try {
    const parsed: unknown = JSON.parse(payload)
    if (!isReaderSettings(parsed)) return null
    return {
      imageWidth: parsed.imageWidth,
      imageGap: parsed.imageGap,
      bgColor: parsed.bgColor,
      layout: READER_LAYOUTS.some(item => item.value === parsed.layout)
        ? parsed.layout
        : 'vertical',
      fits: Object.fromEntries(
        READER_LAYOUTS.map(({ value }) => [
          value,
          READER_FITS.some(item => item.value === parsed.fits?.[value])
            ? parsed.fits[value]
            : DEFAULT_READER_SETTINGS.fits[value],
        ])
      ) as Record<ReaderLayout, ReaderFit>,
      direction: parsed.direction === 'rtl' ? 'rtl' : 'ltr',
      progress:
        parsed.progress === 'pages' || parsed.progress === 'hidden' ? parsed.progress : 'normal',
    }
  } catch {
    return null
  }
}

export function loadReaderSettings(
  storage: Pick<Storage, 'getItem'> = localStorage
): ReaderSettings | null {
  try {
    return parseReaderSettingsPayload(storage.getItem(READER_SETTINGS_KEY))
  } catch {
    return null
  }
}

export function saveReaderSettings(
  settings: ReaderSettings,
  storage: Pick<Storage, 'setItem'> = localStorage
): boolean {
  try {
    storage.setItem(READER_SETTINGS_KEY, JSON.stringify(settings))
    return true
  } catch {
    return false
  }
}
