import type { UiColorSwatchOption } from '@/components/ui/UiColorSwatchGroup.vue'

export interface ReaderSettings {
  imageWidth: number
  imageGap: number
  bgColor: string
}

export const READER_SETTINGS_KEY = 'readerSettings'
export const DEFAULT_READER_SETTINGS: ReaderSettings = {
  imageWidth: 100,
  imageGap: 8,
  bgColor: '#1a1a2e',
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
    }
  } catch {
    return null
  }
}

export function loadReaderSettings(
  storage: Pick<Storage, 'getItem'> = localStorage
): ReaderSettings | null {
  return parseReaderSettingsPayload(storage.getItem(READER_SETTINGS_KEY))
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
