export type RgbArray = [number, number, number]

const HEX_COLOR_PATTERN = /^#?[0-9A-Fa-f]{6}$/
const DARK_LUMINANCE_THRESHOLD = 128

function clampColorChannel(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)))
}

function channelToHex(value: number): string {
  return clampColorChannel(value).toString(16).padStart(2, '0')
}

export function rgbArrayToHex(rgb: RgbArray): string {
  return `#${channelToHex(rgb[0])}${channelToHex(rgb[1])}${channelToHex(rgb[2])}`
}

export function hexToRgbArray(hex: string): RgbArray {
  const cleaned = hex.replace('#', '')
  return [
    Number.parseInt(cleaned.slice(0, 2), 16),
    Number.parseInt(cleaned.slice(2, 4), 16),
    Number.parseInt(cleaned.slice(4, 6), 16),
  ]
}

export function isValidHex(hex: string): boolean {
  return HEX_COLOR_PATTERN.test(hex)
}

export function normalizeHex(hex: string): string {
  return `#${hex.replace('#', '').toLowerCase()}`
}

export function isSameColor(color1: string, color2: string): boolean {
  return normalizeHex(color1) === normalizeHex(color2)
}

export function isRgbEqualToHex(rgb: RgbArray | null | undefined, hex: string): boolean {
  return Boolean(rgb && isSameColor(rgbArrayToHex(rgb), hex))
}

export function colorDifference(rgb1: RgbArray, rgb2: RgbArray): number {
  const redDelta = rgb1[0] - rgb2[0]
  const greenDelta = rgb1[1] - rgb2[1]
  const blueDelta = rgb1[2] - rgb2[2]
  return Math.sqrt(redDelta * redDelta + greenDelta * greenDelta + blueDelta * blueDelta)
}

export function isDarkColor(rgb: RgbArray): boolean {
  const luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
  return luminance < DARK_LUMINANCE_THRESHOLD
}

export function getContrastColor(rgb: RgbArray): string {
  return isDarkColor(rgb) ? '#ffffff' : '#000000'
}

export function formatRgb(rgb: RgbArray): string {
  return `RGB(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
}

export function formatConfidence(confidence: number): string {
  return `${Math.round(confidence * 100)}%`
}
