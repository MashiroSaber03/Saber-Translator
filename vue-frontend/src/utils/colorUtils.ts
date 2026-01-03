/**
 * 颜色工具函数
 * 用于处理自动颜色提取相关的格式转换和验证
 */

/**
 * RGB 数组类型
 */
export type RgbArray = [number, number, number]

/**
 * 将 RGB 数组转换为 Hex 字符串
 * @example rgbArrayToHex([15, 20, 25]) => '#0f1419'
 */
export function rgbArrayToHex(rgb: RgbArray): string {
  const toHex = (n: number): string => {
    const clamped = Math.max(0, Math.min(255, Math.round(n)))
    return clamped.toString(16).padStart(2, '0')
  }
  return `#${toHex(rgb[0])}${toHex(rgb[1])}${toHex(rgb[2])}`
}

/**
 * 将 Hex 字符串转换为 RGB 数组
 * @example hexToRgbArray('#0f1419') => [15, 20, 25]
 */
export function hexToRgbArray(hex: string): RgbArray {
  const cleaned = hex.replace('#', '')
  const r = parseInt(cleaned.slice(0, 2), 16)
  const g = parseInt(cleaned.slice(2, 4), 16)
  const b = parseInt(cleaned.slice(4, 6), 16)
  return [r, g, b]
}

/**
 * 验证 Hex 颜色格式
 * @example isValidHex('#ff0000') => true
 * @example isValidHex('ff0000') => true
 * @example isValidHex('#xyz') => false
 */
export function isValidHex(hex: string): boolean {
  return /^#?[0-9A-Fa-f]{6}$/.test(hex)
}

/**
 * 确保 Hex 颜色带有 # 前缀
 * @example normalizeHex('ff0000') => '#ff0000'
 * @example normalizeHex('#FF0000') => '#ff0000'
 */
export function normalizeHex(hex: string): string {
  const cleaned = hex.replace('#', '').toLowerCase()
  return `#${cleaned}`
}

/**
 * 比较两个颜色是否相同（忽略大小写和 # 前缀）
 */
export function isSameColor(color1: string, color2: string): boolean {
  return normalizeHex(color1) === normalizeHex(color2)
}

/**
 * 检查 RGB 数组对应的 Hex 是否与给定颜色相同
 */
export function isRgbEqualToHex(rgb: RgbArray | null | undefined, hex: string): boolean {
  if (!rgb) return false
  const rgbHex = rgbArrayToHex(rgb)
  return isSameColor(rgbHex, hex)
}

/**
 * 计算两个 RGB 颜色的差异（简化版欧几里得距离）
 * 用于判断颜色是否足够接近
 */
export function colorDifference(rgb1: RgbArray, rgb2: RgbArray): number {
  const dr = rgb1[0] - rgb2[0]
  const dg = rgb1[1] - rgb2[1]
  const db = rgb1[2] - rgb2[2]
  return Math.sqrt(dr * dr + dg * dg + db * db)
}

/**
 * 判断颜色是否为深色
 * 使用感知亮度公式：Y = 0.299*R + 0.587*G + 0.114*B
 */
export function isDarkColor(rgb: RgbArray): boolean {
  const luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
  return luminance < 128
}

/**
 * 获取对比色（用于确保可读性）
 * 如果是深色返回白色，浅色返回黑色
 */
export function getContrastColor(rgb: RgbArray): string {
  return isDarkColor(rgb) ? '#ffffff' : '#000000'
}

/**
 * 格式化 RGB 为显示字符串
 * @example formatRgb([15, 20, 25]) => 'RGB(15, 20, 25)'
 */
export function formatRgb(rgb: RgbArray): string {
  return `RGB(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
}

/**
 * 格式化置信度为百分比字符串
 * @example formatConfidence(0.92) => '92%'
 */
export function formatConfidence(confidence: number): string {
  return `${Math.round(confidence * 100)}%`
}
