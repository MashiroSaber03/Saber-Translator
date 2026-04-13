/**
 * 翻译文本异常检测工具
 */

/** 是否包含平假名 / 片假名 / 长音符 */
export function containsJapaneseKana(text: string): boolean {
  return /[\u3040-\u309f\u30a0-\u30ffー]/.test(text || '')
}

/** 是否命中固定的翻译失败文案 */
export function isTranslationErrorText(text: string): boolean {
  const value = text || ''
  return (
    value.includes('【翻译失败】') ||
    value.includes('[翻译失败]') ||
    value.includes('翻译失败')
  )
}
