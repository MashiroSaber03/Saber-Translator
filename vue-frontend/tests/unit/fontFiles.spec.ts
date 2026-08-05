import { describe, expect, it } from 'vitest'

import {
  FONT_FILE_ACCEPT,
  isSupportedFontFileName,
} from '@/utils/fontFiles'

describe('font file contracts', () => {
  it('keeps picker and validation support aligned, including TTC collections', () => {
    expect(FONT_FILE_ACCEPT).toBe('.ttf,.ttc,.otf,.woff,.woff2')
    expect(isSupportedFontFileName('微软雅黑.TTC')).toBe(true)
    expect(isSupportedFontFileName('font.exe')).toBe(false)
  })
})
