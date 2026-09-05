import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import {
  TEXT_STYLE_DEFAULTS,
  getTextStyleDefaults,
  normalizeImageTextStyleFields,
  normalizeTextStyleSettings,
} from '@/defaults/textStyleDefaults'
import { createDefaultSettings } from '@/stores/settings/defaults'

const bundledDefaults = getTextStyleDefaults()

describe('textStyleDefaults factory fallback', () => {
  it('keeps the bundled fallback immutable and returns defensive copies', () => {
    const first = getTextStyleDefaults()
    first.fontSize += 1

    expect(TEXT_STYLE_DEFAULTS.fontSize).toBe(bundledDefaults.fontSize)
    expect(getTextStyleDefaults()).toEqual(bundledDefaults)
    expect(createDefaultSettings().textStyle).toEqual(bundledDefaults)
  })

  it('keeps text-style field normalization on a shared field builder', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/defaults/textStyleDefaults.ts'), 'utf8')

    expect(source).toContain('function buildTextStyleFields')
    expect(source).not.toContain('fontSize: style.fontSize !== undefined')
    expect(source).not.toContain('fontSize: image.fontSize !== undefined')
  })

  it('normalizes settings and image text-style fields through the same parser rules', () => {
    const partialStyle = {
      fontSize: 24,
      autoFontSize: true,
      fontFamily: 'fonts/custom.ttf',
      layoutDirection: 'horizontal' as const,
      textColor: '#112233',
      fillColor: '#445566',
      inpaintMethod: 'litelama' as const,
      strokeEnabled: true,
      strokeColor: '#778899',
      strokeWidth: 3,
      lineSpacing: 1.25,
      inlineAlign: 'center' as const,
      blockAlign: 'end' as const,
      useAutoTextColor: true,
    }

    expect(normalizeTextStyleSettings(partialStyle)).toMatchObject(partialStyle)
    expect(normalizeImageTextStyleFields(partialStyle)).toMatchObject(partialStyle)
    expect(() => normalizeImageTextStyleFields({ fontSize: -1 })).toThrow('fontSize must be a positive integer')
    expect(() => normalizeImageTextStyleFields({ fontSize: '24' } as never)).toThrow(
      'fontSize must be a positive integer',
    )
  })

  it.each([0, 0.1, 0.5, 1.2, 1.25, 3])('preserves a stroke width of %s', strokeWidth => {
    expect(normalizeTextStyleSettings({ strokeWidth }).strokeWidth).toBe(strokeWidth)
    expect(normalizeImageTextStyleFields({ strokeWidth }).strokeWidth).toBe(strokeWidth)
  })

  it.each([-0.1, NaN, Infinity, true, '1.2'])(
    'rejects invalid stroke width %s',
    strokeWidth => {
      expect(() => normalizeTextStyleSettings({ strokeWidth } as never)).toThrow('strokeWidth')
      expect(() => normalizeImageTextStyleFields({ strokeWidth } as never)).toThrow('strokeWidth')
    },
  )
})
