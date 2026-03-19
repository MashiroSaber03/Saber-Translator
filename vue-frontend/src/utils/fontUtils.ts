import type { FontInfo } from '@/types/api'

export interface FontSelectGroup {
  label: string
  options: Array<{ label: string; value: string }>
}

function inferSource(font: FontInfo): 'builtin' | 'custom' | 'system' | 'local' {
  if (font.source === 'local') {
    return font.is_default ? 'builtin' : 'custom'
  }
  if (font.source) return font.source
  if (font.path.startsWith('fonts/')) {
    return font.is_default ? 'builtin' : 'custom'
  }
  return 'system'
}

export function normalizeFontList(fonts?: FontInfo[] | string[]): FontInfo[] {
  if (!fonts) return []

  const normalized = fonts.map((font) => {
    if (typeof font === 'string') {
      const fileName = font.split(/[\\/]/).pop() || font
      return {
        file_name: fileName,
        display_name: fileName.replace(/\.(ttf|ttc|otf)$/i, ''),
        path: font,
        is_default: font.startsWith('fonts/'),
        source: font.startsWith('fonts/') ? 'builtin' : 'system'
      } satisfies FontInfo
    }

    return {
      ...font,
      source: inferSource(font)
    }
  })

  const deduped = new Map<string, FontInfo>()
  for (const font of normalized) {
    const key = font.path.toLowerCase()
    if (!deduped.has(key)) {
      deduped.set(key, font)
    }
  }

  return Array.from(deduped.values())
}

export function ensureFontPresent(fonts: FontInfo[], fontPath: string, displayName?: string): FontInfo[] {
  if (!fontPath) return fonts
  if (fonts.some((font) => font.path === fontPath)) {
    return fonts
  }

  const fileName = fontPath.split(/[\\/]/).pop() || fontPath
  return [
    {
      file_name: fileName,
      display_name: displayName || fileName.replace(/\.(ttf|ttc|otf)$/i, ''),
      path: fontPath,
      is_default: fontPath.startsWith('fonts/'),
      source: fontPath.startsWith('fonts/') ? 'builtin' : 'system'
    },
    ...fonts
  ]
}

export function createFontSelectGroups(fonts: FontInfo[]): FontSelectGroup[] {
  const buckets: Record<'builtin' | 'custom' | 'system' | 'local', FontInfo[]> = {
    builtin: [],
    custom: [],
    system: [],
    local: []
  }

  for (const font of fonts) {
    buckets[inferSource(font)].push(font)
  }

  const defs: Array<{ key: keyof typeof buckets; label: string }> = [
    { key: 'builtin', label: '内置字体' },
    { key: 'system', label: '系统字体' },
    { key: 'custom', label: '自定义字体' },
    { key: 'local', label: '本地字体' }
  ]

  return defs
    .map(({ key, label }) => ({
      label,
      options: buckets[key]
        .sort((a, b) => a.display_name.localeCompare(b.display_name, 'zh-CN'))
        .map((font) => ({
          label: font.display_name || font.file_name,
          value: font.path
        }))
    }))
    .filter((group) => group.options.length > 0)
}

export function getFontDisplayName(fontPath: string, fonts: FontInfo[]): string {
  const match = fonts.find((font) => font.path === fontPath)
  if (match?.display_name) return match.display_name
  const fileName = fontPath.split(/[\\/]/).pop() || fontPath
  return fileName.replace(/\.(ttf|ttc|otf)$/i, '')
}
