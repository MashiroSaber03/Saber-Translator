export const SUPPORTED_FONT_FILE_EXTENSIONS = [
  '.ttf',
  '.ttc',
  '.otf',
  '.woff',
  '.woff2',
] as const

export const FONT_FILE_ACCEPT = SUPPORTED_FONT_FILE_EXTENSIONS.join(',')
export const FONT_FILE_FORMATS_LABEL = SUPPORTED_FONT_FILE_EXTENSIONS.join('、')

export function isSupportedFontFileName(fileName: string): boolean {
  const normalized = fileName.toLowerCase()
  return SUPPORTED_FONT_FILE_EXTENSIONS.some(extension => normalized.endsWith(extension))
}
