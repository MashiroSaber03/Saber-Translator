export const FONT_SIZE_PRESETS = [16, 20, 24, 28, 32, 36, 40, 48, 56, 64]
export const FONT_SIZE_STEP = 2
export const FONT_SIZE_MIN = 10
export const FONT_SIZE_MAX = 999
export const FONT_SIZE_CUSTOM_PRESETS_KEY = 'customFontSizePresets'

export const EDIT_VIEW_MODE = {
  DUAL: 'dual',
  ORIGINAL: 'original',
  TRANSLATED: 'translated',
} as const

export type EditViewMode = (typeof EDIT_VIEW_MODE)[keyof typeof EDIT_VIEW_MODE]

export const BRUSH_MIN_SIZE = 5
export const BRUSH_MAX_SIZE = 200
export const BRUSH_DEFAULT_SIZE = 30

export const EDIT_MODE_EVENT_NS = '.editModeUi'
export const LAYOUT_MODE_KEY = 'edit_mode_layout'
