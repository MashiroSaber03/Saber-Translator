export const BREAKPOINTS = {
  XS: 480,
  SM: 640,
  MD: 768,
  LG: 1024,
  XL: 1280,
  XXL: 1536,
} as const

export type DeviceType = 'mobile' | 'tablet' | 'desktop'
export type LayoutMode = 'compact' | 'normal' | 'wide'

export function getDeviceType(width: number): DeviceType {
  if (width < BREAKPOINTS.MD) return 'mobile'
  if (width < BREAKPOINTS.LG) return 'tablet'
  return 'desktop'
}

export function getLayoutMode(width: number): LayoutMode {
  if (width < BREAKPOINTS.MD) return 'compact'
  if (width < BREAKPOINTS.XL) return 'normal'
  return 'wide'
}

export function isMobileViewport(width: number): boolean {
  return width < BREAKPOINTS.MD
}
