export interface HsvColor {
  h: number
  s: number
  v: number
}

export function hexToHsv(hex: string): HsvColor {
  const rgb = Number.parseInt(hex.replace('#', ''), 16)
  const r = ((rgb >> 16) & 255) / 255
  const g = ((rgb >> 8) & 255) / 255
  const b = (rgb & 255) / 255
  const v = Math.max(r, g, b)
  const delta = v - Math.min(r, g, b)
  let h = 0
  if (delta) {
    if (v === r) h = ((g - b) / delta) % 6
    else if (v === g) h = (b - r) / delta + 2
    else h = (r - g) / delta + 4
  }
  return { h: (h * 60 + 360) % 360, s: v ? delta / v : 0, v }
}

export function hsvToHex({ h, s, v }: HsvColor): string {
  const sector = ((h % 360 + 360) % 360) / 60
  const chroma = v * s
  const x = chroma * (1 - Math.abs(sector % 2 - 1))
  const offset = v - chroma
  const rgb = [
    [chroma, x, 0], [x, chroma, 0], [0, chroma, x],
    [0, x, chroma], [x, 0, chroma], [chroma, 0, x],
  ][Math.floor(sector)]!
  return `#${rgb.map(channel => Math.round((channel + offset) * 255).toString(16).padStart(2, '0')).join('')}`
}
