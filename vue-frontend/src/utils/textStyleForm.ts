import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'

export const layoutDirectionOptions = [
  { label: '自动 (根据检测)', value: 'auto' },
  { label: '竖向排版', value: 'vertical' },
  { label: '横向排版', value: 'horizontal' },
]

export const textAlignOptions = [
  { label: '起始 (左/顶)', value: 'start' },
  { label: '居中', value: 'center' },
  { label: '末尾 (右/底)', value: 'end' },
]

export const inpaintMethodOptions = [
  { label: '纯色填充', value: 'solid' },
  { label: 'LAMA修复 (速度优化)', value: 'lama_mpe' },
  { label: 'LAMA修复 (通用)', value: 'litelama' },
]

export const FONT_NAME_MAP: Record<string, string> = {
  'fonts/STXINGKA.TTF': '华文行楷',
  'fonts/STXINWEI.TTF': '华文新魏',
  'fonts/STZHONGS.TTF': '华文中宋',
  'fonts/STKAITI.TTF': '楷体',
  'fonts/STLITI.TTF': '隶书',
  'fonts/思源黑体SourceHanSansK-Bold.TTF': '思源黑体',
  'fonts/STSONG.TTF': '华文宋体',
  'fonts/msyh.ttc': '微软雅黑',
  'fonts/msyhbd.ttc': '微软雅黑粗体',
  'fonts/SIMYOU.TTF': '幼圆',
  'fonts/STFANGSO.TTF': '仿宋',
  'fonts/STHUPO.TTF': '华文琥珀',
  'fonts/STXIHEI.TTF': '华文细黑',
  'fonts/simkai.ttf': '中易楷体',
  'fonts/simfang.ttf': '中易仿宋',
  'fonts/simhei.ttf': '中易黑体',
  'fonts/SIMLI.TTF': '中易隶书',
  'fonts/simsun.ttc': '宋体',
}

export const BUILTIN_FONTS = Array.from(new Set([
  TEXT_STYLE_DEFAULTS.fontFamily,
  'fonts/msyh.ttc',
  'fonts/simhei.ttf',
  'fonts/simsun.ttc',
]))

export function getFontDisplayName(fontPath: string): string {
  if (FONT_NAME_MAP[fontPath]) {
    return FONT_NAME_MAP[fontPath]
  }

  const fileName = fontPath.split('/').pop() || fontPath
  for (const [path, name] of Object.entries(FONT_NAME_MAP)) {
    const mapFileName = path.split('/').pop() || ''
    if (mapFileName.toLowerCase() === fileName.toLowerCase()) {
      return name
    }
  }

  return fileName.replace(/\.(ttf|ttc|otf)$/i, '')
}

export function clampLineSpacing(value: number, fallback: number = TEXT_STYLE_DEFAULTS.lineSpacing): number {
  let next = Number(value)
  if (!Number.isFinite(next) || next <= 0) {
    next = fallback
  }
  return Math.max(0.5, Math.min(3.0, next))
}
