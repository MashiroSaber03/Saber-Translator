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
