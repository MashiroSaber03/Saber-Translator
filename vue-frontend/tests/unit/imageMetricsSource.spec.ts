import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('image metrics source contract', () => {
  it('keeps the coordinate utility compact and behavior-oriented', () => {
    const content = source('src/utils/imageMetrics.ts')

    for (const staleNarration of [
      '/**',
      '@param',
      '@returns',
      '@example',
      '图片显示指标计算工具函数',
      '图片显示指标接口',
      '图像内容在屏幕上的实际渲染宽度',
      '计算图像内容在其 img 元素中的实际显示指标',
      '考虑到 object-fit',
      'img 元素在屏幕上的实际渲染尺寸',
      '图片比元素框更',
      '图像内容在其元素框内的偏移',
      '避免除以零',
      '将图片坐标转换为屏幕坐标',
      '将屏幕坐标转换为图片坐标',
      '检查点是否在图片可视区域内',
    ]) {
      expect(content).not.toContain(staleNarration)
    }

    expect(content).toContain('function resolveContainedImageSize')
  })

  it('keeps the property suite focused on behavior contracts', () => {
    const content = source('tests/property/imageMetrics.property.ts')

    for (const staleNarration of [
      '/**',
      '图片显示指标计算属性测试',
      '测试内容',
      '生成有效',
      'Property 37',
      '验证',
      '// 图片坐标',
      '// 屏幕坐标',
      '// 可视内容',
    ]) {
      expect(content).not.toContain(staleNarration)
    }

    expect(content).toContain('imageToScreenCoords')
    expect(content).toContain('screenToImageCoords')
    expect(content).toContain('isPointInVisualContent')
  })
})
