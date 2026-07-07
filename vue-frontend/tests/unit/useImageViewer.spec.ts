import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import { useImageViewer } from '@/composables/useImageViewer'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('useImageViewer', () => {
  it('keeps the viewer composable source free of scaffold narration', () => {
    const content = source('src/composables/useImageViewer.ts')

    for (const staleNarration of [
      '图片查看器组合式函数',
      '提供缩放、平移、视口同步等功能',
      '// ============================================================',
      '类型定义',
      '默认配置',
      '组合式函数',
      '缩放方法',
      '平移方法',
      '重置方法',
      '状态获取/设置方法',
      '滚动到指定位置',
      '@param',
      '@returns',
      '// 状态',
      '// 返回',
    ]) {
      expect(content).not.toContain(staleNarration)
    }
  })

  it('keeps image viewer property tests on the production composable contract', () => {
    const content = source('tests/property/imageViewer.property.ts')

    for (const shadowImplementation of [
      'interface ViewerState',
      'interface ViewerConfig',
      'function zoomAt(',
      'function zoom(',
      'function reset(',
      'function fitToViewport(',
      'function pan(',
      'ImageViewer 组件属性测试',
      '@param',
      '@returns',
    ]) {
      expect(content).not.toContain(shadowImplementation)
    }
  })

  it('clamps setTransform scale before later zoom math uses it', () => {
    const viewer = useImageViewer({ minScale: 0.25, maxScale: 4 })

    viewer.setTransform({ scale: 0, translateX: 12, translateY: 24 })
    expect(viewer.scale.value).toBe(0.25)

    viewer.setScale(2, 100, 100)
    const transform = viewer.getTransform()

    expect(transform.scale).toBeGreaterThanOrEqual(0.25)
    expect(transform.scale).toBeLessThanOrEqual(4)
    expect(Number.isFinite(transform.translateX)).toBe(true)
    expect(Number.isFinite(transform.translateY)).toBe(true)
  })

  it('normalizes invalid scale options before zoom math uses them', () => {
    const viewer = useImageViewer({ minScale: 0, maxScale: 0 })

    viewer.setTransform({ scale: 0 })
    viewer.setScale(2, 100, 100)

    const transform = viewer.getTransform()

    expect(transform.scale).toBeGreaterThan(0)
    expect(Number.isFinite(transform.translateX)).toBe(true)
    expect(Number.isFinite(transform.translateY)).toBe(true)
  })

  it('ignores non-positive image dimensions when fitting to screen', () => {
    const viewer = useImageViewer({ minScale: 0.25, maxScale: 4 })

    viewer.setTransform({ scale: 1.5, translateX: 20, translateY: 30 })
    viewer.fitToScreen(-100, 200, 800, 600)

    expect(viewer.getTransform()).toEqual({
      scale: 1.5,
      translateX: 20,
      translateY: 30,
    })
  })
})
