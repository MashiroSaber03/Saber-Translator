import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import { useImageViewer } from '@/composables/useImageViewer'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('useImageViewer', () => {
  it('preserves the pan position while the viewport is temporarily hidden', () => {
    let visible = true
    const viewport = document.createElement('div')
    viewport.getBoundingClientRect = () => visible ? new DOMRect(10, 20, 600, 400) : new DOMRect()
    const viewer = useImageViewer({ pixelRatio: () => 1.25, viewport: () => viewport })
    viewer.resetZoom()
    const initial = viewer.getTransform()
    for (let cycle = 0; cycle < 10; cycle++) {
      visible = false
      viewer.setTransform({})
      visible = true
      viewer.setTransform({})
    }
    expect(viewer.getTransform()).toEqual(initial)
  })

  it.each([1, 1.25, 1.5, 2])('aligns one image pixel to one physical pixel at DPR %s', ratio => {
    const viewport = document.createElement('div')
    viewport.getBoundingClientRect = () => new DOMRect(10.2, 20.3, 600, 400)
    const viewer = useImageViewer({ pixelRatio: () => ratio, viewport: () => viewport })
    viewer.resetZoom()
    expect(viewer.scale.value * ratio).toBeCloseTo(1)
    expect((10.2 + viewer.translateX.value) * ratio).toBeCloseTo(Math.round(10.2 * ratio))
    expect((20.3 + viewer.translateY.value) * ratio).toBeCloseTo(Math.round(20.3 * ratio))

    const before = viewer.getTransform()
    viewer.startDrag(10, 20)
    for (let x = 11; x <= 75; x++) viewer.drag(x, 20)
    expect(viewer.translateX.value - before.translateX).toBeCloseTo(65)
    viewer.endDrag()
    expect(Math.abs(viewer.translateX.value - before.translateX - 65) * ratio).toBeLessThanOrEqual(.5)
    expect((10.2 + viewer.translateX.value) * ratio).toBeCloseTo(Math.round((10.2 + viewer.translateX.value) * ratio))
  })

  it('keeps fractional transforms for smooth shrinking and for generic viewers', () => {
    const viewer = useImageViewer({ pixelRatio: () => 1.25 })
    viewer.setTransform({ scale: .4, translateX: 12.3, translateY: 24.6 })
    expect(viewer.getTransform()).toEqual({ scale: .4, translateX: 12.3, translateY: 24.6 })
    const generic = useImageViewer()
    generic.setTransform({ translateX: 12.3, translateY: 24.6 })
    expect(generic.getTransform()).toEqual({ scale: 1, translateX: 12.3, translateY: 24.6 })
  })

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

  it('clamps setTransform scale before later zoom math uses it', () => {
    const viewer = useImageViewer({ minScale: 0.25, maxScale: 4 })

    viewer.setTransform({ scale: 0, translateX: 12, translateY: 24 })
    expect(viewer.scale.value).toBe(0.25)

    viewer.zoomAt(50, 50, 8)
    const transform = viewer.getTransform()

    expect(transform.scale).toBeGreaterThanOrEqual(0.25)
    expect(transform.scale).toBeLessThanOrEqual(4)
    expect(Number.isFinite(transform.translateX)).toBe(true)
    expect(Number.isFinite(transform.translateY)).toBe(true)
  })

  it('normalizes invalid scale options before zoom math uses them', () => {
    const viewer = useImageViewer({ minScale: 0, maxScale: 0 })

    viewer.setTransform({ scale: 0 })
    viewer.zoomAt(50, 50, 8)

    const transform = viewer.getTransform()

    expect(transform.scale).toBeGreaterThan(0)
    expect(Number.isFinite(transform.translateX)).toBe(true)
    expect(Number.isFinite(transform.translateY)).toBe(true)
  })

  it('keeps reset and drag state valid for constrained or malformed input', () => {
    const viewer = useImageViewer({ minScale: 2, maxScale: 4 })

    expect(viewer.scale.value).toBe(2)
    viewer.startDrag(Number.NaN, 10)
    viewer.drag(100, 100)
    expect(viewer.getTransform()).toEqual({ scale: 2, translateX: 0, translateY: 0 })

    viewer.startDrag(10, 10)
    viewer.drag(Number.POSITIVE_INFINITY, 20)
    viewer.resetZoom()
    expect(viewer.getTransform()).toEqual({ scale: 2, translateX: 0, translateY: 0 })
  })
})
