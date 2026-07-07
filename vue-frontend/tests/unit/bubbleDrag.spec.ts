import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { calculateDraggedCoords } from '@/utils/bubbleDrag'
import type { BubbleCoords } from '@/types/bubble'

describe('bubble drag geometry', () => {
  it('keeps dragged coordinates inside image bounds while preserving visible size', () => {
    const coords: BubbleCoords = [100, 100, 200, 220]

    expect(calculateDraggedCoords(coords, -150, -180, 500, 500)).toEqual([0, 0, 100, 120])
    expect(calculateDraggedCoords(coords, 480, 460, 500, 500)).toEqual([400, 380, 500, 500])
  })

  it('uses the production drag helper in overlay and edit property tests', () => {
    const overlaySource = readFileSync(resolve(process.cwd(), 'src/components/edit/BubbleOverlay.vue'), 'utf8')
    const propertySource = readFileSync(resolve(process.cwd(), 'tests/property/editMode.property.ts'), 'utf8')

    expect(overlaySource).toContain("import { calculateDraggedCoords } from '@/utils/bubbleDrag'")
    expect(propertySource).toContain("import { calculateDraggedCoords } from '@/utils/bubbleDrag'")
    expect(propertySource).not.toContain('function calculateDraggedCoords')
    for (const staleNarration of [
      '编辑模式属性测试',
      '测试数据生成器',
      '辅助函数',
      '属性测试',
      '生成有效',
      '生成拖拽偏移量',
      '生成调整大小的手柄类型',
      '生成图片尺寸',
      '确保有效的矩形坐标',
      '// ============================================================',
      '/' + '**',
      '验证',
    ]) {
      expect(propertySource).not.toContain(staleNarration)
    }
  })
})
