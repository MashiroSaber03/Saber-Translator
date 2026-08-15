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

  it('keeps the overlay on the shared production drag helper', () => {
    const overlaySource = readFileSync(resolve(process.cwd(), 'src/components/edit/BubbleOverlay.vue'), 'utf8')

    expect(overlaySource).toContain("import { calculateDraggedCoords } from '@/utils/bubbleDrag'")
  })
})
