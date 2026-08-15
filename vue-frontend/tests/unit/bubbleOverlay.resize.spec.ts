import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import BubbleOverlay from '@/components/edit/BubbleOverlay.vue'
import { createBubbleState } from '@/utils/bubbleFactory'
import type { BubbleCoords } from '@/types/bubble'
import type { ResizeHandle } from '@/utils/bubbleResize'

function makeBubble(coords: BubbleCoords, rotationAngle: number) {
  return createBubbleState({
    coords,
    polygon: [],
    rotationAngle,
  })
}

function getHandleVector(
  handle: ResizeHandle,
  halfWidth: number,
  halfHeight: number
): { x: number; y: number } {
  const x = handle.includes('w') ? -halfWidth : handle.includes('e') ? halfWidth : 0
  const y = handle.includes('n') ? -halfHeight : handle.includes('s') ? halfHeight : 0
  return { x, y }
}

function getVisibleHandlePosition(
  coords: BubbleCoords,
  rotationAngle: number,
  handle: ResizeHandle
): { x: number; y: number } {
  const [x1, y1, x2, y2] = coords
  const centerX = (x1 + x2) / 2
  const centerY = (y1 + y2) / 2
  const halfWidth = (x2 - x1) / 2
  const halfHeight = (y2 - y1) / 2
  const angleRad = rotationAngle * Math.PI / 180
  const cos = Math.cos(angleRad)
  const sin = Math.sin(angleRad)
  const local = getHandleVector(handle, halfWidth, halfHeight)

  return {
    x: centerX + local.x * cos - local.y * sin,
    y: centerY + local.x * sin + local.y * cos,
  }
}

function getResizeEndCoords(wrapper: ReturnType<typeof mount>): BubbleCoords {
  const resizeEvents = wrapper.emitted('resizeEnd')
  expect(resizeEvents).toBeTruthy()
  const lastEvent = resizeEvents?.[resizeEvents.length - 1]
  expect(lastEvent).toBeTruthy()
  const emittedCoords = lastEvent?.[1]
  expect(Array.isArray(emittedCoords)).toBe(true)
  return emittedCoords as BubbleCoords
}

describe('BubbleOverlay rotated resize', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('keeps the east handle under the pointer when resizing a 45 degree box', async () => {
    const rotationAngle = 45
    const initialCoords: BubbleCoords = [100, 100, 200, 200]
    const initialHandlePos = getVisibleHandlePosition(initialCoords, rotationAngle, 'e')
    const dragDelta = {
      x: Math.cos(Math.PI / 4) * 10,
      y: Math.sin(Math.PI / 4) * 10,
    }

    const wrapper = mount(BubbleOverlay, {
      props: {
        bubbles: [makeBubble(initialCoords, rotationAngle)],
        selectedIndex: 0,
        selectedIndices: [0],
        scale: 1,
        isDrawingMode: false,
        imageWidth: 1000,
        imageHeight: 1000,
      },
    })

    await wrapper.find('.bubble-overlay__resize-handle--e').trigger('mousedown', {
      button: 0,
      clientX: 320,
      clientY: 240,
    })

    document.dispatchEvent(new MouseEvent('mousemove', {
      bubbles: true,
      clientX: 320 + dragDelta.x,
      clientY: 240 + dragDelta.y,
    }))
    document.dispatchEvent(new MouseEvent('mouseup', {
      bubbles: true,
      button: 0,
      clientX: 320 + dragDelta.x,
      clientY: 240 + dragDelta.y,
    }))

    const newCoords = getResizeEndCoords(wrapper)
    const newHandlePos = getVisibleHandlePosition(newCoords, rotationAngle, 'e')

    expect(newHandlePos.x).toBeCloseTo(initialHandlePos.x + dragDelta.x, 0)
    expect(newHandlePos.y).toBeCloseTo(initialHandlePos.y + dragDelta.y, 0)
  })

  it('keeps the south-east corner under the pointer for rotated corner resizing', async () => {
    const rotationAngle = 30
    const initialCoords: BubbleCoords = [200, 120, 320, 260]
    const initialHandlePos = getVisibleHandlePosition(initialCoords, rotationAngle, 'se')
    const angleRad = rotationAngle * Math.PI / 180
    const dragDelta = {
      x: Math.cos(angleRad) * 12 - Math.sin(angleRad) * 8,
      y: Math.sin(angleRad) * 12 + Math.cos(angleRad) * 8,
    }

    const wrapper = mount(BubbleOverlay, {
      props: {
        bubbles: [makeBubble(initialCoords, rotationAngle)],
        selectedIndex: 0,
        selectedIndices: [0],
        scale: 1,
        isDrawingMode: false,
        imageWidth: 1000,
        imageHeight: 1000,
      },
    })

    await wrapper.find('.bubble-overlay__resize-handle--se').trigger('mousedown', {
      button: 0,
      clientX: 400,
      clientY: 300,
    })

    document.dispatchEvent(new MouseEvent('mousemove', {
      bubbles: true,
      clientX: 400 + dragDelta.x,
      clientY: 300 + dragDelta.y,
    }))
    document.dispatchEvent(new MouseEvent('mouseup', {
      bubbles: true,
      button: 0,
      clientX: 400 + dragDelta.x,
      clientY: 300 + dragDelta.y,
    }))

    const newCoords = getResizeEndCoords(wrapper)
    const newHandlePos = getVisibleHandlePosition(newCoords, rotationAngle, 'se')

    expect(newHandlePos.x).toBeCloseTo(initialHandlePos.x + dragDelta.x, 0)
    expect(newHandlePos.y).toBeCloseTo(initialHandlePos.y + dragDelta.y, 0)
  })

  it('keeps rotated resize results inside image bounds near the edge', async () => {
    const rotationAngle = 35
    const initialCoords: BubbleCoords = [140, 80, 220, 160]
    const wrapper = mount(BubbleOverlay, {
      props: {
        bubbles: [makeBubble(initialCoords, rotationAngle)],
        selectedIndex: 0,
        selectedIndices: [0],
        scale: 1,
        isDrawingMode: false,
        imageWidth: 240,
        imageHeight: 220,
      },
    })

    await wrapper.find('.bubble-overlay__resize-handle--e').trigger('mousedown', {
      button: 0,
      clientX: 260,
      clientY: 160,
    })

    document.dispatchEvent(new MouseEvent('mousemove', {
      bubbles: true,
      clientX: 360,
      clientY: 220,
    }))
    document.dispatchEvent(new MouseEvent('mouseup', {
      bubbles: true,
      button: 0,
      clientX: 360,
      clientY: 220,
    }))

    const [x1, y1, x2, y2] = getResizeEndCoords(wrapper)

    expect(x1).toBeGreaterThanOrEqual(0)
    expect(y1).toBeGreaterThanOrEqual(0)
    expect(x2).toBeLessThanOrEqual(240)
    expect(y2).toBeLessThanOrEqual(220)
    expect(x2).toBeGreaterThan(x1)
    expect(y2).toBeGreaterThan(y1)
  })

  it('rotates a selected bubble without routine console logs', async () => {
    const rotationAngle = 15
    const initialCoords: BubbleCoords = [100, 100, 200, 200]
    const wrapper = mount(BubbleOverlay, {
      props: {
        bubbles: [makeBubble(initialCoords, rotationAngle)],
        selectedIndex: 0,
        selectedIndices: [0],
        scale: 1,
        isDrawingMode: false,
        imageWidth: 1000,
        imageHeight: 1000,
      },
    })

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)

    try {
      await wrapper.find('.bubble-overlay__rotate-handle').trigger('mousedown', {
        button: 0,
        clientX: 180,
        clientY: 80,
      })

      document.dispatchEvent(new MouseEvent('mousemove', {
        bubbles: true,
        clientX: 190,
        clientY: 90,
      }))
      document.dispatchEvent(new MouseEvent('mouseup', {
        bubbles: true,
        button: 0,
        clientX: 190,
        clientY: 90,
      }))

      expect(wrapper.emitted('rotateEnd')).toBeTruthy()
      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }
  })

  it('selects once and does not persist a no-op click as a drag', async () => {
    const wrapper = mount(BubbleOverlay, {
      props: {
        bubbles: [makeBubble([100, 100, 200, 200], 0)],
        selectedIndex: -1,
        selectedIndices: [],
        scale: 1,
        imageWidth: 1000,
        imageHeight: 1000,
      },
    })
    const bubble = wrapper.get('.bubble-overlay__highlight-box')

    await bubble.trigger('mousedown', { button: 0, clientX: 150, clientY: 150 })
    await bubble.trigger('click', { button: 0, clientX: 150, clientY: 150 })

    expect(wrapper.emitted('select')).toEqual([[0]])
    expect(wrapper.emitted('dragEnd')).toBeUndefined()

    await wrapper.setProps({ selectedIndex: 0, selectedIndices: [0] })
    await bubble.trigger('mousedown', { button: 0, clientX: 150, clientY: 150 })
    document.dispatchEvent(new MouseEvent('mouseup', {
      bubbles: true,
      button: 0,
      clientX: 150,
      clientY: 150,
    }))
    expect(wrapper.emitted('dragEnd')).toBeUndefined()
  })

  it('does not start geometry mutations before real image bounds are known', async () => {
    const wrapper = mount(BubbleOverlay, {
      props: {
        bubbles: [makeBubble([100, 100, 200, 200], 0)],
        selectedIndex: 0,
        selectedIndices: [0],
        scale: -1,
        imageWidth: 0,
        imageHeight: 0,
      },
    })
    const bubble = wrapper.get('.bubble-overlay__highlight-box')

    expect(wrapper.get('.bubble-overlay').attributes('style')).toContain('--scale: 1')
    await bubble.trigger('mousedown', { button: 0, clientX: 150, clientY: 150 })
    document.dispatchEvent(new MouseEvent('mousemove', {
      bubbles: true,
      clientX: 180,
      clientY: 180,
    }))
    document.dispatchEvent(new MouseEvent('mouseup', {
      bubbles: true,
      button: 0,
      clientX: 180,
      clientY: 180,
    }))

    expect(wrapper.emitted('dragEnd')).toBeUndefined()
  })

  it('does not own the workspace middle-button drawing path', async () => {
    const wrapper = mount(BubbleOverlay, {
      props: {
        bubbles: [makeBubble([100, 100, 200, 200], 0)],
        selectedIndex: 0,
        selectedIndices: [0],
        scale: 1,
        isDrawingMode: false,
        imageWidth: 10,
        imageHeight: 10,
      },
    })

    await wrapper.get('.bubble-overlay').trigger('mousedown', {
      button: 1,
      clientX: 50,
      clientY: 50,
    })

    expect(document.body.classList.contains('middle-button-drawing')).toBe(false)
    expect(wrapper.emitted('drawBubble')).toBeUndefined()
  })

  it('maps overlay owner colors through semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleOverlay.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    expect(source).toContain('--bubble-overlay-selection-border: var(--color-action-success-bright)')
    expect(source).toContain('--bubble-overlay-box-border: color-mix')
  })

  it('keeps overlay style comments concise and current', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleOverlay.vue'),
      'utf8',
    )

    expect(source).not.toContain('【屏幕像素适配】')
    expect(source).not.toContain('这样边框、手柄等 UI 元素')
    expect(source).toContain('Inverse scaling keeps overlay controls usable while the image zoom changes.')
  })

  it('keeps overlay interaction hooks under the bubble-overlay owner', () => {
    const overlaySource = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleOverlay.vue'),
      'utf8',
    )
    const workspaceSource = readFileSync(
      resolve(process.cwd(), 'src/components/edit/useEditWorkspace.ts'),
      'utf8',
    )

    for (const currentHook of [
      'bubble-overlay__highlight-box',
      'bubble-overlay__highlight-box--selected',
      'bubble-overlay__highlight-box--multi-selected',
      'bubble-overlay__index',
      'bubble-overlay__resize-handle',
      'bubble-overlay__resize-handle--e',
      'bubble-overlay__rotate-handle',
      'bubble-overlay--brush-mode',
    ]) {
      expect(overlaySource).toContain(currentHook)
    }

    expect(workspaceSource).toContain(".closest('.bubble-overlay__highlight-box')")

    for (const oldHook of [
      'bubble-highlight-box',
      'bubble-index',
      'resize-handle',
      'rotate-handle',
      'drawing-rect',
    ]) {
      expect(overlaySource).not.toMatch(new RegExp(`class="[^"]*\\b${oldHook}\\b`))
      expect(overlaySource).not.toMatch(new RegExp(`\\.${oldHook}\\b`))
      expect(workspaceSource).not.toContain(`.${oldHook}`)
    }
    expect(overlaySource).not.toContain("'brush-mode': isBrushMode")
    expect(overlaySource).not.toContain('.bubble-overlay.brush-mode')
    expect(overlaySource).not.toContain('bubble-overlay__drawing-rect')
    expect(overlaySource).not.toContain('drawBubble')
    expect(overlaySource).not.toMatch(/\bselected:\s*index === selectedIndex/)
    expect(overlaySource).not.toContain("'multi-selected'")
  })
})
