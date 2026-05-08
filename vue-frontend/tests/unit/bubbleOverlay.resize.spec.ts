import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
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

    await wrapper.find('.resize-handle.e').trigger('mousedown', {
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

    await wrapper.find('.resize-handle.se').trigger('mousedown', {
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

    await wrapper.find('.resize-handle.e').trigger('mousedown', {
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
})
