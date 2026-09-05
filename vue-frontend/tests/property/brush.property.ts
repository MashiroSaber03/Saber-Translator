import { defineComponent, h } from 'vue'
import { mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import * as fc from 'fast-check'
import { BRUSH_DEFAULT_SIZE, BRUSH_MAX_SIZE, BRUSH_MIN_SIZE } from '@/constants'
import { useBrush } from '@/composables/useBrush'

type BrushApi = ReturnType<typeof useBrush>

const validBrushSizeArb = fc.integer({ min: BRUSH_MIN_SIZE, max: BRUSH_MAX_SIZE })
const anyBrushSizeArb = fc.integer({ min: -100, max: 500 })
const adjustDeltaArb = fc.integer({ min: -300, max: 300 })

const mountedWrappers: Array<{ unmount: () => void }> = []

function createBrushHarness(): BrushApi {
  let brush: BrushApi | null = null
  const wrapper = mount(defineComponent({
    setup() {
      brush = useBrush()
      return () => h('div')
    },
  }))
  mountedWrappers.push(wrapper)

  if (!brush) {
    throw new Error('useBrush harness did not initialize')
  }
  return brush
}

describe('brush composable property contracts', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  afterEach(() => {
    while (mountedWrappers.length > 0) {
      mountedWrappers.pop()?.unmount()
    }
  })

  it('keeps large size adjustments inside the product brush bounds', () => {
    fc.assert(
      fc.property(anyBrushSizeArb, (size) => {
        const brush = createBrushHarness()

        brush.adjustBrushSize(size - BRUSH_DEFAULT_SIZE)

        expect(brush.brushSize.value).toBeGreaterThanOrEqual(BRUSH_MIN_SIZE)
        expect(brush.brushSize.value).toBeLessThanOrEqual(BRUSH_MAX_SIZE)
      }),
      { numRuns: 100 },
    )
  })

  it('keeps relative size adjustments inside the product brush bounds', () => {
    fc.assert(
      fc.property(validBrushSizeArb, adjustDeltaArb, (initialSize, delta) => {
        const brush = createBrushHarness()
        brush.adjustBrushSize(initialSize - BRUSH_DEFAULT_SIZE)

        brush.adjustBrushSize(delta)

        expect(brush.brushSize.value).toBeGreaterThanOrEqual(BRUSH_MIN_SIZE)
        expect(brush.brushSize.value).toBeLessThanOrEqual(BRUSH_MAX_SIZE)
      }),
      { numRuns: 100 },
    )
  })

  it('keeps repeated size adjustments bounded', () => {
    fc.assert(
      fc.property(
        validBrushSizeArb,
        fc.array(adjustDeltaArb, { minLength: 1, maxLength: 10 }),
        (initialSize, deltas) => {
          const brush = createBrushHarness()
          brush.adjustBrushSize(initialSize - BRUSH_DEFAULT_SIZE)

          for (const delta of deltas) {
            brush.adjustBrushSize(delta)
            expect(brush.brushSize.value).toBeGreaterThanOrEqual(BRUSH_MIN_SIZE)
            expect(brush.brushSize.value).toBeLessThanOrEqual(BRUSH_MAX_SIZE)
          }
        },
      ),
      { numRuns: 50 },
    )
  })

  it('keeps the default size inside the configured bounds', () => {
    const brush = createBrushHarness()

    expect(BRUSH_DEFAULT_SIZE).toBeGreaterThanOrEqual(BRUSH_MIN_SIZE)
    expect(BRUSH_DEFAULT_SIZE).toBeLessThanOrEqual(BRUSH_MAX_SIZE)
    expect(brush.brushSize.value).toBe(BRUSH_DEFAULT_SIZE)
  })

  it('keeps mode toggles and active state consistent', () => {
    fc.assert(
      fc.property(fc.constantFrom('repair', 'restore'), (mode) => {
        const brush = createBrushHarness()

        brush.toggleBrushMode(mode)
        expect(brush.brushMode.value).toBe(mode)

        brush.toggleBrushMode(mode)
        expect(brush.brushMode.value).toBeNull()
      }),
      { numRuns: 20 },
    )
  })

  it('switches between brush modes without leaving stale active state', () => {
    const brush = createBrushHarness()

    brush.toggleBrushMode('repair')
    brush.toggleBrushMode('restore')

    expect(brush.brushMode.value).toBe('restore')

    brush.exitBrushMode()

    expect(brush.brushMode.value).toBeNull()
  })
})
