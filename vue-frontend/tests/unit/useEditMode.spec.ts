import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { useEditMode } from '@/composables/useEditMode'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { createBubbleState } from '@/utils/bubbleFactory'

describe('useEditMode', () => {
  it('keeps the small edit-mode entry owner free of scaffold narration', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/composables/useEditMode.ts'), 'utf8')

    expect(source).not.toContain('/**\n * 编辑模式组合式函数')
    expect(source).not.toContain('// ============================================================')
    expect(source).not.toContain('状态定义')
    expect(source).not.toContain('返回接口')
    expect(source).toContain('function exitEditModeWithoutRender')
  })

  beforeEach(() => {
    setActivePinia(createPinia())
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('syncs an empty bubble array on no-render exit without routine logs', () => {
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {})
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const editMode = useEditMode()

    imageStore.addImage('page.png', 'data:image/png;base64,page', {
      bubbleStates: [
        createBubbleState({ coords: [0, 0, 120, 120] }),
      ],
    })
    bubbleStore.clearBubblesLocal()
    editMode.isActive.value = true

    editMode.exitEditModeWithoutRender()

    expect(imageStore.currentImage?.bubbleStates).toEqual([])
    expect(editMode.isActive.value).toBe(false)
    expect(consoleLog).not.toHaveBeenCalled()
  })
})
