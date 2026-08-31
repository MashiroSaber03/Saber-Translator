import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useEditWorkspace } from '@/components/edit/useEditWorkspace'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'
import { useImageStore } from '@/stores/imageStore'
import type { BubbleState } from '@/types/bubble'
import { createBubbleState } from '@/utils/bubbleFactory'
import { addTestImage } from '../helpers/imageFixtures'

const mocks = vi.hoisted(() => ({
  flushPageDocument: vi.fn(),
  getPageDocument: vi.fn(),
  queuePageDocumentSave: vi.fn(),
  registerPageDocument: vi.fn(),
  reRenderFullImage: vi.fn(),
  toast: vi.fn(),
}))

vi.mock('@/api/v2/content', () => ({
  getPageDocument: mocks.getPageDocument,
}))

vi.mock('@/services/pageDocumentPersistence', () => ({
  flushPageDocument: mocks.flushPageDocument,
  queuePageDocumentSave: mocks.queuePageDocumentSave,
  registerPageDocument: mocks.registerPageDocument,
}))

vi.mock('@/composables/useEditRender', () => ({
  useEditRender: () => ({
    reRenderFullImage: mocks.reRenderFullImage,
  }),
}))

vi.mock('@/composables/useTranslationPipeline', () => ({
  useTranslation: () => ({
    translateWithCurrentBubbles: vi.fn(),
  }),
}))

vi.mock('@/utils/toast', () => ({
  showToast: mocks.toast,
}))

interface WorkspaceHarness {
  applyAndNext: () => Promise<void>
  currentImageIndex: number
  isBusy: boolean
}

async function mountWorkspace(bubbles: BubbleState[]) {
  const imageStore = useImageStore()
  addTestImage(imageStore, '1.png', '/api/v2/assets/source-1', {
    chapterId: 'chapter-1',
    documentRevision: 3,
    id: 'page-1',
    translationStatus: 'processing',
  })
  addTestImage(imageStore, '2.png', '/api/v2/assets/source-2', {
    chapterId: 'chapter-1',
    documentRevision: 3,
    id: 'page-2',
  })
  mocks.registerPageDocument.mockReturnValue(bubbles)
  const Harness = defineComponent({
    setup() {
      const workspace = useEditWorkspace(vi.fn())
      return {
        applyAndNext: workspace.applyAndNext,
        currentImageIndex: workspace.currentImageIndex,
        isBusy: workspace.isBusy,
      }
    },
    render: () => h('div'),
  })
  const wrapper = mount(Harness)
  await flushPromises()
  return wrapper
}

describe('EditWorkspace backend-owned navigation', () => {
  const source = readFileSync(
    resolve(process.cwd(), 'src/components/edit/useEditWorkspace.ts'),
    'utf8',
  )

  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    mocks.flushPageDocument.mockResolvedValue(undefined)
    mocks.queuePageDocumentSave.mockResolvedValue(undefined)
    mocks.reRenderFullImage.mockResolvedValue(false)
    mocks.getPageDocument.mockImplementation(async (pageId: string) => ({
      bubbles: [],
      chapterId: 'chapter-1',
      defaultFontId: null,
      documentRevision: 3,
      pageId,
      pageStyleDefaults: { ...TEXT_STYLE_DEFAULTS },
      pageStyleSchemaVersion: 2,
      renderStatus: 'stale',
    }))
  })

  it('persists the authoritative page document before ordinary navigation', () => {
    expect(source).not.toContain('prepareForNavigation')
    expect(source).toContain('await persistCurrentDocument()')
    expect(source).toMatch(/await Promise\.all\(\[\s*queuePageDocumentSave\([\s\S]*?flushPageDocument\(image\.id\),\s*\]\)/)
    expect(source).toContain('navigateAfterPersist(() => imageStore.goToNext())')
    expect(source).toContain('navigateAfterPersist(() => imageStore.goToPrevious())')
    expect(source).toContain('navigateAfterPersist(() => imageStore.setCurrentImageIndex(index))')
  })

  it.each([
    ['没有气泡', () => []],
    ['气泡译文为空', () => [createBubbleState({
      backendBubbleId: 'bubble-1',
      coords: [0, 0, 100, 60],
      originalText: '原文',
      translatedText: '',
    })]],
  ] satisfies Array<[string, () => BubbleState[]]>)(
    '在页面状态仍为 stale 且%s时，只等待文档落盘便切换下一张',
    async (_label, createBubbles) => {
      const wrapper = await mountWorkspace(createBubbles())
      try {
        const workspace = wrapper.vm as unknown as WorkspaceHarness
        expect(workspace.isBusy).toBe(false)

        await workspace.applyAndNext()

        expect(workspace.currentImageIndex).toBe(1)
        expect(mocks.queuePageDocumentSave).toHaveBeenCalledWith(
          'page-1',
          3,
          expect.any(Array),
        )
        expect(mocks.flushPageDocument).toHaveBeenCalledWith('page-1')
        expect(mocks.reRenderFullImage).not.toHaveBeenCalled()
      } finally {
        wrapper.unmount()
      }
    },
  )

  it('在最后一张保存后给出提示并恢复按钮状态', async () => {
    const wrapper = await mountWorkspace([])
    try {
      const imageStore = useImageStore()
      imageStore.setCurrentImageIndex(1)
      await flushPromises()
      const workspace = wrapper.vm as unknown as WorkspaceHarness
      expect(workspace.isBusy).toBe(false)

      await workspace.applyAndNext()

      expect(workspace.currentImageIndex).toBe(1)
      expect(workspace.isBusy).toBe(false)
      expect(mocks.queuePageDocumentSave).toHaveBeenCalledWith(
        'page-2',
        3,
        expect.any(Array),
      )
      expect(mocks.flushPageDocument).toHaveBeenCalledWith('page-2')
      expect(mocks.toast).toHaveBeenCalledWith('已是最后一张图片', 'info')
      expect(mocks.reRenderFullImage).not.toHaveBeenCalled()
    } finally {
      wrapper.unmount()
    }
  })

  it('contains no legacy chapter-session initialization or browser save steps', () => {
    expect(source).not.toContain('isBookshelfSessionInitialized')
    expect(source).not.toContain('forceInitializeBookshelfSession')
    expect(source).not.toContain('saveBookshelfPageProgress')
    expect(source).not.toContain('saveStep')
  })

  it('waits for the authoritative page style before allowing edits', () => {
    expect(source).toContain('const isPageDocumentReady = ref(false)')
    expect(source).toContain('...document.pageStyleDefaults')
    expect(source).toContain('? { fontFamily: document.defaultFontId }')
    expect(source).toContain('!isPageDocumentReady.value')
    expect(source).toContain("{ flush: 'sync' }")
  })
})
