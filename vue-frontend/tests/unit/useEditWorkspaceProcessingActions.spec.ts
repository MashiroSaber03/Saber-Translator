import { createPinia, setActivePinia, storeToRefs } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import { useEditWorkspaceProcessingActions } from '@/composables/edit/useEditWorkspaceProcessingActions'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { addTestImage } from '../helpers/imageFixtures'
import { createBubbleState } from '@/utils/bubbleFactory'

const mocks = vi.hoisted(() => ({
  confirm: vi.fn(),
  createChapterDetectJob: vi.fn(),
  getPageDocument: vi.fn(),
  queuePageDocumentSave: vi.fn(),
  registerPageDocument: vi.fn(),
  runPageOperation: vi.fn(),
  toast: vi.fn(),
  translateWithCurrentBubbles: vi.fn(),
}))

vi.mock('@/api/v2/content', () => ({
  getPageDocument: mocks.getPageDocument,
}))

vi.mock('@/api/v2/operations', () => ({
  runPageOperation: mocks.runPageOperation,
}))

vi.mock('@/api/v2/translation', () => ({
  createChapterDetectJob: mocks.createChapterDetectJob,
}))

vi.mock('@/services/pageDocumentPersistence', () => ({
  queuePageDocumentSave: mocks.queuePageDocumentSave,
  registerPageDocument: mocks.registerPageDocument,
}))

vi.mock('@/composables/useTranslationPipeline', () => ({
  useTranslation: () => ({
    translateWithCurrentBubbles: mocks.translateWithCurrentBubbles,
  }),
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: mocks.confirm,
}))

vi.mock('@/utils/toast', () => ({
  showToast: mocks.toast,
}))

function createActions(pageCount = 1) {
  const imageStore = useImageStore()
  const bubbleStore = useBubbleStore()
  for (let index = 0; index < pageCount; index += 1) {
    addTestImage(imageStore, `${index + 1}.png`, `/api/v2/assets/source-${index + 1}`, {
      chapterId: 'chapter-1',
      documentRevision: 3,
      id: `page-${index + 1}`,
    })
  }
  const bubble = createBubbleState({
    backendBubbleId: 'bubble-1',
    coords: [0, 0, 120, 80],
    originalText: '原文',
    polygon: [],
  })
  bubbleStore.setBubbles([bubble])
  const { bubbles } = storeToRefs(bubbleStore)
  return {
    actions: useEditWorkspaceProcessingActions({
      images: ref(imageStore.images),
      currentImage: ref(imageStore.currentImage),
      currentImageIndex: ref(imageStore.currentImageIndex),
      bubbles,
      selectFirstBubbleIfExists: vi.fn(),
    }),
    bubble,
  }
}

describe('useEditWorkspaceProcessingActions', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    mocks.confirm.mockResolvedValue(true)
    mocks.queuePageDocumentSave.mockResolvedValue(undefined)
    mocks.runPageOperation.mockResolvedValue({
      id: 'operation-1',
      result: {},
      status: 'completed',
    })
    mocks.getPageDocument.mockResolvedValue({
      bubbles: [],
      chapterId: 'chapter-1',
      defaultFontId: null,
      documentRevision: 4,
      pageId: 'page-1',
      pageStyleDefaults: {},
      pageStyleSchemaVersion: 1,
      renderStatus: 'not_rendered',
    })
    mocks.registerPageDocument.mockReturnValue([])
    mocks.createChapterDetectJob.mockResolvedValue({
      batchId: 'batch-1',
      jobIds: ['job-1'],
      status: 'queued',
    })
  })

  it('persists the page document and runs bubble translation on the backend', async () => {
    const { actions, bubble } = createActions()

    await actions.handleReTranslateBubble(0)

    expect(mocks.queuePageDocumentSave).toHaveBeenCalledWith(
      'page-1',
      3,
      [bubble],
    )
    expect(mocks.runPageOperation).toHaveBeenCalledWith(
      'page-1',
      {
        baseRevision: 3,
        bubbleId: 'bubble-1',
        kind: 'bubble_translate',
      },
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    )
    expect(mocks.toast).toHaveBeenCalledWith('重新翻译完成', 'success')
  })

  it('submits batch detection as a durable chapter job', async () => {
    const { actions } = createActions(2)

    await actions.detectAllImages()

    expect(mocks.createChapterDetectJob).toHaveBeenCalledWith(
      'chapter-1',
      ['page-1', 'page-2'],
    )
    expect(mocks.toast).toHaveBeenCalledWith(
      '批量检测已加入任务中心；关闭浏览器也会继续执行',
      'success',
    )
  })

  it('rejects an operation reload that belongs to another chapter', async () => {
    mocks.getPageDocument.mockResolvedValueOnce({
      bubbles: [],
      chapterId: 'other-chapter',
      defaultFontId: null,
      documentRevision: 4,
      pageId: 'page-1',
      pageStyleDefaults: {},
      pageStyleSchemaVersion: 1,
      renderStatus: 'not_rendered',
    })
    const { actions } = createActions()

    await actions.handleReTranslateBubble(0)

    expect(mocks.registerPageDocument).not.toHaveBeenCalled()
    expect(mocks.toast).toHaveBeenCalledWith(
      '页面 page-1 的后端文档身份不匹配',
      'error',
    )
    expect(mocks.toast).not.toHaveBeenCalledWith('重新翻译完成', 'success')
  })

  it('does not coerce a malformed detection result into an empty success', async () => {
    mocks.runPageOperation.mockResolvedValueOnce({
      id: 'operation-1',
      result: { bubbleCount: '0' },
      status: 'completed',
    })
    const { actions } = createActions()

    await actions.autoDetectBubbles()

    expect(mocks.toast).toHaveBeenCalledWith(
      '后端检测结果缺少有效的气泡数量',
      'error',
    )
    expect(mocks.toast).not.toHaveBeenCalledWith('未检测到文本框', 'info')
  })
})
