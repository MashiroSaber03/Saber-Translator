import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { createBubbleState } from '@/utils/bubbleFactory'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'

const {
  createStyleJobMock,
  queueMutationMock,
  refreshTasksMock,
  showToastMock,
} = vi.hoisted(() => ({
  createStyleJobMock: vi.fn(),
  queueMutationMock: vi.fn(),
  refreshTasksMock: vi.fn(),
  showToastMock: vi.fn(),
}))

vi.mock('@/api/v2/content', () => ({
  getPageSummary: vi.fn(),
}))

vi.mock('@/api/v2/translation', () => ({
  createChapterStyleApplyJob: createStyleJobMock,
}))

vi.mock('@/services/pageDocumentPersistence', () => ({
  queuePageDocumentMutation: queueMutationMock,
}))

vi.mock('@/stores/taskCenterStore', () => ({
  useTaskCenterStore: () => ({ refresh: refreshTasksMock }),
}))

vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

describe('useTextStyleSync backend ownership', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    createStyleJobMock.mockReset().mockResolvedValue({
      batchId: 'batch-1',
      jobIds: ['job-1'],
      status: 'queued',
    })
    queueMutationMock.mockReset().mockResolvedValue(undefined)
    refreshTasksMock.mockReset().mockResolvedValue(undefined)
    showToastMock.mockReset()
  })

  function mountPage() {
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const bubble = createBubbleState({
      autoBgColor: [10, 11, 12],
      autoFgColor: [1, 2, 3],
      coords: [0, 0, 120, 80],
      fontSize: 22,
      polygon: [],
      translatedText: '第一页译文',
    })
    const image = imageStore.addImage('page-1.png', '/api/v2/assets/source-1', {
      bubbleStates: [bubble],
      chapterId: 'chapter-1',
      documentRevision: 4,
    })
    bubbleStore.setBubbles([bubble])
    return { bubbleStore, image, imageStore }
  }

  it('contains no browser render or Base64 pipeline', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/composables/useTextStyleSync.ts'),
      'utf8',
    )

    expect(source).not.toContain('executeRender')
    expect(source).not.toContain('buildEditRenderInput')
    expect(source).not.toContain('base64')
    expect(source).toContain('queuePageDocumentMutation')
    expect(source).toContain('createChapterStyleApplyJob')
  })

  it('writes one page style and its bubble projection through the CAS coordinator', async () => {
    const { image } = mountPage()
    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')

    await useTextStyleSync().handleTextStyleChanged('fontFamily', 'font-id-2')

    expect(queueMutationMock).toHaveBeenCalledWith(
      image.id,
      4,
      [expect.objectContaining({ fontFamily: 'font-id-2' })],
      {
        defaultFontId: 'font-id-2',
        pageStyleDefaultsPatch: { fontFamily: 'font-id-2' },
        propagateStyleFields: ['fontFamily'],
      },
    )
  })

  it('materializes stored automatic colors before the backend CAS', async () => {
    const { image, imageStore } = mountPage()
    useSettingsStore().updateTextStyle({ useAutoTextColor: true })
    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')

    await useTextStyleSync().handleAutoTextColorChanged(true)

    expect(imageStore.currentImage?.bubbleStates?.[0]).toMatchObject({
      fillColor: '#0a0b0c',
      textColor: '#010203',
    })
    expect(queueMutationMock).toHaveBeenCalledWith(
      image.id,
      4,
      [expect.objectContaining({
        fillColor: '#0a0b0c',
        textColor: '#010203',
      })],
      {
        pageStyleDefaultsPatch: { useAutoTextColor: true },
        propagateStyleFields: ['useAutoTextColor'],
      },
    )
  })

  it('submits apply-to-all as one durable backend job', async () => {
    const { image } = mountPage()
    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')

    await useTextStyleSync().handleApplyToAll({
      fillColor: false,
      fontFamily: false,
      fontSize: true,
      layoutDirection: false,
      lineSpacing: false,
      strokeColor: false,
      strokeEnabled: false,
      strokeWidth: false,
      textAlign: true,
      textColor: false,
    })

    expect(createStyleJobMock).toHaveBeenCalledWith('chapter-1', {
      selectedFields: ['fontSize', 'textAlign'],
      sourceDocumentRevision: 4,
      sourcePageId: image.id,
    })
    expect(refreshTasksMock).toHaveBeenCalled()
    expect(showToastMock).toHaveBeenCalledWith(
      '样式应用任务已加入后端任务中心，可安全关闭页面',
      'success',
    )
  })
})
