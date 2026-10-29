import { flushPromises, mount } from '@vue/test-utils'
import { reactive } from 'vue'
import { beforeEach, expect, it, vi } from 'vitest'
import ChapterTextSearch from '@/components/edit/ChapterTextSearch.vue'
import { createBubbleState } from '@/utils/bubbleFactory'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'
import type { V2PageDocument } from '@/api/v2/content'

const mocks = vi.hoisted(() => ({ get: vi.fn(), document: vi.fn(), flush: vi.fn(), register: vi.fn(), queue: vi.fn(), toast: vi.fn() }))
const images = reactive<{ currentImage: { id: string; chapterId: string; documentRevision: number; bubbleStates: ReturnType<typeof createBubbleState>[] } | null }>({ currentImage: null })
vi.mock('@/api/client', () => ({ apiClient: { get: mocks.get } }))
vi.mock('@/api/v2/content', () => ({ getPageDocument: mocks.document }))
vi.mock('@/api/v2/translation', () => ({ getChapterTextExportUrl: (chapter: string) => `/chapters/${chapter}/text` }))
vi.mock('@/services/pageDocumentPersistence', () => ({ flushPageDocument: mocks.flush, registerPageDocument: mocks.register, queuePageDocumentMutation: mocks.queue }))
vi.mock('@/stores/imageStore', () => ({ useImageStore: () => images }))
vi.mock('@/utils/toast', () => ({ showToast: mocks.toast }))

function document(page: string, revision = 1): V2PageDocument {
  const { fontFamily: _font, ...payload } = createBubbleState({ translatedText: '术语😀', richText: {
    runs: [{ len: 6, style: { size_pt: 24 } }, { len: 4, style: { size_pt: 36, underline: true } }],
    paragraphs: [{ len: 10, style: { align: 'Center' } }],
  } })
  return { pageId: page, chapterId: 'chapter', documentRevision: revision, renderStatus: 'ready', defaultFontId: 'font',
    pageStyleDefaults: TEXT_STYLE_DEFAULTS, bubbles: [{ bubbleId: `bubble-${page}`, ordinal: 1, fontId: 'font', updatedRevision: revision, payload }] }
}
const documents = () => ['page-1', 'page-2'].map(page => document(page))
const textExport = () => ({ pages: documents().map((page, i) => ({ page_id: page.pageId, page_number: i + 1,
  document_revision: page.documentRevision, bubbles: [{ bubble_id: page.bubbles[0]!.bubbleId, translated_text: '术语😀' }] })) })

beforeEach(() => {
  vi.clearAllMocks()
  images.currentImage = { id: 'page-1', chapterId: 'chapter', documentRevision: 1,
    bubbleStates: [createBubbleState({ backendBubbleId: 'bubble-page-1', translatedText: '术语😀' })] }
  mocks.get.mockResolvedValue(textExport())
  mocks.document.mockImplementation(async (page: string) => document(page))
  mocks.flush.mockResolvedValue(undefined)
  mocks.queue.mockResolvedValue(undefined)
})

async function openSearch() {
  const prepare = vi.fn().mockResolvedValue(undefined)
  const wrapper = mount(ChapterTextSearch, { props: { chapterId: 'chapter', prepare } })
  await wrapper.get('button').trigger('click')
  await flushPromises()
  await wrapper.get('input[aria-label="查找译文"]').setValue('术语')
  await wrapper.get('input[aria-label="替换为"]').setValue('称谓')
  return { wrapper, prepare, button: (text: string) => wrapper.findAll('button').find(button => (button.attributes('aria-label') ?? button.text()) === text)! }
}

it('locates matches without loading image data or committing a document', async () => {
  const { wrapper, button, prepare } = await openSearch()
  expect(wrapper.get('[role="status"]').text()).toBe('共 2 处 / 2 个框')
  await button('第 2 页 · 术语😀').trigger('click')
  expect(wrapper.emitted('locate')).toEqual([['page-2', 'bubble-page-2']])
  expect(prepare).toHaveBeenCalledOnce()
  expect(mocks.document).not.toHaveBeenCalled()
  expect(mocks.queue).not.toHaveBeenCalled()
  wrapper.unmount()
})

it('requires confirmation, then commits only text fields through the existing coordinator for each page', async () => {
  const { wrapper, button } = await openSearch()
  await button('全部替换').trigger('click')
  expect(mocks.queue).not.toHaveBeenCalled()
  await button('确认替换全部 2 处').trigger('click')
  await flushPromises()
  expect(mocks.queue).toHaveBeenCalledTimes(2)
  for (const [page, revision, bubbles, patch] of mocks.queue.mock.calls) {
    expect(revision).toBe(1)
    expect(patch).toEqual({ bubbleFields: { [`bubble-${page}`]: ['translatedText', 'richText'] } })
    expect(bubbles[0].translatedText).toBe('称谓😀')
    expect(bubbles[0].richText.runs).toEqual([{ len: 6, style: { size_pt: 24 } }, { len: 4, style: { size_pt: 36, underline: true } }])
  }
  expect(wrapper.emitted('busy')).toEqual([[true], [false]])
  wrapper.unmount()
})

it('keeps a stale-preview error visible after reloading results, without overwriting newer data', async () => {
  const { wrapper, button } = await openSearch()
  mocks.document.mockResolvedValue(document('page-1', 2))
  await button('替换首个框').trigger('click')
  await flushPromises()
  expect(mocks.queue).not.toHaveBeenCalled()
  expect(wrapper.get('[role="alert"]').text()).toContain('章节文字已发生变化')
  expect(mocks.get).toHaveBeenCalledTimes(2)
  wrapper.unmount()
})

it('reports partial completion and stops on the failing page', async () => {
  const { wrapper, button } = await openSearch()
  mocks.queue.mockResolvedValueOnce(undefined).mockRejectedValueOnce(new Error('第二页保存失败'))
  await button('全部替换').trigger('click')
  await button('确认替换全部 2 处').trigger('click')
  await flushPromises()
  expect(wrapper.get('[role="alert"]').text()).toContain('第二页保存失败（已完成 1 页）')
  expect(wrapper.emitted('busy')).toEqual([[true], [false]])
  wrapper.unmount()
})
