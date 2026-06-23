import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ImageUpload from '@/components/translate/ImageUpload.vue'
import { useWebImportStore } from '@/stores/webImportStore'

vi.mock('@/api/system', () => ({
  parsePdfStart: vi.fn(),
  parsePdfBatch: vi.fn(),
  parsePdfCleanup: vi.fn(),
  parseMobiStart: vi.fn(),
  parseMobiBatch: vi.fn(),
  parseMobiCleanup: vi.fn(),
}))

describe('ImageUpload', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('uses button semantics for upload action triggers', async () => {
    const webImportStore = useWebImportStore()
    const openModalSpy = vi.spyOn(webImportStore, 'openModal')
    const wrapper = mount(ImageUpload)

    const actionButtons = wrapper.findAll('.select-link')
    expect(actionButtons).toHaveLength(3)
    expect(actionButtons.map((button) => button.element.tagName)).toEqual(['BUTTON', 'BUTTON', 'BUTTON'])

    await wrapper.get('.web-import-link').trigger('click')

    expect(openModalSpy).toHaveBeenCalled()
  })
})
