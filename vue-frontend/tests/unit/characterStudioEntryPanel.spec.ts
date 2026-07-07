import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { mount } from '@vue/test-utils'
import CharacterStudioEntryPanel from '@/components/insight/CharacterStudioEntryPanel.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import { useInsightStore } from '@/stores/insightStore'

const pushMock = vi.fn()

vi.mock('vue-router', () => ({
  useRouter: () => ({
    push: pushMock,
  }),
}))

describe('CharacterStudioEntryPanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    pushMock.mockReset()
  })

  it('uses the product card shell and opens the studio for the current book', async () => {
    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'

    const wrapper = mount(CharacterStudioEntryPanel)

    expect(wrapper.findComponent(ProductRecordCard).exists()).toBe(true)

    await wrapper.get('button').trigger('click')

    expect(pushMock).toHaveBeenCalledWith({
      name: 'character-studio',
      query: { book: 'book-1' },
    })
  })

  it('keeps entry card hooks under the character studio entry owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/CharacterStudioEntryPanel.vue'), 'utf8')

    expect(source).not.toMatch(/(?<![\w-])eyebrow(?![\w-])/)
    expect(source).not.toContain('entry-content')
    expect(source).not.toMatch(/\.entry-content\s+[hp]\d?/)
    expect(source).toContain('character-studio-entry-panel__eyebrow')
    expect(source).toContain('character-studio-entry-panel__content')
    expect(source).toContain('character-studio-entry-panel__title')
    expect(source).toContain('character-studio-entry-panel__description')
  })
})
