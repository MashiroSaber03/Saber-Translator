import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import TagManageModal from '@/components/bookshelf/TagManageModal.vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { showToast } from '@/utils/toast'

vi.mock('@/utils/toast', () => ({
  showToast: vi.fn(),
}))

const BaseModalStub = defineComponent({
  template: '<section class="base-modal-stub"><slot /><footer><slot name="footer" /></footer></section>',
})

function mountModal() {
  setActivePinia(createPinia())
  return mount(TagManageModal, {
    global: {
      stubs: {
        BaseModal: BaseModalStub,
      },
    },
  })
}

describe('TagManageModal', () => {
  beforeEach(() => {
    vi.mocked(showToast).mockClear()
  })

  it('does not report success when tag creation fails', async () => {
    const wrapper = mountModal()
    const store = useBookshelfStore()
    store.createTag = vi.fn().mockResolvedValue(null)

    await wrapper.get('input[type="text"]').setValue('New Tag')
    await wrapper.get('button').trigger('click')
    await flushPromises()

    expect(showToast).toHaveBeenCalledWith('创建失败', 'error')
    expect(showToast).not.toHaveBeenCalledWith('标签创建成功', 'success')
  })
})
