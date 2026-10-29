import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h, nextTick, reactive } from 'vue'
import { beforeEach, expect, it, vi } from 'vitest'
import { useTextStyleFavorites } from '@/components/edit/useTextStyleFavorites'

const account = reactive({ user: { id: 'alice' } })
vi.mock('@/stores/authStore', () => ({ useAuthStore: () => account }))
vi.mock('@/utils/toast', () => ({ showToast: vi.fn() }))
beforeEach(() => { setActivePinia(createPinia()); localStorage.clear(); account.user.id = 'alice' })

it('reloads named favorites, isolates accounts, and applies snapshots without adding a page-data owner', async () => {
  let api!: ReturnType<typeof useTextStyleFavorites>
  const harness = defineComponent({ setup() { api = useTextStyleFavorites(); return () => h('div') } })
  const wrapper = mount(harness)
  api.name.value = '对白'
  const style = { character: { size_pt: 24 }, box: { strokeEnabled: true } }
  api.save(style)
  expect(api.favorites.value[0]).toEqual({ name: '对白', style })
  account.user.id = 'bob'
  await nextTick()
  expect(api.favorites.value).toEqual([])
  account.user.id = 'alice'
  await nextTick()
  expect(api.favorites.value[0]).toEqual({ name: '对白', style })
  api.selected.value = '对白'
  api.remove()
  expect(JSON.parse(localStorage.getItem('saber.text-styles.alice')!)).toEqual([])
  wrapper.unmount()
})
