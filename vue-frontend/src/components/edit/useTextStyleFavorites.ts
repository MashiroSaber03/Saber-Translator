import { computed, ref, watch } from 'vue'
import { useAuthStore } from '@/stores/authStore'
import { showToast } from '@/utils/toast'
import type { TextStyleTemplate } from './textStyleClipboard'

type Favorite = { name: string; style: TextStyleTemplate }

/** Browser preferences only. Applying a favorite still commits normal page data. */
export function useTextStyleFavorites() {
  const auth = useAuthStore()
  const favorites = ref<Favorite[]>([])
  const name = ref('')
  const selected = ref('')
  const key = computed(() => `saber.text-styles.${auth.user?.id ?? 'local'}`)
  watch(key, value => {
    selected.value = ''
    try {
      const items: unknown = JSON.parse(localStorage.getItem(value) ?? '[]')
      favorites.value = Array.isArray(items) ? items.slice(0, 12).filter(item => item
        && typeof item.name === 'string' && item.name.length <= 30 && item.style?.character
        && typeof item.style.character === 'object' && item.style.box && typeof item.style.box === 'object') : []
    } catch { favorites.value = [] }
  }, { immediate: true })
  function persist(next: Favorite[]): boolean {
    try { localStorage.setItem(key.value, JSON.stringify(next)); favorites.value = next; return true }
    catch { showToast('无法保存样式收藏，请检查浏览器存储空间', 'error'); return false }
  }
  function save(style: TextStyleTemplate): void {
    const label = name.value.trim().slice(0, 30)
    if (!label) { showToast('请为样式命名，例如：对白、旁白、注释', 'warning'); return }
    const next = favorites.value.filter(item => item.name !== label)
    if (next.length >= 12) { showToast('最多收藏 12 个样式，请先删除不用的样式', 'warning'); return }
    if (persist([...next, { name: label, style }])) { selected.value = label; name.value = '' }
  }
  function remove(): void { if (persist(favorites.value.filter(item => item.name !== selected.value))) selected.value = '' }
  return { favorites, name, selected, save, remove }
}
