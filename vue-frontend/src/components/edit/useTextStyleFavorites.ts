import { computed, inject, provide, ref, watch, type InjectionKey } from 'vue'
import type { LayerStyleRecipe } from '@/api/v2/content'
import { useAuthStore } from '@/stores/authStore'
import { showToast } from '@/utils/toast'
import type { TextStyleTemplate } from './textStyleClipboard'

type Favorite = { name: string } & ({ style: TextStyleTemplate; recipe?: never } | { recipe: LayerStyleRecipe; style?: never })
const favoritesKey: InjectionKey<ReturnType<typeof createTextStyleFavorites>> = Symbol('textStyleFavorites')

/** Browser preferences only. Applying a favorite still commits normal page data. */
function createTextStyleFavorites() {
  const auth = useAuthStore()
  const favorites = ref<Favorite[]>([])
  const name = ref('')
  const key = computed(() => `saber.text-styles.${auth.user?.id ?? 'local'}`)
  watch(key, value => {
    try {
      const items: unknown = JSON.parse(localStorage.getItem(value) ?? '[]')
      favorites.value = Array.isArray(items) ? items.filter(item => item
        && typeof item.name === 'string' && item.name.length <= 30
        && ((item.style?.character && typeof item.style.character === 'object' && item.style.box && typeof item.style.box === 'object')
          || (item.recipe && typeof item.recipe.fillOpacity === 'number' && Array.isArray(item.recipe.effects)
            && item.recipe.effects.length >= 1 && item.recipe.effects.length <= 8))) : []
    } catch { favorites.value = [] }
  }, { immediate: true })
  function persist(next: Favorite[]): boolean {
    try { localStorage.setItem(key.value, JSON.stringify(next)); favorites.value = next; return true }
    catch { showToast('无法保存样式收藏，请检查浏览器存储空间', 'error'); return false }
  }
  function save(style: TextStyleTemplate): string | undefined {
    const label = name.value.trim().slice(0, 30)
    if (!label) { showToast('请为样式命名，例如：对白、旁白、注释', 'warning'); return }
    const next = favorites.value.filter(item => item.recipe || item.name !== label)
    if (next.filter(item => item.style).length >= 12) { showToast('最多收藏 12 个排版，请先删除不用的收藏', 'warning'); return }
    if (persist([...next, { name: label, style }])) {
      name.value = ''
      return label
    }
  }
  function saveRecipe(label: string, recipe: LayerStyleRecipe): boolean {
    const clean = label.trim().slice(0, 30)
    if (!clean) return false
    return persist([...favorites.value.filter(item => !item.recipe || item.name !== clean), { name: clean, recipe }])
  }
  function remove(label: string, recipe = false): boolean {
    return persist(favorites.value.filter(item => item.name !== label || Boolean(item.recipe) !== recipe))
  }
  return { favorites, name, save, saveRecipe, remove }
}

export function provideTextStyleFavorites(): void {
  provide(favoritesKey, createTextStyleFavorites())
}

export function useTextStyleFavorites() {
  return inject(favoritesKey, null) ?? createTextStyleFavorites()
}
