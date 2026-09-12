<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import SettingsView from './studio/SettingsView.vue'
import TasksView from './studio/TasksView.vue'
import TranslationView from './studio/TranslationView.vue'
import StudioIcon from './studio/StudioIcon.vue'
import { usePageBridge } from './studio/pageBridge'
import type { StudioAction, StudioTab } from './studio/protocol'
import type { PluginSettingsApi } from '../../vue-frontend/src/types/browserExtensionSettings'
const props = defineProps<{ api: PluginSettingsApi }>()
const { state, request: pageRequest, notify } = usePageBridge()
const settingsEditor = ref<InstanceType<typeof SettingsView>>()
const startError = ref('')
async function request<T = void>(action: StudioAction, payload?: unknown): Promise<T> {
  if (['confirm', 'retry-start', 'restart'].includes(action)) {
    startError.value = ''
    if (settingsEditor.value && !(await settingsEditor.value.save())) {
      startError.value = '配置尚未保存，请检查配置中的错误后再开始翻译。'
      selectTab('settings')
      throw new Error(startError.value)
    }
  }
  return pageRequest<T>(action, payload)
}
const currentHash = () =>
  location.hash === '#settings' ? 'settings' : location.hash === '#tasks' ? 'tasks' : 'translate'
const tab = ref<StudioTab>(currentHash())
const settingsVisited = ref(tab.value === 'settings')
const tasksVisited = ref(tab.value === 'tasks')
const visible = ref(!document.hidden)
const tasksActive = computed(
  () => visible.value && (state.value?.open ?? true) && tab.value === 'tasks'
)
const tabs = [
  { id: 'translate', label: '翻译', name: '漫画翻译' },
  { id: 'settings', label: '配置', name: '翻译配置' },
  { id: 'tasks', label: '任务', name: '任务中心' },
] as const
function selectTab(value: StudioTab) {
  if (tab.value === value) return
  tab.value = value
  if (state.value) state.value.tab = value
  document.querySelector('.studio-content')?.scrollTo({ top: 0 })
  if (value === 'settings') settingsVisited.value = true
  if (value === 'tasks') tasksVisited.value = true
  if (location.hash !== `#${value}`) history.replaceState(null, '', `#${value}`)
  notify('tab', value)
}
watch(
  () => state.value?.tab,
  value => {
    if (value) selectTab(value)
  }
)
function tabKey(event: KeyboardEvent, current: number) {
  const next =
    event.key === 'ArrowRight'
      ? (current + 1) % tabs.length
      : event.key === 'ArrowLeft'
        ? (current + tabs.length - 1) % tabs.length
        : event.key === 'Home'
          ? 0
          : event.key === 'End'
            ? tabs.length - 1
            : -1
  if (next < 0) return
  event.preventDefault()
  selectTab(tabs[next]!.id)
  document.querySelectorAll<HTMLButtonElement>('[role=tab]')[next]?.focus()
}
function hashChange() {
  selectTab(currentHash())
}
function visibilityChange() {
  visible.value = !document.hidden
}
function dragStart(event: PointerEvent) {
  if (event.button !== 0 || (event.target as Element).closest('button')) return
  event.preventDefault()
  notify('drag-start', { x: event.screenX, y: event.screenY })
}
onMounted(() => {
  document.addEventListener('visibilitychange', visibilityChange)
  window.addEventListener('hashchange', hashChange)
})
onBeforeUnmount(() => {
  document.removeEventListener('visibilitychange', visibilityChange)
  window.removeEventListener('hashchange', hashChange)
})
</script>
<template>
  <div class="studio">
    <header class="studio-header" @pointerdown="dragStart">
      <div class="brand-mark" aria-hidden="true">S<span>✦</span></div>
      <div class="brand-copy">
        <strong>Saber<span class="brand-label">TRANSLATOR</span></strong
        ><span>漫画阅读，轻一点</span>
      </div>
      <button
        v-if="state"
        class="icon-button close-button"
        aria-label="关闭悬浮窗"
        @click="notify('close')"
      >
        <StudioIcon name="close" />
      </button>
    </header>
    <nav class="studio-nav" aria-label="插件功能" role="tablist">
      <button
        v-for="(item, index) in tabs"
        :key="item.id"
        role="tab"
        :aria-label="item.name"
        :aria-selected="tab === item.id"
        :tabindex="tab === item.id ? 0 : -1"
        :aria-controls="`view-${item.id}`"
        :id="`tab-${item.id}`"
        @keydown="tabKey($event, index)"
        :class="{ active: tab === item.id }"
        @click="selectTab(item.id)"
      >
        <StudioIcon :name="item.id" /><span>{{ item.label }}</span>
      </button>
    </nav>
    <main class="studio-content">
      <div
        id="view-translate"
        role="tabpanel"
        aria-labelledby="tab-translate"
        v-show="tab === 'translate'"
      >
        <TranslationView v-if="state" :state="state" :request="request" />
        <div v-else class="empty-state">
          <span class="empty-icon">✦</span>
          <h3>打开一页漫画</h3>
          <p>从网页上的 Saber 按钮开始翻译</p>
        </div>
      </div>
      <div
        id="view-settings"
        role="tabpanel"
        aria-labelledby="tab-settings"
        v-if="settingsVisited"
        v-show="tab === 'settings'"
      >
        <div v-if="startError" class="notice error" role="alert">{{ startError }}</div>
        <SettingsView
          ref="settingsEditor"
          :api="props.api"
          :active="visible && (state?.open ?? true) && tab === 'settings'"
        />
      </div>
      <div
        id="view-tasks"
        role="tabpanel"
        aria-labelledby="tab-tasks"
        v-if="tasksVisited"
        v-show="tab === 'tasks'"
      >
        <TasksView :api="props.api" :active="tasksActive" />
      </div>
    </main>
    <footer class="studio-footer">
      本机 Saber 处理<span class="footer-separator">·</span
      >页面退出后清理临时数据
    </footer>
  </div>
</template>
