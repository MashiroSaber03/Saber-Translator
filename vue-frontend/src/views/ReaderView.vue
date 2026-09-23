<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, watch, type ComponentPublicInstance } from 'vue'
import AppShell from '@/components/ui/AppShell.vue'
import { useRouter } from 'vue-router'
import { getBook, listChapterPages, type V2BookDetail, type V2PageSummary } from '@/api/v2/content'
import { useToast } from '@/utils/toast'
import { useAuthStore } from '@/stores/authStore'
import { usePublicUserAccess } from '@/composables/usePublicUserAccess'
import { useDialogLifecycle } from '@/composables/useDialogLifecycle'
import UiButton from '@/components/ui/UiButton.vue'
import ReaderCanvas from '@/components/reader/ReaderCanvas.vue'
import ReaderControls from '@/components/reader/ReaderControls.vue'
import ReaderProgress from '@/components/reader/ReaderProgress.vue'
import {
  DEFAULT_READER_SETTINGS,
  READER_FITS,
  loadReaderSettings,
  saveReaderSettings,
  type ReaderSettings,
} from '@/components/reader/readerSettings'
import { pageGroups, resolvePosition, type ReaderPosition } from '@/components/reader/readerLayout'
import {
  loadReaderRecord,
  saveReaderRecord,
  readerHistoryKey,
} from '@/components/reader/readerHistory'

const props = defineProps<{ bookId: string; chapterId: string }>()
const router = useRouter()
const toast = useToast()
const auth = useAuthStore()
const access = usePublicUserAccess()
const shell = ref<ComponentPublicInstance | null>(null)
const root = computed(() => shell.value?.$el as HTMLElement | undefined)
const drawer = ref<HTMLElement | null>(null)
const book = ref<V2BookDetail | null>(null)
const pages = ref<V2PageSummary[]>([])
const sizes = ref<Record<string, { width: number; height: number }>>({})
const images = computed(() =>
  pages.value.map(page => (sizes.value[page.id] ? { ...page, ...sizes.value[page.id] } : page))
)
const loading = ref(true)
const error = ref('')
const viewMode = ref<'original' | 'translated'>('translated')
const settings = ref<ReaderSettings>(
  loadReaderSettings() ?? { ...DEFAULT_READER_SETTINGS, fits: { ...DEFAULT_READER_SETTINGS.fits } }
)
const position = ref<ReaderPosition>({ pageId: '', index: 0, fraction: 0 })
const navigationId = ref(0)
const offset = ref(false)
const controls = ref(true)
const narrow = ref(false)
const fullscreen = ref(false)
let beforeFullscreen = true
let media: MediaQueryList | undefined
let mounted = false
let sequence = 0
let recordKey = ''
let saveTimer: ReturnType<typeof setTimeout> | undefined
const chapters = computed(() => book.value?.chapters ?? [])
const chapterIndex = computed(() => chapters.value.findIndex(c => c.id === props.chapterId))
const chapterTitle = computed(() => chapters.value[chapterIndex.value]?.title ?? '')
const paged = computed(
  () => settings.value.layout === 'single' || settings.value.layout === 'double'
)
const groups = computed(() =>
  pageGroups(images.value, settings.value.layout === 'double', offset.value)
)
const currentIndex = computed(() => resolvePosition(images.value, position.value))
const groupIndex = computed(() =>
  Math.max(
    0,
    groups.value.findIndex(group => group.includes(currentIndex.value))
  )
)
const group = computed(() => groups.value[groupIndex.value] ?? [])
const neighbours = computed(() => [
  ...(groups.value[groupIndex.value - 1] ?? []),
  ...(groups.value[groupIndex.value + 1] ?? []),
])
const first = computed(() =>
  images.value.length ? (paged.value ? (group.value[0] ?? 0) : currentIndex.value) + 1 : 0
)
const last = computed(() => (paged.value ? (group.value.at(-1) ?? -1) + 1 : first.value))
const range = computed(() =>
  first.value === last.value ? String(first.value) : `${first.value}–${last.value}`
)
const canPrev = computed(() => first.value > 1)
const canNext = computed(() => last.value < images.value.length)
const hasPrevChapter = computed(() => chapterIndex.value > 0)
const hasNextChapter = computed(
  () => chapterIndex.value >= 0 && chapterIndex.value < chapters.value.length - 1
)
const modal = computed(() => narrow.value && controls.value)
useDialogLifecycle({
  open: modal,
  container: drawer,
  close: () => {
    controls.value = false
  },
})

function persist() {
  clearTimeout(saveTimer)
  if (recordKey && pages.value.length && position.value.pageId)
    saveReaderRecord({ key: recordKey, ...position.value, offset: offset.value })
}
function scheduleSave() {
  clearTimeout(saveTimer)
  saveTimer = setTimeout(persist, 250)
}
function setPosition(value: ReaderPosition) {
  position.value = value
  scheduleSave()
}
function jump(page: number) {
  if (!images.value.length || !Number.isFinite(page)) return
  const index = Math.max(0, Math.min(images.value.length - 1, Math.trunc(page) - 1))
  setPosition({ pageId: images.value[index]!.id, index, fraction: 0 })
  navigationId.value++
}
function navigate(delta: number) {
  if (paged.value) {
    const next = groups.value[groupIndex.value + delta]
    if (next) jump(next[0]! + 1)
  } else {
    const next = currentIndex.value + delta
    if (next >= 0 && next < images.value.length) jump(next + 1)
  }
}
function updateSettings(value: ReaderSettings) {
  settings.value = value
  saveReaderSettings(value)
}
function toggleOffset() {
  offset.value = !offset.value
  scheduleSave()
}
function restart() {
  offset.value = false
  jump(1)
}
function cycleFit() {
  const layout = settings.value.layout
  const next =
    READER_FITS[
      (READER_FITS.findIndex(f => f.value === settings.value.fits[layout]) + 1) % READER_FITS.length
    ]!.value
  updateSettings({ ...settings.value, fits: { ...settings.value.fits, [layout]: next } })
}
function learnedSize(id: string, width: number, height: number) {
  const page = pages.value.find(p => p.id === id)
  if (page && (!page.width || !page.height) && !sizes.value[id] && width > 0 && height > 0)
    sizes.value = { ...sizes.value, [id]: { width, height } }
}
async function load() {
  persist()
  const request = ++sequence
  const bookId = props.bookId,
    chapterId = props.chapterId
  recordKey = ''
  loading.value = true
  error.value = ''
  pages.value = []
  sizes.value = {}
  book.value = null
  try {
    const [detail, result] = await Promise.all([
      getBook(bookId),
      listChapterPages(chapterId, { all: true }),
    ])
    if (!mounted || request !== sequence) return
    if (!detail.chapters.some(c => c.id === chapterId)) throw Error('章节不属于当前书籍')
    if (result.nextCursor !== null || result.items.some(p => p.chapterId !== chapterId))
      throw Error('章节页面列表不完整或归属不一致')
    recordKey = readerHistoryKey(auth.user?.id ?? 'local', bookId, chapterId)
    const record = loadReaderRecord(recordKey)
    book.value = detail
    pages.value = result.items
    offset.value = record?.offset ?? false
    const index = record ? resolvePosition(result.items, record) : 0
    position.value = {
      pageId: result.items[index]?.id ?? '',
      index,
      fraction: record?.fraction ?? 0,
    }
    navigationId.value++
    document.title = `${chapterTitle.value} - ${detail.title}`
  } catch (e) {
    if (!mounted || request !== sequence) return
    error.value = e instanceof Error ? e.message : '未知错误'
    toast.error(`加载失败: ${error.value}`)
  } finally {
    if (mounted && request === sequence) loading.value = false
  }
}
function chapter(id: string) {
  if (id === props.chapterId || !chapters.value.some(c => c.id === id)) return
  persist()
  void router.push({ path: '/reader', query: { book: props.bookId, chapter: id } })
}
function translate() {
  persist()
  void router.push({ path: '/translate', query: { book: props.bookId, chapter: props.chapterId } })
}
async function toggleFullscreen() {
  try {
    if (document.fullscreenElement) await document.exitFullscreen()
    else if (root.value?.requestFullscreen) {
      beforeFullscreen = controls.value
      await root.value.requestFullscreen()
    } else toast.error('当前浏览器不支持全屏阅读')
  } catch {
    toast.error('无法进入全屏，请通过全屏按钮重试')
  }
}
function fullscreenChanged() {
  const active = document.fullscreenElement === root.value
  if (active === fullscreen.value) return
  fullscreen.value = active
  controls.value = active ? false : beforeFullscreen
}
function keydown(event: KeyboardEvent) {
  if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || event.isComposing)
    return
  const target = event.target
  if (
    target instanceof Element &&
    target.closest(
      'input,textarea,select,[contenteditable="true"],[role="combobox"],.reader-controls'
    )
  )
    return
  if (event.key.toLowerCase() === 'm') {
    event.preventDefault()
    controls.value = !controls.value
    return
  }
  if (modal.value) return
  switch (event.key.toLowerCase()) {
    case 'arrowleft':
      event.preventDefault()
      navigate(settings.value.direction === 'rtl' ? 1 : -1)
      break
    case 'arrowright':
      event.preventDefault()
      navigate(settings.value.direction === 'rtl' ? -1 : 1)
      break
    case 'home':
      event.preventDefault()
      jump(1)
      break
    case 'end':
      event.preventDefault()
      jump(images.value.length)
      break
    case 'f':
      event.preventDefault()
      if (!event.repeat) void toggleFullscreen()
      break
    case 'i':
      event.preventDefault()
      cycleFit()
      break
    case 'o':
      if (settings.value.layout === 'double') {
        event.preventDefault()
        toggleOffset()
      }
      break
    case 'escape':
      controls.value = false
      break
  }
}
function mediaChanged() {
  narrow.value = media?.matches ?? false
}
function visibilityChanged() {
  if (document.visibilityState === 'hidden') persist()
}
onMounted(() => {
  mounted = true
  media = window.matchMedia('(max-width: 768px)')
  mediaChanged()
  media.addEventListener('change', mediaChanged)
  document.addEventListener('keydown', keydown)
  document.addEventListener('fullscreenchange', fullscreenChanged)
  document.addEventListener('visibilitychange', visibilityChanged)
  window.addEventListener('pagehide', persist)
  void load()
})
onBeforeUnmount(() => {
  persist()
  mounted = false
  sequence++
  media?.removeEventListener('change', mediaChanged)
  document.removeEventListener('keydown', keydown)
  document.removeEventListener('fullscreenchange', fullscreenChanged)
  document.removeEventListener('visibilitychange', visibilityChanged)
  window.removeEventListener('pagehide', persist)
  if (document.fullscreenElement === root.value) void document.exitFullscreen().catch(() => {})
})
watch(
  () => [props.bookId, props.chapterId, auth.user?.id],
  () => {
    if (mounted) void load()
  }
)
</script>

<template>
  <AppShell
    ref="shell"
    class="reader-page"
    viewport-mode="immersive"
    :style="{ background: settings.bgColor }"
  >
    <header v-if="controls" class="reader-header">
      <UiButton variant="inverse" size="sm" @click="router.push('/')">← 书架</UiButton>
      <div class="reader-header__title">
        <strong>{{ book?.title || '阅读器' }}</strong><span>{{ chapterTitle }}</span>
      </div>
      <span class="reader-header__page-info">{{ range }} / {{ images.length }}</span>
      <div class="reader-header__modes">
        <UiButton
          v-for="mode in ['original', 'translated'] as const"
          :key="mode"
          size="sm"
          :variant="viewMode === mode ? 'primary' : 'inverse'"
          :aria-pressed="viewMode === mode"
          :data-mode="mode"
          @click="viewMode = mode"
        >
          {{ mode === 'original' ? '原图' : '译图' }}
        </UiButton>
      </div>
      <UiButton variant="inverse" size="sm" @click="toggleFullscreen">
        {{
          fullscreen ? '退出全屏' : '全屏'
        }}
      </UiButton>
      <UiButton
        v-if="access.featureAllowed('translation')"
        class="reader-header__translate"
        variant="inverse"
        size="sm"
        @click="translate"
      >
        编辑翻译
      </UiButton>
      <UiButton variant="inverse" size="sm" aria-label="收起菜单" @click="controls = false">
        收起
      </UiButton>
    </header>
    <UiButton
      v-else
      class="reader-menu-trigger"
      variant="inverse"
      size="sm"
      aria-label="打开阅读菜单"
      @click="controls = true"
    >
      ☰ 菜单
    </UiButton>
    <div class="reader-workspace">
      <div class="reader-stage">
        <div v-if="error" class="reader-error" role="alert">
          <p>{{ error }}</p>
          <UiButton @click="load">重新加载</UiButton>
        </div>
        <ReaderCanvas
          v-else
          :images="images"
          :is-loading="loading"
          :view-mode="viewMode"
          :settings="settings"
          :position="position"
          :navigation-id="navigationId"
          :group="group"
          :neighbours="neighbours"
          :can-prev="canPrev"
          :can-next="canNext"
          @position-change="setPosition"
          @go-translate="translate"
          @toggle-controls="controls = !controls"
          @navigate="navigate"
          @size="learnedSize"
        />
        <div
          v-if="!loading && images.length && (!canPrev || !canNext)"
          class="reader-boundary"
        >
          <UiButton
            v-if="!canPrev && hasPrevChapter"
            variant="inverse"
            size="sm"
            @click="chapter(chapters[chapterIndex - 1]!.id)"
          >
            上一章
          </UiButton>
          <template v-if="!canNext">
            <span>本章已读完</span><UiButton
              v-if="hasNextChapter"
              variant="inverse"
              size="sm"
              @click="chapter(chapters[chapterIndex + 1]!.id)"
            >
              下一章
            </UiButton>
          </template>
        </div>
        <ReaderProgress
          :page="first"
          :end="last"
          :total="images.length"
          :direction="settings.direction"
          :mode="settings.progress"
          :expanded="controls"
          @jump="jump"
        />
      </div>
      <div v-if="modal" class="reader-drawer-backdrop" @click="controls = false"></div>
      <div
        v-if="controls"
        ref="drawer"
        class="reader-drawer"
        :role="narrow ? 'dialog' : undefined"
        :aria-modal="narrow ? true : undefined"
        aria-label="阅读菜单"
        tabindex="-1"
      >
        <ReaderControls
          :settings="settings"
          :page="currentIndex + 1"
          :total="images.length"
          :range="range"
          :offset="offset"
          :can-prev="canPrev"
          :can-next="canNext"
          :has-prev-chapter="hasPrevChapter"
          :has-next-chapter="hasNextChapter"
          :chapter-id="chapterId"
          :chapters="chapters"
          @settings-change="updateSettings"
          @navigate="navigate"
          @jump="jump"
          @chapter="chapter"
          @offset="toggleOffset"
          @restart="restart"
          @close="controls = false"
        />
      </div>
    </div>
  </AppShell>
</template>

<style scoped>
.reader-page {
  color: var(--color-text-inverse);
}

.reader-header {
  display: flex;
  gap: 10px;
  align-items: center;
  min-height: 56px;
  padding: 8px 16px;
  background: var(--color-surface-inverse-raised);
  border-bottom: 1px solid var(--color-overlay-inverse-subtle);
}

.reader-header__title {
  display: flex;
  flex: 1;
  min-width: 0;
  gap: 12px;
  align-items: baseline;
}

.reader-header__title strong,
.reader-header__title span {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.reader-header__title span {
  font-size: 12px;
  opacity: 0.7;
}

.reader-header__modes {
  display: flex;
  gap: 3px;
}

.reader-header__page-info {
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.reader-workspace {
  display: flex;
  position: relative;
  flex: 1;
  min-height: 0;
}

.reader-stage {
  flex: 1;
  min-width: 0;
  min-height: 0;
  position: relative;
  padding-bottom: 32px;
}

.reader-drawer {
  display: flex;
  min-height: 0;
  outline: none;
}

.reader-menu-trigger {
  position: absolute;
  top: 12px;
  right: 16px;
  z-index: var(--z-local-overlay);
  opacity: 0.75;
  background: var(--color-surface-inverse-raised);
}

.reader-menu-trigger:hover,
.reader-menu-trigger:focus-visible {
  opacity: 1;
}

.reader-boundary {
  position: absolute;
  bottom: 48px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 12px;
  align-items: center;
  font-size: 12px;
  padding: 6px 12px;
  border-radius: 8px;
  background: var(--color-overlay-scrim);
}

.reader-boundary:empty {
  display: none;
}

.reader-error {
  display: flex;
  height: 100%;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 16px;
}

.reader-drawer-backdrop {
  position: absolute;
  inset: 0;
  background: var(--color-overlay-scrim);
  z-index: var(--z-local-overlay);
}

@media (--breakpoint-md-down) {
  .reader-drawer {
    width: min(320px, 88vw);
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    z-index: var(--z-local-panel);
    box-shadow: -12px 0 40px var(--color-overlay-scrim);
  }

  .reader-header {
    gap: 6px;
    padding: 8px;
    flex-wrap: wrap;
  }

  .reader-header__title {
    order: -1;
    flex-basis: 100%;
  }

  .reader-header__translate {
    display: none;
  }

  .reader-header__page-info {
    margin-right: auto;
  }
}
</style>
