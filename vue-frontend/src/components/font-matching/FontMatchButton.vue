<script setup lang="ts">
import { nextTick, onBeforeUnmount, ref, watch, type CSSProperties } from 'vue'
import { ApiClientError } from '@/api/client'
import { newIdempotencyKey } from '@/api/v2/content'
import { getFontMatch, getFontMatchConfiguration, startFontMatch, type FontMatchConfiguration, type FontMatchStatus } from '@/api/v2/fontMatching'
import { jobsApi, TERMINAL_JOB_STATUSES } from '@/api/v2/jobs'
import OverlayLayer from '@/components/ui/OverlayLayer.vue'
import UiButton from '@/components/ui/UiButton.vue'
import ReferenceImageInput from '@/components/common/ReferenceImageInput.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import { showToast } from '@/utils/toast'

const props = defineProps<{
  fontId: string
  scope: string
  disabled?: boolean
  applyCurrentFont?: boolean
  selectFont: (fontId: string) => (() => void | Promise<void>) | null
}>()
const emit = defineEmits<{ openSettings: [] }>()
const anchor = ref<HTMLElement>()
const popup = ref<HTMLElement>()
const referenceInput = ref<InstanceType<typeof ReferenceImageInput>>()
const referenceReady = ref(false)
const opened = ref(false)
const position = ref<CSSProperties>()
const readiness = ref<FontMatchConfiguration>()
const capturing = ref(false)
const busy = ref(false)
const error = ref('')
const progress = ref('')
const undo = ref<{ fontId: string; restore: () => void | Promise<void> }>()
let epoch = 0
let controller: AbortController | undefined
let timer: ReturnType<typeof setTimeout> | undefined
let jobId: string | undefined
let submission: { image: Blob; key: string } | undefined
let observer: ResizeObserver | undefined

function referenceChanged(ready: boolean): void {
  referenceReady.value = ready
  submission = undefined
  if (readiness.value?.ready) error.value = ''
}

function cancelJob(id: string): void {
  void jobsApi.cancel(id).catch(() => showToast('后台字体匹配取消失败，可在任务中心重试', 'warning'))
}

function close(restoreFocus = false): void {
  epoch++
  controller?.abort()
  controller = undefined
  clearTimeout(timer)
  if (jobId) cancelJob(jobId)
  jobId = undefined
  referenceChanged(false)
  capturing.value = busy.value = opened.value = false
  readiness.value = undefined
  error.value = progress.value = ''
  observer?.disconnect()
  document.removeEventListener('pointerdown', outside, true)
  window.removeEventListener('scroll', reposition, true)
  window.removeEventListener('resize', reposition)
  if (restoreFocus) anchor.value?.querySelector('button')?.focus({ preventScroll: true })
}

function outside(event: PointerEvent): void {
  const target = event.target
  if (target instanceof Node && !popup.value?.contains(target) && !anchor.value?.contains(target)) close()
}

function reposition(): void {
  const rect = anchor.value?.getBoundingClientRect()
  if (!rect?.width || !popup.value) return
  const gap = 8
  const width = Math.min(340, window.innerWidth - gap * 2)
  const available = window.innerHeight - gap
  const height = Math.min(popup.value.scrollHeight + 2, available - gap)
  const above = rect.bottom + gap + height > available && rect.top > window.innerHeight - rect.bottom
  const top = above ? rect.top - gap - height : rect.bottom + gap
  position.value = {
    left: Math.max(gap, Math.min(rect.right - width, window.innerWidth - width - gap)) + 'px',
    top: Math.max(gap, Math.min(top, available - height)) + 'px',
    width: width + 'px', maxHeight: available - gap + 'px',
  }
}

async function open(): Promise<void> {
  if (opened.value) { close(true); return }
  if (props.disabled) return
  const id = ++epoch
  const request = controller = new AbortController()
  position.value = { visibility: 'hidden' }
  opened.value = true
  await nextTick()
  if (id !== epoch || !popup.value) return
  reposition()
  popup.value.focus({ preventScroll: true })
  observer = new ResizeObserver(reposition)
  observer.observe(popup.value)
  document.addEventListener('pointerdown', outside, true)
  window.addEventListener('scroll', reposition, true)
  window.addEventListener('resize', reposition)
  try {
    const result = await getFontMatchConfiguration(request.signal)
    if (id === epoch) readiness.value = result
  } catch {
    if (id === epoch) error.value = '无法读取字体匹配设置，请关闭后重试'
  }
}

const phases = { samples: '准备字体样张', loading_model: '加载本地模型', style: '比较全库风格', attributes: '评估风格属性' }
function finish(match: NonNullable<FontMatchStatus['result']>): void {
  const current = props.fontId
  close(true)
  if (match.fontId === current && !props.applyCurrentFont) {
    showToast(match.method === 'm7' ? '已匹配到当前字体' : '当前字体整体风格最接近，参考属性评估未完成', 'success')
    return
  }
  const restore = props.selectFont(match.fontId)
  if (!restore) { showToast('匹配字体已不可用，请刷新字体库后重试', 'warning'); return }
  undo.value = { fontId: match.fontId, restore }
  showToast(match.method === 'm7' ? '已应用匹配字体' : '已按整体风格应用字体，参考属性评估未完成', 'success')
}

function isTemporaryError(cause: unknown): boolean {
  return cause instanceof ApiClientError && (
    cause.status === 0 || cause.status === 408 || cause.status === 429 || cause.status >= 500
  )
}

async function poll(id: number, accepted: string, retryMs = 1000): Promise<void> {
  try {
    const state = await getFontMatch(accepted, controller?.signal)
    if (id !== epoch) return
    if (!TERMINAL_JOB_STATUSES.has(state.status) && state.status !== 'interrupted') {
      progress.value = state.progress
        ? phases[state.progress.phase] + ' ' + state.progress.done + '/' + state.progress.total
        : state.status === 'paused' ? '任务已暂停' : '正在等待匹配'
      timer = setTimeout(() => void poll(id, accepted), 900)
      return
    }
    jobId = undefined
    if (state.status === 'completed' && state.result) { finish(state.result); return }
    busy.value = false
    error.value = state.error || (state.status === 'cancelled' ? '匹配已取消' : '匹配未完成，请重试')
  } catch (cause) {
    if (id !== epoch) return
    if (isTemporaryError(cause)) {
      progress.value = '连接中断，正在重新查询匹配进度…'
      timer = setTimeout(() => void poll(id, accepted, Math.min(retryMs * 2, 8000)), retryMs)
      return
    }
    jobId = undefined
    busy.value = false
    error.value = cause instanceof Error ? cause.message : '无法读取匹配进度'
  }
}

async function start(): Promise<void> {
  if (busy.value || capturing.value || !readiness.value?.ready || !referenceReady.value || !referenceInput.value) return
  const id = epoch
  popup.value?.focus({ preventScroll: true })
  busy.value = true
  error.value = ''
  progress.value = '正在提交匹配'
  try {
    if (!submission) {
      const image = await referenceInput.value.read()
      if (id !== epoch) return
      submission = { image, key: newIdempotencyKey() }
    }
    // If the control closes in transit, still retrieve the job ID so it can be cancelled.
    const [accepted] = (await startFontMatch(submission.image, submission.key)).jobIds
    if (!accepted) throw new Error('后台未接受匹配任务')
    if (id !== epoch) { cancelJob(accepted); return }
    submission = undefined
    jobId = accepted
    void poll(id, accepted)
  } catch (cause) {
    if (id === epoch) {
      busy.value = false
      if (submission && isTemporaryError(cause)) {
        error.value = '暂未收到提交确认，再次点击「开始匹配」会继续本次提交'
      } else {
        submission = undefined
        error.value = cause instanceof Error ? cause.message : '无法开始匹配'
      }
    }
  }
}

function goToSettings(): void {
  close()
  emit('openSettings')
}

function restoreFont(): void {
  const action = undo.value
  undo.value = undefined
  if (action && action.fontId === props.fontId && !props.disabled) {
    close()
    void action.restore()
  }
}

watch(() => [props.scope, props.disabled] as const, () => { undo.value = undefined; close() }, { flush: 'sync' })
watch(() => props.fontId, fontId => {
  if (undo.value?.fontId !== fontId) undo.value = undefined
  if (opened.value) close()
}, { flush: 'sync' })
onBeforeUnmount(() => close())
</script>

<template>
  <span ref="anchor" class="font-match-button">
    <UiButton
      class="font-match-button__trigger"
      size="xs"
      variant="secondary"
      :disabled="disabled"
      aria-haspopup="dialog"
      :aria-expanded="opened"
      @click="open"
    >匹配</UiButton>
    <UiIconButton
      v-if="undo?.fontId === fontId"
      size="xs" variant="plain" label="撤销本次选择" title="撤销本次选择"
      :disabled="disabled" @click="restoreFont"
    ><UiIcon name="rotate-ccw" size="14" /></UiIconButton>
  </span>
  <Teleport v-if="opened" to="body">
    <OverlayLayer level="popover" passthrough>
      <div
        ref="popup" class="font-match-popup" role="dialog" aria-label="以图匹配字体" tabindex="-1"
        :style="position" @keydown.esc.prevent.stop="close(true)"
      >
        <div class="font-match-popup__row">
          <strong>以图匹配字体</strong>
          <UiButton size="sm" variant="ghost" @click="close(true)">{{ busy ? '取消' : '关闭' }}</UiButton>
        </div>
        <ReferenceImageInput ref="referenceInput" :disabled="busy" @change="referenceChanged" @busy="capturing = $event" />
        <p v-if="error" class="font-match-popup__error" role="alert">{{ error }}</p>
        <p v-if="!readiness && !error" class="font-match-popup__hint" role="status">正在读取设置…</p>
        <div v-if="readiness && !readiness.ready" class="font-match-popup__row">
          <span class="font-match-popup__hint">{{ readiness.message }}</span>
          <UiButton size="sm" variant="ghost" @click="goToSettings">打开设置</UiButton>
        </div>
        <p v-if="busy" class="font-match-popup__hint" role="status">{{ progress }}</p>
        <UiButton v-else size="sm" variant="primary" :disabled="!readiness?.ready || !referenceReady || capturing" @click="start">
          开始匹配
        </UiButton>
      </div>
    </OverlayLayer>
  </Teleport>
</template>

<style scoped>
.font-match-button {
  display: inline-flex;
  align-items: center;
  flex: none;
  gap: 2px;
  white-space: nowrap;
}

.font-match-button__trigger {
  --ui-button-xs-padding: 2px 8px;

  min-height: 24px;
}

.font-match-popup {
  position: absolute;
  box-sizing: border-box;
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: 9px;
  padding: 12px;
  border: 1px solid var(--color-border-muted);
  border-radius: 12px;
  background: var(--color-surface-base);
  box-shadow: var(--shadow-lg);
  color: var(--color-text-default);
}

.font-match-popup__row { display: flex; align-items: center; justify-content: space-between; gap: 6px; }
.font-match-popup__hint, .font-match-popup__error { margin: 0; font-size: 12px; line-height: 1.5; }
.font-match-popup__hint { color: var(--color-text-supporting); }
.font-match-popup__error { color: var(--color-status-error); }
</style>
