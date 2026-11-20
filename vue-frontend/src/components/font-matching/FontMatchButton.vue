<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch, type CSSProperties } from 'vue'
import { ApiClientError } from '@/api/client'
import { newIdempotencyKey } from '@/api/v2/content'
import { getFontMatch, getFontMatchConfiguration, startFontMatch, type FontMatchConfiguration, type FontMatchStatus } from '@/api/v2/fontMatching'
import { jobsApi, TERMINAL_JOB_STATUSES } from '@/api/v2/jobs'
import OverlayLayer from '@/components/ui/OverlayLayer.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
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
type Region = { x: number; y: number; w: number; h: number }
const fullImage: Region = { x: 0, y: 0, w: 1, h: 1 }
const anchor = ref<HTMLElement>()
const popup = ref<HTMLElement>()
const preview = ref<HTMLImageElement>()
const upload = ref<InstanceType<typeof UiFileInput>>()
const opened = ref(false)
const position = ref<CSSProperties>()
const readiness = ref<FontMatchConfiguration>()
const reference = ref('')
const region = ref<Region>()
const capturing = ref(false)
const busy = ref(false)
const error = ref('')
const progress = ref('')
const undo = ref<{ fontId: string; restore: () => void | Promise<void> }>()
const regionStyle = computed(() => region.value && ({
  left: region.value.x * 100 + '%', top: region.value.y * 100 + '%',
  width: region.value.w * 100 + '%', height: region.value.h * 100 + '%',
}))
let epoch = 0
let controller: AbortController | undefined
let timer: ReturnType<typeof setTimeout> | undefined
let stream: MediaStream | undefined
let jobId: string | undefined
let submission: { image: Blob; key: string } | undefined
let drag: { x: number; y: number } | undefined
let observer: ResizeObserver | undefined

function stopCapture(): void {
  stream?.getTracks().forEach(track => track.stop())
  stream = undefined
}

function releaseReference(): void {
  if (reference.value) URL.revokeObjectURL(reference.value)
  reference.value = ''
  region.value = undefined
  drag = undefined
  submission = undefined
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
  stopCapture()
  releaseReference()
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

function replaceReference(file: Blob, croppedByUser: boolean): void {
  releaseReference()
  error.value = ''
  reference.value = URL.createObjectURL(file)
  region.value = croppedByUser ? undefined : { ...fullImage }
}

function uploadReference(files: File[]): void {
  const [file] = files
  upload.value?.clear()
  if (file && opened.value) replaceReference(file, false)
}

function toPng(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => canvas.toBlob(blob => {
    if (blob) resolve(blob)
    else reject(new Error('无法读取参考图片'))
  }, 'image/png'))
}

async function captureScreen(): Promise<void> {
  if (!navigator.mediaDevices?.getDisplayMedia) {
    error.value = '当前浏览器或页面地址不支持截图，请上传图片'
    return
  }
  const id = epoch
  popup.value?.focus({ preventScroll: true })
  capturing.value = true
  error.value = ''
  let video: HTMLVideoElement | undefined
  try {
    const capture = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false })
    if (id !== epoch) { capture.getTracks().forEach(track => track.stop()); return }
    stream = capture
    video = document.createElement('video')
    video.muted = true
    video.playsInline = true
    video.srcObject = capture
    await video.play()
    if (id !== epoch) return
    if (!video.videoWidth || !video.videoHeight) throw new Error('empty frame')
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const context = canvas.getContext('2d')
    if (!context) throw new Error('canvas unavailable')
    context.drawImage(video, 0, 0)
    const image = await toPng(canvas)
    if (id === epoch) replaceReference(image, true)
  } catch (cause) {
    if (id === epoch) error.value = cause instanceof DOMException && cause.name === 'NotAllowedError'
      ? '截图已取消或授权被拒绝，可以上传图片'
      : '当前无法截图，请上传图片'
  } finally {
    if (video) { video.pause(); video.srcObject = null }
    if (id === epoch) { stopCapture(); capturing.value = false }
  }
}

function pointer(event: PointerEvent): { x: number; y: number } {
  const box = (event.currentTarget as HTMLElement).getBoundingClientRect()
  const clamp = (coordinate: number) => Math.max(0, Math.min(1, coordinate))
  return { x: clamp((event.clientX - box.left) / box.width), y: clamp((event.clientY - box.top) / box.height) }
}

function beginCrop(event: PointerEvent): void {
  if (busy.value || capturing.value || event.button !== 0) return
  const target = event.currentTarget as HTMLElement
  target.setPointerCapture(event.pointerId)
  drag = pointer(event)
  region.value = undefined
  submission = undefined
}

function moveCrop(event: PointerEvent): void {
  if (!drag) return
  const { x, y } = pointer(event)
  region.value = { x: Math.min(x, drag.x), y: Math.min(y, drag.y), w: Math.abs(x - drag.x), h: Math.abs(y - drag.y) }
}

function endCrop(event: PointerEvent): void {
  moveCrop(event)
  drag = undefined
  const selected = region.value
  const image = preview.value
  if (!selected || !image || selected.w * image.naturalWidth < 2 || selected.h * image.naturalHeight < 2) {
    region.value = undefined
  }
}

async function chosenImage(): Promise<Blob> {
  const image = preview.value
  const area = region.value
  if (!image || !area) throw new Error('请框选参考文字')
  await image.decode()
  const left = Math.floor(area.x * image.naturalWidth)
  const top = Math.floor(area.y * image.naturalHeight)
  const width = Math.ceil((area.x + area.w) * image.naturalWidth) - left
  const height = Math.ceil((area.y + area.h) * image.naturalHeight) - top
  if (width < 2 || height < 2 || width * height > 20_000_000) throw new Error('请选择不超过 2000 万像素的文字区域')
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  if (!context) throw new Error('无法读取文字选区')
  context.drawImage(image, left, top, width, height, 0, 0, width, height)
  const blob = await toPng(canvas)
  if (blob.size > 12 * 1024 ** 2) throw new Error('文字选区超过 12 MiB，请缩小选区')
  return blob
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
  if (busy.value || capturing.value || !readiness.value?.ready || !region.value || drag) return
  const id = epoch
  popup.value?.focus({ preventScroll: true })
  busy.value = true
  error.value = ''
  progress.value = '正在提交匹配'
  try {
    if (!submission) {
      const image = await chosenImage()
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
        <div class="font-match-popup__reference">
          <div
            v-if="reference" class="font-match-popup__image"
            @pointerdown.prevent="beginCrop" @pointermove="moveCrop" @pointerup="endCrop"
            @pointercancel="drag = undefined; region = undefined"
          >
            <img
              ref="preview" :src="reference" alt="参考图片" draggable="false"
              @error="releaseReference(); error = '无法读取参考图片，请重新选取'"
            >
            <span v-if="regionStyle" class="font-match-popup__region" :style="regionStyle" />
          </div>
          <span v-else>选择一张参考文字图片</span>
        </div>
        <p v-if="reference" class="font-match-popup__hint">{{ region ? '可拖动重新框选文字' : '请拖动框选参考文字' }}</p>
        <div v-if="!busy" class="font-match-popup__row font-match-popup__input">
          <UiFileInput ref="upload" accept="image/png,image/jpeg,image/webp,image/bmp,image/gif" hidden @files-change="uploadReference" />
          <UiButton size="sm" variant="secondary" :disabled="capturing" @click="upload?.click()">{{ reference ? '更换图片' : '上传图片' }}</UiButton>
          <UiButton size="sm" variant="secondary" :disabled="capturing" @click="captureScreen">{{ capturing ? '正在截图…' : '截图' }}</UiButton>
        </div>
        <p v-if="error" class="font-match-popup__error" role="alert">{{ error }}</p>
        <p v-if="!readiness && !error" class="font-match-popup__hint" role="status">正在读取设置…</p>
        <div v-if="readiness && !readiness.ready" class="font-match-popup__row">
          <span class="font-match-popup__hint">{{ readiness.message }}</span>
          <UiButton size="sm" variant="ghost" @click="goToSettings">打开设置</UiButton>
        </div>
        <p v-if="busy" class="font-match-popup__hint" role="status">{{ progress }}</p>
        <UiButton v-else size="sm" variant="primary" :disabled="!readiness?.ready || !region || capturing" @click="start">
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
.font-match-popup__input { justify-content: flex-start; }

.font-match-popup__reference {
  display: grid;
  place-items: center;
  min-height: 82px;
  padding: 4px;
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  background: var(--color-surface-subtle);
  color: var(--color-text-supporting);
  font-size: 13px;
}
.font-match-popup__image { position: relative; overflow: hidden; cursor: crosshair; touch-action: none; }
.font-match-popup__image img { display: block; max-width: 100%; max-height: min(42dvh, 280px); }

.font-match-popup__region {
  position: absolute;
  box-sizing: border-box;
  border: 2px solid var(--color-action-brand);
  box-shadow: 0 0 0 999px var(--color-overlay-scrim-subtle);
  pointer-events: none;
}
.font-match-popup__hint, .font-match-popup__error { margin: 0; font-size: 12px; line-height: 1.5; }
.font-match-popup__hint { color: var(--color-text-supporting); }
.font-match-popup__error { color: var(--color-status-error); }
</style>
