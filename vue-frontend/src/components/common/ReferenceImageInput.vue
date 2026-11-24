<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'

defineProps<{ disabled?: boolean }>()
const emit = defineEmits<{ change: [ready: boolean]; busy: [capturing: boolean] }>()
const upload = ref<InstanceType<typeof UiFileInput>>()
const preview = ref<HTMLImageElement>()
const reference = ref('')
const capturing = ref(false)
const error = ref('')
type Point = { x: number; y: number }
type Region = Point & { w: number; h: number }
const region = ref<Region>()
const regionStyle = computed(() => region.value && ({
  left: region.value.x * 100 + '%', top: region.value.y * 100 + '%',
  width: region.value.w * 100 + '%', height: region.value.h * 100 + '%',
}))
let drag: Point | undefined
let stream: MediaStream | undefined
let disposed = false

function releaseReference(): void {
  if (reference.value) URL.revokeObjectURL(reference.value)
  reference.value = ''
  region.value = undefined
  drag = undefined
}

function replace(file: Blob, needsCrop: boolean): void {
  releaseReference()
  error.value = ''
  reference.value = URL.createObjectURL(file)
  region.value = needsCrop ? undefined : { x: 0, y: 0, w: 1, h: 1 }
  emit('change', !needsCrop)
}

function uploadReference(files: File[]): void {
  const [file] = files
  upload.value?.clear()
  if (file) replace(file, false)
}

function toPng(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => canvas.toBlob(blob => {
    if (blob) resolve(blob)
    else reject(new Error('无法读取参考图片'))
  }, 'image/png'))
}

function stopCapture(): void {
  stream?.getTracks().forEach(track => track.stop())
  stream = undefined
}

async function captureScreen(): Promise<void> {
  if (!navigator.mediaDevices?.getDisplayMedia) {
    error.value = '当前浏览器或页面地址不支持截图，请上传图片'
    return
  }
  capturing.value = true
  emit('busy', true)
  error.value = ''
  let video: HTMLVideoElement | undefined
  try {
    const capture = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false })
    if (disposed) { capture.getTracks().forEach(track => track.stop()); return }
    stream = capture
    video = document.createElement('video')
    video.muted = true
    video.playsInline = true
    video.srcObject = capture
    await video.play()
    if (disposed) return
    if (!video.videoWidth || !video.videoHeight) throw new Error('empty frame')
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const context = canvas.getContext('2d')
    if (!context) throw new Error('canvas unavailable')
    context.drawImage(video, 0, 0)
    const image = await toPng(canvas)
    if (!disposed) replace(image, true)
  } catch (cause) {
    if (!disposed) error.value = cause instanceof DOMException && cause.name === 'NotAllowedError'
      ? '截图已取消或授权被拒绝，可以上传图片'
      : '当前无法截图，请上传图片'
  } finally {
    if (video) { video.pause(); video.srcObject = null }
    stopCapture()
    if (!disposed) { capturing.value = false; emit('busy', false) }
  }
}

function pointer(event: PointerEvent): Point {
  const box = (event.currentTarget as HTMLElement).getBoundingClientRect()
  const clamp = (coordinate: number) => Math.max(0, Math.min(1, coordinate))
  return { x: clamp((event.clientX - box.left) / box.width), y: clamp((event.clientY - box.top) / box.height) }
}

function beginCrop(event: PointerEvent): void {
  if (capturing.value || event.button !== 0) return
  const target = event.currentTarget as HTMLElement
  target.setPointerCapture(event.pointerId)
  drag = pointer(event)
  region.value = undefined
  emit('change', false)
}

function moveCrop(event: PointerEvent): void {
  if (!drag) return
  const { x, y } = pointer(event)
  region.value = { x: Math.min(x, drag.x), y: Math.min(y, drag.y), w: Math.abs(x - drag.x), h: Math.abs(y - drag.y) }
}

function endCrop(event: PointerEvent): void {
  moveCrop(event)
  drag = undefined
  const area = region.value
  const image = preview.value
  if (!area || !image || area.w * image.naturalWidth < 2 || area.h * image.naturalHeight < 2) region.value = undefined
  emit('change', Boolean(region.value))
}

async function read(): Promise<Blob> {
  const image = preview.value
  const area = region.value
  if (!image || !area || drag) throw new Error('请框选参考文字')
  await image.decode()
  if (disposed || region.value !== area) throw new Error('参考选区已变更')
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

function badImage(): void {
  releaseReference()
  error.value = '无法读取参考图片，请重新选取'
  emit('change', false)
}

onBeforeUnmount(() => { disposed = true; stopCapture(); releaseReference() })
defineExpose({ read })
</script>

<template>
  <div class="reference-image-input">
    <div class="reference-image-input__preview">
      <div
        v-if="reference"
        class="reference-image-input__image"
        @pointerdown.prevent="!disabled && beginCrop($event)"
        @pointermove="moveCrop"
        @pointerup="drag && endCrop($event)"
        @pointercancel="drag = undefined; region = undefined; emit('change', false)"
      >
        <img ref="preview" :src="reference" alt="参考图片" draggable="false" @error="badImage">
        <span v-if="regionStyle" class="reference-image-input__region" :style="regionStyle" />
      </div>
      <span v-else>选择一张参考文字图片</span>
    </div>
    <p v-if="reference" class="reference-image-input__hint">{{ region ? '可拖动重新框选文字' : '请拖动框选参考文字' }}</p>
    <div v-if="!disabled" class="reference-image-input__actions">
      <UiFileInput ref="upload" accept="image/png,image/jpeg,image/webp,image/bmp,image/gif" hidden @files-change="uploadReference" />
      <UiButton size="sm" variant="secondary" :disabled="capturing" @click="upload?.click()">{{ reference ? '更换图片' : '上传图片' }}</UiButton>
      <UiButton size="sm" variant="secondary" :disabled="capturing" @click="captureScreen">{{ capturing ? '正在截图…' : '截图' }}</UiButton>
    </div>
    <p v-if="error" class="reference-image-input__error" role="alert">{{ error }}</p>
  </div>
</template>

<style scoped>
.reference-image-input { display: grid; gap: 8px; }

.reference-image-input__preview {
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
.reference-image-input__image { position: relative; overflow: hidden; cursor: crosshair; touch-action: none; }
.reference-image-input__image img { display: block; max-width: 100%; max-height: min(42dvh, 280px); }

.reference-image-input__region {
  position: absolute;
  box-sizing: border-box;
  border: 2px solid var(--color-action-brand);
  box-shadow: 0 0 0 999px var(--color-overlay-scrim-subtle);
  pointer-events: none;
}
.reference-image-input__actions { display: flex; gap: 6px; }
.reference-image-input__hint, .reference-image-input__error { margin: 0; font-size: 12px; line-height: 1.5; }
.reference-image-input__hint { color: var(--color-text-supporting); }
.reference-image-input__error { color: var(--color-status-error); }
</style>
