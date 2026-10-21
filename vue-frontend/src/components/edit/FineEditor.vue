<template>
  <section class="fine-editor" :class="{ 'fine-editor--inactive': !active }" :aria-hidden="!active || undefined" aria-label="精细编辑">
    <div class="fine-editor__bar">
      <slot name="toolbar-start" />
      <div class="fine-editor__history" role="group" aria-label="编辑历史">
        <UiIconButton size="sm" variant="inverse" label="撤销" title="撤销 (Ctrl+Z)" :disabled="loading || loadFailed || repairing || !canUndo" @click="command('edit.undo')"><UiIcon name="rotate-ccw" :size="16" /></UiIconButton>
        <UiIconButton size="sm" variant="inverse" label="重做" title="重做 (Ctrl+Shift+Z)" :disabled="loading || loadFailed || repairing || !canRedo" @click="command('edit.redo')"><UiIcon name="rotate-cw" :size="16" /></UiIconButton>
      </div>
      <div class="fine-editor__method"><UiSelect size="sm" aria-label="背景修复方式" teleport-to=".fine-editor" :options="repairMethods" :model-value="inpaintMethod" :disabled="loading || loadFailed || repairing" @update:model-value="emit('method', $event as InpaintMethod)" /></div>
      <UiColorInput v-if="inpaintMethod === 'solid'" aria-label="填充颜色" :model-value="fillColor" :disabled="loading || loadFailed || repairing" @update:model-value="emit('fillColor', $event)" />
      <UiButton size="sm" variant="inverse" :disabled="loading || loadFailed || repairing" title="使用当前背景修复方式修复选区" @click="repairSelection"><UiIcon name="wand-sparkles" :size="15" />修复选区</UiButton>
      <span class="fine-editor__status" :class="{ 'fine-editor__status--error': error }" role="status" :title="error || '修改会自动保存；也可按 Ctrl+S 保存'">
        <UiIcon :name="error ? 'alert-triangle' : loading || saving || repairing ? 'loading' : 'check'" :size="14" :class="{ 'fine-editor__spinner': loading || saving || repairing }" />
        <span>{{ loading ? '正在加载…' : error || (repairing ? '正在修复…' : saving ? '正在保存…' : '已自动保存') }}</span>
      </span>
      <UiButton v-if="loadFailed || saveFailed" size="sm" variant="inverse" @click="retry">重试</UiButton>
      <UiButton size="sm" variant="inverse" :disabled="loading || loadFailed || repairing" @click="retry"><UiIcon name="save" :size="15" />保存</UiButton>
      <slot name="toolbar-end" />
    </div>
    <div class="fine-editor__stage">
      <iframe ref="frame" :inert="loading || loadFailed || repairing || undefined" class="fine-editor__canvas" title="精细编辑画布" src="/photocraft/index.html" />
      <div v-if="loading || loadFailed" class="fine-editor__loading"><UiIcon :name="loadFailed ? 'alert-triangle' : 'loading'" :class="{ 'fine-editor__spinner': loading }" :size="24" /><span>{{ loadFailed ? '本页加载失败，请点击上方“重试”' : '正在准备精细编辑…' }}</span></div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiColorInput from '@/components/ui/UiColorInput.vue'
import { usePublicUserAccess } from '@/composables/usePublicUserAccess'
import { inpaintMethodOptions } from '@/utils/textStyleForm'
import { apiClient } from '@/api/client'
import { getPageDocument, type V2PageDocument, type V2PageDocumentMutationResponse } from '@/api/v2/content'
import { flushPageDocument, hasPendingPageDocument, queueEditorDocumentSave, registerPageDocument, registerPageEditor, withDocumentTransportReplay } from '@/services/pageDocumentPersistence'
import { useEditRender } from '@/composables/useEditRender'
import { createUuid } from '@/utils/uuid'
import { triggerBlobDownload } from '@/utils/browserDownload'
import type { InpaintMethod } from '@/types/bubble'
import { createMaskRepair, waitForOperation } from '@/api/v2/operations'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'

const props = defineProps<{ pageId: string; active: boolean; inpaintMethod: InpaintMethod; fillColor: string }>()
const emit = defineEmits<{ busy: [value: boolean]; method: [value: InpaintMethod]; fillColor: [value: string]; 'document-loaded': [pageId: string] }>()
const access = usePublicUserAccess()
const repairMethods = computed(() => access.modelOptions(inpaintMethodOptions, { lama_mpe: 'lama_mpe', litelama: 'litelama', lama_manga: 'lama_manga' }))
const frame = ref<HTMLIFrameElement>()
const loading = ref(true)
const loadFailed = ref(false)
const saveFailed = ref(false)
const saving = ref(false)
const repairing = ref(false)
const error = ref('')
const canUndo = ref(false)
const canRedo = ref(false)
const render = useEditRender({ onRenderError: message => { error.value = message } })
let ready = false
let loadedPage = ''
let silentRevision = 0
let loadVersion = 0
let unregister: (() => void) | undefined
let disposed = false
const loadedFonts = new Set<string>()
let hiddenSignature = ''
const requests = new Map<string, { resolve: (value: Reply) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>()
interface Reply { requestId?: string; error?: string; event?: string; filename?: string; revision?: number; pageId?: string; bytes?: Uint8Array; canUndo?: boolean; canRedo?: boolean }

async function applyTheme(): Promise<void> {
  const style = getComputedStyle(document.documentElement)
  const colors = Object.fromEntries(Object.entries({
    surface: '--color-surface-inverse', canvas: '--color-surface-inverse-canvas',
    background: '--color-surface-inverse-depth', field: '--color-surface-inverse-raised',
    text: '--color-text-inverse', accent: '--color-surface-brand',
    danger: '--color-status-error-bright', warning: '--color-status-warning-bright',
  }).map(([key, variable]) => {
    const value = style.getPropertyValue(variable).trim()
    return [key, /^#[\da-f]{3}$/i.test(value) ? `#${value.slice(1).split('').map(c => c + c).join('')}` : value]
  }))
  await request({ op: 'theme', colors,
    radius: Number.parseFloat(style.getPropertyValue('--radius-md')),
    radiusLarge: Number.parseFloat(style.getPropertyValue('--radius-lg')) })
}

function request(payload: Record<string, unknown>, bytes?: ArrayBuffer): Promise<Reply> {
  if (!frame.value?.contentWindow || disposed) return Promise.reject(new Error('编辑画布尚未就绪'))
  const requestId = createUuid()
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { requests.delete(requestId); reject(new Error('编辑内核响应超时，请保留当前页面并重试')) }, 60_000)
    requests.set(requestId, { resolve, reject, timer })
    frame.value!.contentWindow!.postMessage({ saberEditor: true, payload: { ...payload, requestId }, bytes }, location.origin)
  })
}

function connectEditor(page: string): void {
  unregister?.()
  unregister = registerPageEditor(page, { save })
}

function adopt(document: V2PageDocument): void {
  const bubbles = registerPageDocument(document)
  const images = useImageStore()
  if (images.currentImage?.id !== document.pageId) return
  images.updateCurrentImage({ ...document.pageStyleDefaults,
    ...(document.defaultFontId ? { fontFamily: document.defaultFontId } : {}),
    documentRevision: document.documentRevision, bubbleStates: bubbles, hasUnsavedChanges: false })
  useBubbleStore().setBubbles(bubbles, true)
}

async function save(baseRevision: number): Promise<V2PageDocumentMutationResponse> {
  const page = loadedPage
  const snapshot = await request({ op: 'snapshot' })
  if (!snapshot.bytes) throw new Error('编辑快照为空')
  const body = new FormData()
  body.append('baseRevision', String(baseRevision))
  body.append('file', new Blob([new Uint8Array(snapshot.bytes).buffer], { type: 'application/x-photocraft' }), 'page.pcraft')
  const response = await withDocumentTransportReplay(key => apiClient.put<V2PageDocumentMutationResponse>(`/api/v2/pages/${page}/editor-document`, body, { headers: { 'Idempotency-Key': key } }))
  await request({ op: 'saved', revision: snapshot.revision })
  silentRevision = Math.max(silentRevision, snapshot.revision ?? 0)
  return response
}

async function load(): Promise<void> {
  if (!ready || !props.pageId || !props.active) return
  const token = ++loadVersion
  const page = props.pageId
  loading.value = true
  loadFailed.value = false
  emit('busy', true)
  error.value = ''
  try {
    await applyTheme()
    if (loadedPage) await flushPageDocument(loadedPage)
    const catalog = await apiClient.get<{ items: { id: string; url: string; isDefault: boolean }[] }>('/api/v2/editor/fonts')
    for (const font of catalog.items) {
      if (loadedFonts.has(font.id)) continue
      const bytes = await apiClient.get<ArrayBuffer>(font.url, { responseType: 'arraybuffer' })
      await request({ op: 'font', ui: font.isDefault }, bytes)
      loadedFonts.add(font.id)
    }
    const document = await getPageDocument(page)
    const response = await fetch(`/api/v2/pages/${page}/editor-document`, { credentials: 'same-origin' })
    if (!response.ok) throw new Error((await response.json()).message || '编辑文档加载失败')
    const revision = Number(response.headers.get('X-Document-Revision'))
    if (revision !== document.documentRevision) throw new Error('页面刚刚发生变化，请重试加载最新文档')
    const bytes = await response.arrayBuffer()
    if (token !== loadVersion || disposed) return
    unregister?.()
    const reply = await request({ op: loadedPage === page ? 'replace' : 'open', pageId: page, label: '普通编辑' }, bytes)
    if (token !== loadVersion || disposed) return
    await request({ op: 'saved', revision: reply.revision })
    adopt(document)
    emit('document-loaded', page)
    loadedPage = page
    silentRevision = reply.revision ?? 0
    connectEditor(page)
    await request({ op: 'presentation', fine: props.active })
  } catch (reason) {
    if (token !== loadVersion || disposed) return
    loadFailed.value = true
    error.value = reason instanceof Error ? reason.message : '精细编辑加载失败'
  } finally {
    if (token === loadVersion) { loading.value = false; emit('busy', false) }
  }
}

async function persist(revision: number): Promise<void> {
  if (!loadedPage || loading.value) return
  saving.value = true
  saveFailed.value = false
  error.value = ''
  try {
    await queueEditorDocumentSave(loadedPage, revision)
    await render.reRenderFullImage()
  } catch (reason) {
    saveFailed.value = true
    error.value = reason instanceof Error ? reason.message : '保存失败，修改仍保留在当前画布'
  }
  finally { saving.value = false }
}

async function saveNow(): Promise<void> {
  if (loadFailed.value) {
    // The failed page stayed inert. Only previously queued work can need saving.
    if (loadedPage) await flushPageDocument(loadedPage)
    return
  }
  if (!loadedPage || loading.value) return
  const reply = await request({ op: 'finish' })
  if ((reply.revision ?? 0) > silentRevision || hasPendingPageDocument(loadedPage)) await persist(reply.revision ?? 0)
  await flushPageDocument(loadedPage)
  error.value = ''
}
async function retry(): Promise<void> {
  try { if (!loadedPage || loadFailed.value) await load(); else await saveNow() }
  catch (reason) { error.value = reason instanceof Error ? reason.message : '保存失败，请重试' }
}
async function command(id: string): Promise<void> {
  try { await request({ op: 'command', command: id }) }
  catch (reason) { error.value = reason instanceof Error ? reason.message : '操作失败' }
}
async function repairSelection(): Promise<void> {
  if (repairing.value) return
  repairing.value = true
  error.value = ''
  emit('busy', true)
  try {
    await saveNow()
    const selected = await request({ op: 'selection' })
    if (!selected.bytes) throw new Error('当前选区为空')
    const image = useImageStore().currentImage
    if (!image?.documentRevision) throw new Error('页面文档尚未保存')
    saving.value = true
    const options = props.inpaintMethod === 'solid'
      ? { method: 'solid' as const, fillColor: props.fillColor, baseRevision: image.documentRevision }
      : { method: props.inpaintMethod, baseRevision: image.documentRevision }
    const accepted = await createMaskRepair(loadedPage, new Blob([new Uint8Array(selected.bytes).buffer], { type: 'image/png' }), options)
    await waitForOperation(accepted.operationId)
    const document = await getPageDocument(loadedPage)
    const bytes = await apiClient.get<ArrayBuffer>(`/api/v2/pages/${loadedPage}/editor-document`, { responseType: 'arraybuffer' })
    const reply = await request({ op: 'replace' }, bytes)
    silentRevision = reply.revision ?? silentRevision
    await request({ op: 'saved', revision: reply.revision })
    adopt(document)
    await render.reRenderFullImage()
  } catch (reason) { error.value = reason instanceof Error ? reason.message : '选区修复失败' }
  finally { saving.value = false; repairing.value = false; emit('busy', false) }
}
function onMessage(event: MessageEvent): void {
  if (event.origin !== location.origin || event.source !== frame.value?.contentWindow || event.data?.saberEditor !== true) return
  const reply = event.data as Reply
  if (reply.canUndo !== undefined) canUndo.value = reply.canUndo
  if (reply.canRedo !== undefined) canRedo.value = reply.canRedo
  if (reply.requestId) {
    const pending = requests.get(reply.requestId)
    if (pending) {
      clearTimeout(pending.timer); requests.delete(reply.requestId)
      if (reply.error) pending.reject(new Error(reply.error)); else pending.resolve(reply)
    }
  } else if (reply.event === 'ready') { ready = true; void load() }
  else if (reply.event === 'save') void retry()
  else if (reply.event === 'download' && reply.bytes && reply.filename) {
    triggerBlobDownload(new Blob([new Uint8Array(reply.bytes).buffer], { type: 'application/octet-stream' }), reply.filename)
  }
  else if (reply.event === 'changed' && reply.pageId === loadedPage && props.active && (reply.revision ?? 0) > silentRevision) void persist(reply.revision ?? 0)
}
function imageSignature(): string {
  const image = useImageStore().currentImage
  return `${image?.id}:${image?.documentRevision}:${image?.cleanAssetUrl}`
}
watch(() => props.pageId, () => { if (loadedPage !== props.pageId && props.active) void load() })
watch(() => props.active, value => {
  if (!ready) return
  if (!value) {
    unregister?.(); unregister = undefined
    hiddenSignature = imageSignature()
    return
  }
  if (loadedPage !== props.pageId || hiddenSignature !== imageSignature()) {
    void load()
    return
  }
  connectEditor(loadedPage)
})
onMounted(() => window.addEventListener('message', onMessage))
onBeforeUnmount(() => {
  disposed = true; loadVersion += 1; unregister?.(); window.removeEventListener('message', onMessage)
  for (const pending of requests.values()) { clearTimeout(pending.timer); pending.reject(new Error('编辑画布已关闭')) }
  requests.clear()
})
defineExpose({ saveNow, command })
</script>

<style scoped>
.fine-editor { position: absolute; inset: 0; min-height: 0; display: flex; flex-direction: column; }
.fine-editor--inactive { visibility: hidden; pointer-events: none; }
.fine-editor__bar { display: flex; flex-shrink: 0; flex-wrap: wrap; align-items: center; gap: 8px; padding: 7px 14px; background: var(--color-surface-inverse); color: var(--color-text-inverse); border-bottom: 1px solid var(--color-overlay-inverse-subtle); }
.fine-editor__history { display: flex; flex-shrink: 0; gap: 4px; padding-right: 10px; margin-right: 2px; border-right: 1px solid var(--color-overlay-inverse-muted); }
.fine-editor__status { display: flex; align-items: center; gap: 6px; flex: 1; min-width: 120px; font-size: 12px; color: var(--color-action-success-bright); }
.fine-editor__status > span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fine-editor__status--error { color: var(--color-status-error-bright); }
.fine-editor__stage { position: relative; flex: 1; min-height: 0; background: var(--color-surface-inverse-canvas); }
.fine-editor__canvas { display: block; width: 100%; height: 100%; border: 0; }
.fine-editor__loading { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; gap: 12px; background: var(--color-surface-inverse-canvas); color: var(--color-text-inverse); font-size: 13px; }
.fine-editor__spinner { animation: fine-editor-spin 1s linear infinite; }
@keyframes fine-editor-spin { to { transform: rotate(360deg); } }
.fine-editor__method { width: 190px; flex-shrink: 0; }
.fine-editor {
  --ui-select-background: var(--color-surface-inverse);
  --ui-select-color: var(--color-text-inverse);
  --ui-select-border: 1px solid color-mix(in srgb, var(--color-text-inverse) 30%, transparent);
  --ui-selector-dropdown-background: var(--color-surface-inverse);
  --ui-selector-control-text: var(--color-text-inverse);
}
</style>
