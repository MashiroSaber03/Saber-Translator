import { createUuid } from '@/utils/uuid'
import { apiClient } from '@/api/client'
import {
  mutatePageDocument,
  type V2PageDocument,
  type V2PageDocumentBatchMutation,
  type V2PageDocumentMutationResponse,
  type V2CompleteBubbleMutationFields,
} from '@/api/v2/content'
import { pageDocumentToBubbles } from '@/adapters/v2ContentAdapter'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import type { BubbleState } from '@/types/bubble'
import { deepClone } from '@/utils/deepClone'

interface PersistedPageState {
  nativeVersion: number
  savedNativeVersion: number
  debounceResolve: (() => void) | null
  debounceTimer: ReturnType<typeof setTimeout> | null
  defaultFontChanged: boolean
  desiredDefaultFontId: string | null
  desiredPropagateStyleFields: Set<string>
  desiredStylePatch: Record<string, unknown>
  desiredBubbleFields: Record<string, Set<string>>
  desired: BubbleState[]
  desiredVersion: number
  documentRevision: number
  lastError: Error | null
  lastQueuedAt: number
  persisted: BubbleState[]
  promise: Promise<void> | null
  saving: boolean
  flushRequested: boolean
}

const PAGE_DOCUMENT_TRAILING_MS = 150
const PAGE_DOCUMENT_CACHE_SIZE = 3
const states = new Map<string, PersistedPageState>()

export interface PageEditorPort {
  save(baseRevision: number): Promise<V2PageDocumentMutationResponse>
}
const editorPorts = new Map<string, PageEditorPort>()

export function registerPageEditor(pageId: string, port: PageEditorPort): () => void {
  editorPorts.set(pageId, port)
  return () => { if (editorPorts.get(pageId) === port) editorPorts.delete(pageId) }
}

export function queueEditorDocumentSave(pageId: string, revision: number): Promise<void> {
  const state = states.get(pageId)
  if (!state) throw new Error('页面文档尚未加载')
  state.nativeVersion = Math.max(state.nativeVersion, revision)
  return queuePageDocumentMutation(pageId, state.documentRevision, state.desired)
}

function cloneBubbles(bubbles: BubbleState[]): BubbleState[] {
  // Pinia/Vue exposes the current editor state as reactive proxies, which
  // structuredClone rejects. Page documents are JSON values by contract.
  return deepClone(bubbles)
}

function canonical(value: unknown): string {
  return JSON.stringify(value)
}

function bubbleFields(bubble: BubbleState): V2CompleteBubbleMutationFields {
  return {
    originalText: bubble.originalText,
    translatedText: bubble.translatedText,
    textboxText: bubble.textboxText,
    coords: [...bubble.coords],
    polygon: bubble.polygon.map(point => [...point]),
    fontSize: bubble.fontSize,
    textDirection: bubble.textDirection,
    textColor: bubble.textColor,
    fillColor: bubble.fillColor,
    rotationAngle: bubble.rotationAngle,
    position: { ...bubble.position },
    strokeEnabled: bubble.strokeEnabled,
    strokeColor: bubble.strokeColor,
    strokeWidth: bubble.strokeWidth,
    lineSpacing: bubble.lineSpacing,
    inlineAlign: bubble.inlineAlign,
    blockAlign: bubble.blockAlign,
    inpaintMethod: bubble.inpaintMethod,
    autoFgColor: bubble.autoFgColor ? [...bubble.autoFgColor] : null,
    autoBgColor: bubble.autoBgColor ? [...bubble.autoBgColor] : null,
    colorConfidence: bubble.colorConfidence,
    textlines: bubble.textlines.map(line => ({
      polygon: line.polygon.map(point => [...point]),
      direction: line.direction,
      confidence: line.confidence,
    })),
    ocrResult: bubble.ocrResult ? { ...bubble.ocrResult } : null,
    richText: bubble.richText ?? null,
    fontId: bubble.fontFamily,
  }
}

function ensureClientMutationIds(bubbles: BubbleState[]): void {
  for (const bubble of bubbles) {
    if (!bubble.backendBubbleId && !bubble.clientMutationId) {
      bubble.clientMutationId = createUuid()
    }
  }
}

function touchState(pageId: string, state: PersistedPageState): void {
  states.delete(pageId)
  states.set(pageId, state)
}

function evictSettledStates(protectedPageId: string): void {
  if (states.size <= PAGE_DOCUMENT_CACHE_SIZE) return
  for (const [pageId, state] of states) {
    if (states.size <= PAGE_DOCUMENT_CACHE_SIZE) return
    if (pageId === protectedPageId || state.promise || state.saving) continue
    states.delete(pageId)
  }
}

function mutationsFor(
  persisted: BubbleState[],
  desired: BubbleState[],
  requestedFields: Record<string, Set<string>>,
): V2PageDocumentBatchMutation['mutations'] {
  const previous = new Map(
    persisted
      .filter(bubble => bubble.backendBubbleId)
      .map(bubble => [bubble.backendBubbleId!, bubble])
  )
  const currentIds = new Set(
    desired.map(bubble => bubble.backendBubbleId).filter((id): id is string => Boolean(id))
  )
  const mutations: V2PageDocumentBatchMutation['mutations'] = []

  for (const bubble of persisted) {
    const id = bubble.backendBubbleId
    if (id && !currentIds.has(id)) {
      mutations.push({
        bubbleId: id,
        clientMutationId: createUuid(),
        op: 'delete',
      })
    }
  }
  for (const bubble of desired) {
    const id = bubble.backendBubbleId
    const fields = bubbleFields(bubble)
    if (!id) {
      mutations.push({
        clientMutationId: bubble.clientMutationId!,
        ...(bubble.duplicateSourceId ? { sourceBubbleId: bubble.duplicateSourceId } : {}),
        fields,
        op: 'create',
      })
      continue
    }
    const old = previous.get(id)
    const before = old ? bubbleFields(old) : undefined
    const requested = requestedFields[id] ?? new Set<string>()
    const patch = Object.fromEntries(Object.entries(fields).filter(([key, value]) =>
      requested.has(key) || canonical(before?.[key as keyof V2CompleteBubbleMutationFields]) !== canonical(value)))
    if (old && Object.keys(patch).length) {
      mutations.push({
        bubbleId: id,
        clientMutationId: createUuid(),
        fields: patch,
        op: 'patch',
      })
    }
  }
  return mutations
}

function applyCreatedBubbleIds(
  bubbles: BubbleState[],
  results: V2PageDocumentMutationResponse['mutationResults']
): void {
  const createdIds = new Map(
    results
      .filter(result => result.op === 'create')
      .map(result => [result.clientMutationId, result.bubbleId])
  )
  for (const bubble of bubbles) {
    if (!bubble.clientMutationId) continue
    const backendBubbleId = createdIds.get(bubble.clientMutationId)
    if (!backendBubbleId) continue
    bubble.backendBubbleId = backendBubbleId
    delete bubble.clientMutationId
  }
}

function mergeSavedFields(desired: BubbleState[], sent: BubbleState[], saved: BubbleState[]): void {
  const sentById = new Map(sent.map(bubble => [bubble.backendBubbleId, bubble]))
  const savedById = new Map(saved.map(bubble => [bubble.backendBubbleId, bubble]))
  for (const bubble of desired) {
    if (!bubble.backendBubbleId) continue
    const before = sentById.get(bubble.backendBubbleId)
    const after = savedById.get(bubble.backendBubbleId)
    if (!before || !after) continue
    const dx = bubble.coords[0] - before.coords[0], dy = bubble.coords[1] - before.coords[1]
    if (bubble.coords[2] - before.coords[2] === dx && bubble.coords[3] - before.coords[3] === dy) {
      // A style save can resize point-text bounds. Preserve a later translation
      // relative to those new bounds, without turning it into an unintended resize.
      bubble.coords = after.coords.map((value, i) => value + (i % 2 ? dy : dx)) as BubbleState['coords']
    }
    // Carry native projections (character runs, normalized geometry, etc.) forward
    // only where the user has not made another change while the save was in flight.
    for (const key of Object.keys(after) as (keyof BubbleState)[]) {
      if (canonical(bubble[key]) === canonical(before[key])) {
        Object.assign(bubble, { [key]: deepClone(after[key]) })
      }
    }
  }
}

function isAmbiguousTransportError(error: unknown): boolean {
  return (
    error instanceof Error &&
    'code' in error &&
    error.code === 'network_error' &&
    'status' in error &&
    error.status === 0
  )
}

async function mutateWithTransportReplay(
  pageId: string,
  command: V2PageDocumentBatchMutation
): Promise<V2PageDocumentMutationResponse> {
  return withDocumentTransportReplay(key => mutatePageDocument(pageId, command, key))
}

export async function withDocumentTransportReplay<T>(submit: (key: string) => Promise<T>): Promise<T> {
  const idempotencyKey = createUuid()
  try {
    return await submit(idempotencyKey)
  } catch (error) {
    if (!isAmbiguousTransportError(error)) throw error
    return submit(idempotencyKey)
  }
}

export function saveEditorDocument(pageId: string, baseRevision: number, file: Blob, snapshot?: V2PageDocument): Promise<V2PageDocumentMutationResponse> {
  const body = new FormData()
  body.append('baseRevision', String(baseRevision))
  body.append('file', file, 'page.pcraft')
  if (snapshot) body.append('snapshot', JSON.stringify(snapshot))
  return withDocumentTransportReplay(key => apiClient.put<V2PageDocumentMutationResponse>(`/api/v2/pages/${encodeURIComponent(pageId)}/editor-document`, body, { headers: { 'Idempotency-Key': key } }))
}

export function registerPageDocument(document: V2PageDocument): BubbleState[] {
  const bubbles = pageDocumentToBubbles(document)
  const existing = states.get(document.pageId)
  const image = useImageStore().images.find(image => image.id === document.pageId)
  // Reads can finish out of order, including reads started before a successful
  // save. Never publish that old projection or replace an unsaved draft with it.
  if (document.documentRevision < Math.max(existing?.documentRevision ?? 0, image?.documentRevision ?? 0)) {
    return cloneBubbles(existing?.desired ?? image?.bubbleStates ?? [])
  }
  if (
    !existing ||
    (!existing.saving &&
      !existing.promise &&
      ((existing.lastError && document.documentRevision !== existing.documentRevision)
        || canonical(existing.desired) === canonical(existing.persisted)))
  ) {
    const state: PersistedPageState = {
      nativeVersion: 0,
      savedNativeVersion: 0,
      debounceResolve: null,
      debounceTimer: null,
      defaultFontChanged: false,
      desiredDefaultFontId: document.defaultFontId ?? null,
      desiredPropagateStyleFields: new Set(),
      desiredStylePatch: {},
      desiredBubbleFields: {},
      desired: cloneBubbles(bubbles),
      desiredVersion: 0,
      documentRevision: document.documentRevision,
      lastError: null,
      lastQueuedAt: 0,
      persisted: cloneBubbles(bubbles),
      promise: null,
      saving: false,
      flushRequested: false,
    }
    touchState(document.pageId, state)
    evictSettledStates(document.pageId)
    publishDocument(document, bubbles, false)
    return bubbles
  }
  touchState(document.pageId, existing)
  evictSettledStates(document.pageId)
  return cloneBubbles(existing.desired)
}

function publishDocument(document: V2PageDocument, bubbles: BubbleState[], dirty: boolean): void {
  const images = useImageStore()
  const index = images.images.findIndex(image => image.id === document.pageId)
  if (index < 0) return
  images.updateImageByIndex(index, {
    ...document.pageStyleDefaults,
    ...(document.defaultFontId ? { fontFamily: document.defaultFontId } : {}),
    documentRevision: document.documentRevision,
    hasUnsavedChanges: dirty,
    isManuallyAnnotated: document.isManuallyAnnotated ?? false,
  })
  if (images.currentImage?.id === document.pageId) useBubbleStore().setBubbles(bubbles, true)
  else images.updateImageByIndex(index, { bubbleStates: cloneBubbles(bubbles) })
}

export function queuePageDocumentSave(
  pageId: string,
  documentRevision: number,
  bubbles: BubbleState[]
): Promise<void> {
  return queuePageDocumentMutation(pageId, documentRevision, bubbles)
}

export interface PageDocumentStyleMutation {
  bubbleFields?: Record<string, string[]>
  defaultFontId?: string | null
  pageStyleDefaultsPatch?: Record<string, unknown>
  propagateStyleFields?: string[]
}

export function queuePageDocumentMutation(
  pageId: string,
  documentRevision: number,
  bubbles: BubbleState[],
  style: PageDocumentStyleMutation = {}
): Promise<void> {
  const state = states.get(pageId)
  if (!state) {
    throw new Error(`页面文档 ${pageId} 尚未从后端注册`)
  }
  if (
    !Number.isSafeInteger(documentRevision) ||
    documentRevision < 1 ||
    documentRevision !== state.documentRevision
  ) {
    throw new Error(
      `页面文档 ${pageId} 版本已变化：当前为 ${state.documentRevision}，提交版本为 ${documentRevision}`
    )
  }
  ensureClientMutationIds(bubbles)
  touchState(pageId, state)
  state.desired = cloneBubbles(bubbles)
  if (Object.hasOwn(style, 'defaultFontId')) {
    state.defaultFontChanged = true
    state.desiredDefaultFontId = style.defaultFontId ?? null
  }
  Object.assign(state.desiredStylePatch, style.pageStyleDefaultsPatch ?? {})
  for (const [id, fields] of Object.entries(style.bubbleFields ?? {})) {
    const requested = state.desiredBubbleFields[id] ??= new Set()
    for (const field of fields) requested.add(field === 'fontFamily' ? 'fontId' : field)
  }
  for (const field of style.propagateStyleFields ?? []) {
    state.desiredPropagateStyleFields.add(field)
  }
  state.desiredVersion += 1
  state.lastQueuedAt = Date.now()
  state.lastError = null
  if (!state.promise) {
    state.promise = persistLoop(pageId, state).finally(() => {
      state!.promise = null
      evictSettledStates(pageId)
    })
  }
  return state.promise
}

async function persistLoop(pageId: string, state: PersistedPageState): Promise<void> {
  state.saving = true
  let inFlightFields: Record<string, Set<string>> = {}
  try {
    while (true) {
      await waitForTrailingWindow(state)
      const sentVersion = state.desiredVersion
      const nativeVersion = state.nativeVersion
      const nativeDirty = nativeVersion !== state.savedNativeVersion
      const sent = cloneBubbles(state.desired)
      const mutations = nativeDirty ? [] : mutationsFor(state.persisted, sent, state.desiredBubbleFields)
      const sentStylePatch = deepClone(state.desiredStylePatch)
      const sentPropagation = [...state.desiredPropagateStyleFields]
      const sentDefaultFont = state.desiredDefaultFontId
      const sentDefaultFontChanged = state.defaultFontChanged
      const hasStyleCommand =
        Object.keys(sentStylePatch).length > 0 ||
        sentDefaultFontChanged ||
        sentPropagation.length > 0
      if (mutations.length === 0 && !hasStyleCommand && !nativeDirty) {
        // A no-op preview may follow a save while native IDs/geometry are being
        // projected back. Settle on that projection just like a successful write.
        state.desired = cloneBubbles(state.persisted)
        const images = useImageStore()
        const index = images.images.findIndex(image => image.id === pageId)
        if (index >= 0 && images.images[index]?.documentRevision === state.documentRevision)
          images.updateImageByIndex(index, { hasUnsavedChanges: false })
        return
      }
      // A sidebar propagation and editor bubble delta are separate domain
      // commands. Flush the bubble delta first so the backend never has to
      // guess an overwrite order for the same bubble field.
      const sendStyleCommand = mutations.length === 0 && hasStyleCommand && !nativeDirty
      const command: V2PageDocumentBatchMutation = {
        baseRevision: state.documentRevision,
        mutations,
        ...(sendStyleCommand && sentDefaultFontChanged ? { defaultFontId: sentDefaultFont } : {}),
        ...(sendStyleCommand && Object.keys(sentStylePatch).length > 0
          ? {
              pageStyleDefaultsPatch: sentStylePatch,
            }
          : {}),
        ...(sendStyleCommand && sentPropagation.length > 0
          ? {
              propagateStyleFields:
                sentPropagation as V2PageDocumentBatchMutation['propagateStyleFields'],
            }
          : {}),
      }
      const port = editorPorts.get(pageId)
      if (nativeDirty && !port) throw new Error('精细编辑画布已关闭，请先保存当前修改')
      if (!nativeDirty && !sendStyleCommand) {
        inFlightFields = state.desiredBubbleFields
        state.desiredBubbleFields = {}
      }
      const response = nativeDirty && port
        ? await port.save(state.documentRevision)
        : await mutateWithTransportReplay(pageId, command)
      const document = response.document
      if (nativeDirty) state.savedNativeVersion = nativeVersion
      inFlightFields = {}
      applyCreatedBubbleIds(sent, response.mutationResults)
      applyCreatedBubbleIds(state.desired, response.mutationResults)
      const bubbleStore = useBubbleStore()
      const imageStore = useImageStore()
      if (imageStore.currentImage?.id === pageId) {
        applyCreatedBubbleIds(bubbleStore.bubbles, response.mutationResults)
      }
      if (sendStyleCommand) {
        for (const [field, value] of Object.entries(sentStylePatch)) {
          if (canonical(state.desiredStylePatch[field]) === canonical(value)) {
            delete state.desiredStylePatch[field]
          }
        }
        if (sentVersion === state.desiredVersion) {
          for (const field of sentPropagation) {
            state.desiredPropagateStyleFields.delete(field)
          }
        }
        if (
          sentDefaultFontChanged &&
          state.defaultFontChanged &&
          state.desiredDefaultFontId === sentDefaultFont
        ) {
          state.defaultFontChanged = false
        }
      }
      state.documentRevision = document.documentRevision
      state.persisted = pageDocumentToBubbles(document)
      if (sentVersion === state.desiredVersion) state.desired = cloneBubbles(state.persisted)
      else mergeSavedFields(state.desired, sent, state.persisted)

      publishDocument(document, state.desired, sentVersion !== state.desiredVersion)
      if (sentVersion === state.desiredVersion) {
        if (
          Object.keys(state.desiredStylePatch).length === 0 &&
          !state.defaultFontChanged &&
          state.desiredPropagateStyleFields.size === 0
          && state.savedNativeVersion === state.nativeVersion
        )
          return
      }
    }
  } catch (error) {
    for (const [id, fields] of Object.entries(inFlightFields)) {
      const pending = state.desiredBubbleFields[id] ??= new Set()
      fields.forEach(field => pending.add(field))
    }
    state.lastError = error instanceof Error ? error : new Error('页面文档写入失败')
    throw state.lastError
  } finally {
    if (state.debounceTimer) clearTimeout(state.debounceTimer)
    state.debounceTimer = null
    state.debounceResolve = null
    state.flushRequested = false
    state.saving = false
  }
}

async function waitForTrailingWindow(state: PersistedPageState): Promise<void> {
  while (!state.flushRequested) {
    const remaining = PAGE_DOCUMENT_TRAILING_MS - (Date.now() - state.lastQueuedAt)
    if (remaining <= 0) return
    await new Promise<void>(resolve => {
      let settled = false
      const finish = () => {
        if (settled) return
        settled = true
        state.debounceTimer = null
        state.debounceResolve = null
        resolve()
      }
      state.debounceResolve = finish
      state.debounceTimer = setTimeout(finish, remaining)
    })
  }
  state.flushRequested = false
}

export function hasPendingPageDocument(pageId: string): boolean {
  const state = states.get(pageId)
  return Boolean(state?.promise || state?.lastError)
}

export function isPageDocumentRegistered(pageId: string): boolean {
  return states.has(pageId)
}

export function discardPageDocument(pageId: string): boolean {
  const state = states.get(pageId)
  if (!state) return true
  if (state.promise || state.saving) return false
  states.delete(pageId)
  return true
}

export async function flushPageDocument(pageId: string): Promise<void> {
  const state = states.get(pageId)
  if (!state) return
  if (state.promise) {
    state.flushRequested = true
    if (state.debounceTimer) clearTimeout(state.debounceTimer)
    state.debounceResolve?.()
  }
  if (state.promise) await state.promise
  if (state.lastError) throw state.lastError
}
