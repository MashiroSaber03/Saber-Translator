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
  debounceResolve: (() => void) | null
  debounceTimer: ReturnType<typeof setTimeout> | null
  defaultFontChanged: boolean
  desiredDefaultFontId: string | null
  desiredPropagateStyleFields: Set<string>
  desiredStylePatch: Record<string, unknown>
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
    textAlign: bubble.textAlign,
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
    fontId: bubble.fontFamily,
  }
}

function ensureClientMutationIds(bubbles: BubbleState[]): void {
  for (const bubble of bubbles) {
    if (!bubble.backendBubbleId && !bubble.clientMutationId) {
      bubble.clientMutationId = crypto.randomUUID()
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
    if (
      pageId === protectedPageId
      || state.promise
      || state.saving
    ) continue
    states.delete(pageId)
  }
}

function mutationsFor(
  persisted: BubbleState[],
  desired: BubbleState[],
): V2PageDocumentBatchMutation['mutations'] {
  const previous = new Map(
    persisted
      .filter(bubble => bubble.backendBubbleId)
      .map(bubble => [bubble.backendBubbleId!, bubble]),
  )
  const currentIds = new Set(
    desired
      .map(bubble => bubble.backendBubbleId)
      .filter((id): id is string => Boolean(id)),
  )
  const mutations: V2PageDocumentBatchMutation['mutations'] = []

  for (const bubble of persisted) {
    const id = bubble.backendBubbleId
    if (id && !currentIds.has(id)) {
      mutations.push({
        bubbleId: id,
        clientMutationId: crypto.randomUUID(),
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
        fields,
        op: 'create',
      })
      continue
    }
    const old = previous.get(id)
    if (old && canonical(bubbleFields(old)) !== canonical(fields)) {
      mutations.push({
        bubbleId: id,
        clientMutationId: crypto.randomUUID(),
        fields,
        op: 'reset',
      })
    }
  }
  return mutations
}

function applyCreatedBubbleIds(
  bubbles: BubbleState[],
  results: V2PageDocumentMutationResponse['mutationResults'],
): void {
  const createdIds = new Map(
    results
      .filter(result => result.op === 'create')
      .map(result => [result.clientMutationId, result.bubbleId]),
  )
  for (const bubble of bubbles) {
    if (!bubble.clientMutationId) continue
    const backendBubbleId = createdIds.get(bubble.clientMutationId)
    if (!backendBubbleId) continue
    bubble.backendBubbleId = backendBubbleId
    delete bubble.clientMutationId
  }
}

function isAmbiguousTransportError(error: unknown): boolean {
  if (!(error instanceof Error)) return false
  const code = (error as Error & { code?: unknown }).code
  return typeof code === 'string' && [
    'ECONNABORTED',
    'ECONNRESET',
    'EPIPE',
    'ERR_NETWORK',
    'ETIMEDOUT',
  ].includes(code)
}

async function mutateWithTransportReplay(
  pageId: string,
  command: V2PageDocumentBatchMutation,
): Promise<V2PageDocumentMutationResponse> {
  const idempotencyKey = crypto.randomUUID()
  try {
    return await mutatePageDocument(pageId, command, idempotencyKey)
  } catch (error) {
    if (!isAmbiguousTransportError(error)) throw error
    return mutatePageDocument(pageId, command, idempotencyKey)
  }
}

export function registerPageDocument(document: V2PageDocument): BubbleState[] {
  const bubbles = pageDocumentToBubbles(document)
  const existing = states.get(document.pageId)
  if (
    !existing
    || (
      !existing.saving
      && !existing.promise
      && (
        existing.lastError
        || canonical(existing.desired) === canonical(existing.persisted)
      )
    )
  ) {
    const state: PersistedPageState = {
      debounceResolve: null,
      debounceTimer: null,
      defaultFontChanged: false,
      desiredDefaultFontId: document.defaultFontId ?? null,
      desiredPropagateStyleFields: new Set(),
      desiredStylePatch: {},
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
    return bubbles
  }
  touchState(document.pageId, existing)
  evictSettledStates(document.pageId)
  return cloneBubbles(existing.desired)
}

export function queuePageDocumentSave(
  pageId: string,
  documentRevision: number,
  bubbles: BubbleState[],
): Promise<void> {
  return queuePageDocumentMutation(pageId, documentRevision, bubbles)
}

export interface PageDocumentStyleMutation {
  defaultFontId?: string | null
  pageStyleDefaultsPatch?: Record<string, unknown>
  propagateStyleFields?: string[]
}

export function queuePageDocumentMutation(
  pageId: string,
  documentRevision: number,
  bubbles: BubbleState[],
  style: PageDocumentStyleMutation = {},
): Promise<void> {
  const state = states.get(pageId)
  if (!state) {
    throw new Error(`页面文档 ${pageId} 尚未从后端注册`)
  }
  if (
    !Number.isSafeInteger(documentRevision)
    || documentRevision < 1
    || documentRevision !== state.documentRevision
  ) {
    throw new Error(
      `页面文档 ${pageId} 版本已变化：当前为 ${state.documentRevision}，提交版本为 ${documentRevision}`,
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

async function persistLoop(
  pageId: string,
  state: PersistedPageState,
): Promise<void> {
  state.saving = true
  try {
    while (true) {
      await waitForTrailingWindow(state)
      const sentVersion = state.desiredVersion
      const sent = cloneBubbles(state.desired)
      const mutations = mutationsFor(state.persisted, sent)
      const sentStylePatch = deepClone(state.desiredStylePatch)
      const sentPropagation = [...state.desiredPropagateStyleFields]
      const sentDefaultFont = state.desiredDefaultFontId
      const sentDefaultFontChanged = state.defaultFontChanged
      const hasStyleCommand = (
        Object.keys(sentStylePatch).length > 0
        || sentDefaultFontChanged
        || sentPropagation.length > 0
      )
      if (
        mutations.length === 0
        && !hasStyleCommand
      ) return
      // A sidebar propagation and editor bubble delta are separate domain
      // commands. Flush the bubble delta first so the backend never has to
      // guess an overwrite order for the same bubble field.
      const sendStyleCommand = mutations.length === 0 && hasStyleCommand
      const command: V2PageDocumentBatchMutation = {
        baseRevision: state.documentRevision,
        mutations,
        ...(sendStyleCommand && sentDefaultFontChanged
          ? { defaultFontId: sentDefaultFont }
          : {}),
        ...(sendStyleCommand && Object.keys(sentStylePatch).length > 0
          ? {
              pageStyleDefaultsPatch: sentStylePatch,
            }
          : {}),
        ...(sendStyleCommand && sentPropagation.length > 0
          ? {
              propagateStyleFields: sentPropagation as V2PageDocumentBatchMutation['propagateStyleFields'],
            }
          : {}),
      }
      const response = await mutateWithTransportReplay(pageId, command)
      const document = response.document
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
          sentDefaultFontChanged
          && state.defaultFontChanged
          && state.desiredDefaultFontId === sentDefaultFont
        ) {
          state.defaultFontChanged = false
        }
      }
      state.documentRevision = document.documentRevision
      state.persisted = pageDocumentToBubbles(document)

      const imageIndex = imageStore.images.findIndex(image => image.id === pageId)
      if (imageIndex >= 0) {
        imageStore.updateImageByIndex(imageIndex, {
          documentRevision: document.documentRevision,
          bubbleStates: sentVersion === state.desiredVersion
            ? cloneBubbles(state.persisted)
            : cloneBubbles(state.desired),
          hasUnsavedChanges: sentVersion !== state.desiredVersion,
        })
      }
      if (sentVersion === state.desiredVersion) {
        state.desired = cloneBubbles(state.persisted)
        if (imageStore.currentImage?.id === pageId) {
          bubbleStore.setBubbles(cloneBubbles(state.persisted), true)
          bubbleStore.saveAsInitial()
        }
        if (
          Object.keys(state.desiredStylePatch).length === 0
          && !state.defaultFontChanged
          && state.desiredPropagateStyleFields.size === 0
        ) return
      }
    }
  } catch (error) {
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
    await new Promise<void>((resolve) => {
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
