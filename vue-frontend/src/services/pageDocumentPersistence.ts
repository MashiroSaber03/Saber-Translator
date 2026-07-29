import {
  mutatePageDocument,
  type V2PageDocument,
  type V2PageDocumentBatchMutation,
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

function bubbleFields(bubble: BubbleState): Record<string, unknown> {
  const fields = { ...deepClone(bubble) } as Record<string, unknown>
  delete fields.backendBubbleId
  const fontId = typeof fields.fontFamily === 'string' ? fields.fontFamily : null
  delete fields.fontFamily
  fields.fontId = fontId
  return fields
}

function ensureBubbleIds(bubbles: BubbleState[]): void {
  for (const bubble of bubbles) {
    if (!bubble.backendBubbleId) bubble.backendBubbleId = crypto.randomUUID()
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
      || state.lastError
    ) continue
    states.delete(pageId)
  }
}

function mutationsFor(
  persisted: BubbleState[],
  desired: BubbleState[],
): Array<{
  bubbleId: string
  fields?: Record<string, unknown>
  op: 'create' | 'delete' | 'reset'
}> {
  const previous = new Map(
    persisted
      .filter(bubble => bubble.backendBubbleId)
      .map(bubble => [bubble.backendBubbleId!, bubble]),
  )
  const currentIds = new Set(desired.map(bubble => bubble.backendBubbleId!))
  const mutations: Array<{
    bubbleId: string
    fields?: Record<string, unknown>
    op: 'create' | 'delete' | 'reset'
  }> = []

  for (const bubble of persisted) {
    const id = bubble.backendBubbleId
    if (id && !currentIds.has(id)) mutations.push({ bubbleId: id, op: 'delete' })
  }
  for (const bubble of desired) {
    const id = bubble.backendBubbleId!
    const fields = bubbleFields(bubble)
    const old = previous.get(id)
    if (!old) {
      mutations.push({ bubbleId: id, fields, op: 'create' })
    } else if (canonical(bubbleFields(old)) !== canonical(fields)) {
      mutations.push({ bubbleId: id, fields, op: 'reset' })
    }
  }
  return mutations
}

export function registerPageDocument(document: V2PageDocument): BubbleState[] {
  const bubbles = pageDocumentToBubbles(document)
  const existing = states.get(document.pageId)
  if (
    !existing
    || (
      !existing.saving
      && canonical(existing.desired) === canonical(existing.persisted)
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
  ensureBubbleIds(bubbles)
  let state = states.get(pageId)
  if (!state) {
    state = {
      debounceResolve: null,
      debounceTimer: null,
      defaultFontChanged: false,
      desiredDefaultFontId: null,
      desiredPropagateStyleFields: new Set(),
      desiredStylePatch: {},
      desired: [],
      desiredVersion: 0,
      documentRevision,
      lastError: null,
      lastQueuedAt: 0,
      persisted: [],
      promise: null,
      saving: false,
      flushRequested: false,
    }
    states.set(pageId, state)
  }
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
      if (
        mutations.length === 0
        && Object.keys(sentStylePatch).length === 0
        && !sentDefaultFontChanged
      ) return
      const document = await mutatePageDocument(pageId, {
        baseRevision: state.documentRevision,
        mutations,
        ...(sentDefaultFontChanged ? { defaultFontId: sentDefaultFont } : {}),
        ...(Object.keys(sentStylePatch).length > 0
          ? {
              pageStyleDefaultsPatch: sentStylePatch,
              propagateStyleFields: sentPropagation as V2PageDocumentBatchMutation['propagateStyleFields'],
            }
          : {}),
      })
      for (const [field, value] of Object.entries(sentStylePatch)) {
        if (canonical(state.desiredStylePatch[field]) === canonical(value)) {
          delete state.desiredStylePatch[field]
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
      state.documentRevision = document.documentRevision
      state.persisted = pageDocumentToBubbles(document)

      const imageStore = useImageStore()
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
          const bubbleStore = useBubbleStore()
          bubbleStore.setBubbles(cloneBubbles(state.persisted), true)
          bubbleStore.saveAsInitial()
        }
        if (
          Object.keys(state.desiredStylePatch).length === 0
          && !state.defaultFontChanged
        ) return
      }
    }
  } catch (error) {
    state.lastError = error instanceof Error ? error : new Error('页面文档写入失败')
    throw state.lastError
  } finally {
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
