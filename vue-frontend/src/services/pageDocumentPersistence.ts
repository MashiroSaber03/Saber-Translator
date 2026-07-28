import { mutatePageDocument, type V2PageDocument } from '@/api/v2/content'
import { pageDocumentToBubbles } from '@/adapters/v2ContentAdapter'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import type { BubbleState } from '@/types/bubble'

interface PersistedPageState {
  desired: BubbleState[]
  desiredVersion: number
  documentRevision: number
  lastError: Error | null
  persisted: BubbleState[]
  promise: Promise<void> | null
  saving: boolean
}

const states = new Map<string, PersistedPageState>()

function cloneBubbles(bubbles: BubbleState[]): BubbleState[] {
  return structuredClone(bubbles)
}

function canonical(value: unknown): string {
  return JSON.stringify(value)
}

function bubbleFields(bubble: BubbleState): Record<string, unknown> {
  const fields = { ...structuredClone(bubble) } as Record<string, unknown>
  delete fields.backendBubbleId
  return fields
}

function ensureBubbleIds(bubbles: BubbleState[]): void {
  for (const bubble of bubbles) {
    if (!bubble.backendBubbleId) bubble.backendBubbleId = crypto.randomUUID()
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
    states.set(document.pageId, {
      desired: cloneBubbles(bubbles),
      desiredVersion: 0,
      documentRevision: document.documentRevision,
      lastError: null,
      persisted: cloneBubbles(bubbles),
      promise: null,
      saving: false,
    })
    return bubbles
  }
  return cloneBubbles(existing.desired)
}

export function queuePageDocumentSave(
  pageId: string,
  documentRevision: number,
  bubbles: BubbleState[],
): Promise<void> {
  ensureBubbleIds(bubbles)
  let state = states.get(pageId)
  if (!state) {
    state = {
      desired: [],
      desiredVersion: 0,
      documentRevision,
      lastError: null,
      persisted: [],
      promise: null,
      saving: false,
    }
    states.set(pageId, state)
  }
  state.desired = cloneBubbles(bubbles)
  state.desiredVersion += 1
  state.lastError = null
  if (!state.promise) {
    state.promise = persistLoop(pageId, state).finally(() => {
      state!.promise = null
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
      const sentVersion = state.desiredVersion
      const sent = cloneBubbles(state.desired)
      const mutations = mutationsFor(state.persisted, sent)
      if (mutations.length === 0) return
      const document = await mutatePageDocument(pageId, {
        baseRevision: state.documentRevision,
        mutations,
      })
      state.documentRevision = document.documentRevision
      state.persisted = pageDocumentToBubbles(document)

      const imageStore = useImageStore()
      const imageIndex = imageStore.images.findIndex(image => image.id === pageId)
      if (imageIndex >= 0) {
        imageStore.updateImageByIndex(imageIndex, {
          documentRevision: document.documentRevision,
          hasUnsavedChanges: sentVersion !== state.desiredVersion,
        })
      }
      if (sentVersion === state.desiredVersion) {
        state.desired = cloneBubbles(state.persisted)
        if (imageStore.currentImage?.id === pageId) {
          const bubbleStore = useBubbleStore()
          bubbleStore.saveAsInitial()
        }
        return
      }
    }
  } catch (error) {
    state.lastError = error instanceof Error ? error : new Error('页面文档写入失败')
    throw state.lastError
  } finally {
    state.saving = false
  }
}

export function hasPendingPageDocument(pageId: string): boolean {
  const state = states.get(pageId)
  return Boolean(state?.saving || state?.lastError)
}

export async function flushPageDocument(pageId: string): Promise<void> {
  const state = states.get(pageId)
  if (!state) return
  if (state.promise) await state.promise
  if (state.lastError) throw state.lastError
}
