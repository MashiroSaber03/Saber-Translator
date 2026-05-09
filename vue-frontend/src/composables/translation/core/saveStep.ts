/**
 * 自动保存步骤实现
 *
 * 统一使用 TaskContext + PersistenceService，
 * 不再直接从 imageStore 回读旧状态拼保存 payload。
 */

import { useSessionStore } from '@/stores/sessionStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import {
  createPipelineRuntime,
  getBookshelfSessionPath,
  hydrateTaskContextFromImage,
} from './runtime'
import { executeAtomicStep } from './atomicSteps'
import {
  isSessionInitialized as checkSessionInitialized,
  persistAllPages,
  persistSessionMeta,
} from './persistenceService'

let sessionPathCache: string | null = null
let preSaveCompleted = false

export function shouldEnableAutoSave(): boolean {
  const sessionStore = useSessionStore()
  const settingsStore = useSettingsStore()
  return settingsStore.settings.autoSaveInBookshelfMode && sessionStore.isBookshelfMode
}

export function getSessionPath(): string | null {
  const sessionStore = useSessionStore()
  return getBookshelfSessionPath(sessionStore.currentBookId, sessionStore.currentChapterId)
}

function getResolvedSessionPath(): string | null {
  const currentSessionPath = getSessionPath()
  if (currentSessionPath) {
    sessionPathCache = currentSessionPath
    return currentSessionPath
  }
  return sessionPathCache
}

function createSaveRuntime(forceEnabled: boolean = false) {
  const sessionStore = useSessionStore()
  const settingsStore = useSettingsStore()
  return createPipelineRuntime('standard', {
    settingsSnapshot: settingsStore.settings,
    autoSaveEnabled: forceEnabled || shouldEnableAutoSave(),
    sessionPath: getResolvedSessionPath(),
    bookId: sessionStore.currentBookId,
    chapterId: sessionStore.currentChapterId,
  })
}

function getTaskContextForImage(pageIndex: number) {
  const imageStore = useImageStore()
  const image = imageStore.images[pageIndex]
  if (!image) {
    throw new Error(`页面 ${pageIndex + 1} 不存在`)
  }
  const runtime = createSaveRuntime(true)
  return {
    runtime,
    context: hydrateTaskContextFromImage(pageIndex, image, 'standard', runtime),
  }
}

async function clearUnsavedFlag(pageIndex: number): Promise<void> {
  const imageStore = useImageStore()
  imageStore.updateImageByIndex(pageIndex, { hasUnsavedChanges: false })
}

interface InitializeSessionOptions {
  respectAutoSaveSetting: boolean
  markPreSaveCompleted?: boolean
  progressCallback?: PreSaveProgressCallback
}

async function initializeBookshelfSession(options: InitializeSessionOptions): Promise<boolean> {
  const imageStore = useImageStore()
  const { progressCallback, respectAutoSaveSetting, markPreSaveCompleted = false } = options

  if (respectAutoSaveSetting && !shouldEnableAutoSave()) {
    console.log('[AutoSave] 自动保存未启用或非书架模式，跳过预保存')
    return true
  }

  const runtime = createSaveRuntime(true)
  if (!runtime.sessionPath) {
    console.warn('[AutoSave] 缺少书籍/章节ID，跳过预保存')
    progressCallback?.onError?.('缺少书籍/章节ID')
    return false
  }

  const allImages = imageStore.images
  const totalImages = allImages.length
  if (totalImages === 0) {
    console.warn('[AutoSave] 没有图片，跳过预保存')
    progressCallback?.onError?.('没有图片')
    return false
  }

  const contexts = allImages.map((image, index) => hydrateTaskContextFromImage(index, image, 'standard', runtime))
  console.log(`[AutoSave] 预保存开始：${totalImages} 页（逐页保存）`)
  progressCallback?.onStart?.(totalImages)

  try {
    await persistAllPages(contexts, runtime, {
      includeOriginal: true,
      includeDerivedImagesFromSource: true,
      currentImageIndex: imageStore.currentImageIndex,
      onProgress: (current, total) => {
        progressCallback?.onProgress?.(current, total)
      },
    })

    for (let index = 0; index < imageStore.images.length; index++) {
      imageStore.updateImageByIndex(index, { hasUnsavedChanges: false })
    }

    sessionPathCache = runtime.sessionPath
    if (markPreSaveCompleted) {
      preSaveCompleted = true
    }

    console.log(`[AutoSave] 预保存完成，共保存 ${totalImages}/${totalImages} 页`)
    progressCallback?.onComplete?.()
    return true
  } catch (error) {
    console.error('[AutoSave] 预保存失败:', error)
    progressCallback?.onError?.(error instanceof Error ? error.message : '预保存失败')
    sessionPathCache = null
    preSaveCompleted = false
    return false
  }
}

export interface PreSaveProgressCallback {
  onStart?: (totalImages: number) => void
  onProgress?: (current: number, total: number) => void
  onComplete?: () => void
  onError?: (error: string) => void
}

export async function preSaveOriginalImages(
  progressCallback?: PreSaveProgressCallback
): Promise<boolean> {
  return initializeBookshelfSession({
    respectAutoSaveSetting: true,
    markPreSaveCompleted: true,
    progressCallback,
  })
}

export async function isBookshelfSessionInitialized(): Promise<boolean> {
  const runtime = createSaveRuntime(true)
  const initialized = await checkSessionInitialized(runtime)
  if (initialized && runtime.sessionPath) {
    sessionPathCache = runtime.sessionPath
  }
  return initialized
}

export async function forceInitializeBookshelfSession(): Promise<boolean> {
  return initializeBookshelfSession({
    respectAutoSaveSetting: false,
    markPreSaveCompleted: false,
  })
}

export async function saveBookshelfPageProgress(pageIndex: number, currentImageIndex: number): Promise<void> {
  const imageStore = useImageStore()
  const { runtime, context } = getTaskContextForImage(pageIndex)

  await executeAtomicStep('save', context, runtime)
  await clearUnsavedFlag(pageIndex)
  await persistSessionMeta(runtime, {
    totalPages: imageStore.images.length,
    currentImageIndex,
  })
}

export async function finalizeSave(): Promise<void> {
  if (!shouldEnableAutoSave()) {
    return
  }

  const runtime = createSaveRuntime(true)
  if (!runtime.sessionPath || !preSaveCompleted) {
    console.log('[AutoSave] 未执行预保存，跳过完成保存')
    return
  }

  const imageStore = useImageStore()
  try {
    await persistSessionMeta(runtime, {
      totalPages: imageStore.images.length,
      currentImageIndex: imageStore.currentImageIndex,
    })
    console.log('[AutoSave] 会话保存完成')
  } catch (error) {
    console.error('[AutoSave] 完成保存失败:', error)
  } finally {
    sessionPathCache = null
    preSaveCompleted = false
  }
}

export function resetSaveState(): void {
  sessionPathCache = null
  preSaveCompleted = false
  console.log('[AutoSave] 保存状态已重置')
}
