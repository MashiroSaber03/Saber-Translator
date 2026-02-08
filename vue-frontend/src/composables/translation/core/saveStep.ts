/**
 * 自动保存步骤实现 - 重构版
 * 
 * 使用新的单页独立保存架构：
 * 1. 预保存阶段：保存所有页面的当前状态（原图 + 元数据）
 * 2. 单页保存阶段：每页翻译完成后只更新该页的译图和干净背景
 * 3. 完成阶段：更新会话元数据
 * 
 * 优势：
 * - 每页独立保存，无竞态问题
 * - 增量更新，只保存变化的内容
 * - 简单统一的逻辑
 */

import { useSessionStore } from '@/stores/sessionStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import {
    saveTranslatedPage,
    saveSessionMeta
} from '@/api/pageStorage'

// ============================================================
// 模块状态
// ============================================================

/** 当前会话路径缓存 */
let sessionPathCache: string | null = null

/** 是否已完成预保存 */
let preSaveCompleted = false

// ============================================================
// 工具函数
// ============================================================

/**
 * 检查是否应该启用自动保存
 */
export function shouldEnableAutoSave(): boolean {
    const sessionStore = useSessionStore()
    const settingsStore = useSettingsStore()

    return settingsStore.settings.autoSaveInBookshelfMode && sessionStore.isBookshelfMode
}

/**
 * 获取会话路径
 */
export function getSessionPath(): string | null {
    const sessionStore = useSessionStore()
    const bookId = sessionStore.currentBookId
    const chapterId = sessionStore.currentChapterId

    if (!bookId || !chapterId) {
        return null
    }

    // 使用新格式路径: bookshelf/{book_id}/chapters/{chapter_id}/session
    return `bookshelf/${bookId}/chapters/${chapterId}/session`
}

/**
 * 构建 UI 设置对象（仅保存文本样式设置）
 */
function buildUiSettings(): Record<string, unknown> {
    const settingsStore = useSettingsStore()
    const { textStyle } = settingsStore.settings

    return {
        fontSize: textStyle.fontSize,
        autoFontSize: textStyle.autoFontSize,
        fontFamily: textStyle.fontFamily,
        layoutDirection: textStyle.layoutDirection,
        textColor: textStyle.textColor,
        useInpaintingMethod: textStyle.inpaintMethod,
        fillColor: textStyle.fillColor,
        strokeEnabled: textStyle.strokeEnabled,
        strokeColor: textStyle.strokeColor,
        strokeWidth: textStyle.strokeWidth,
        useAutoTextColor: textStyle.useAutoTextColor,
    }
}

// ============================================================
// 预保存阶段
// ============================================================

/** 预保存进度回调 */
export interface PreSaveProgressCallback {
    onStart?: (totalImages: number) => void
    onProgress?: (current: number, total: number) => void
    onComplete?: () => void
    onError?: (error: string) => void
}

/**
 * 预保存所有页面（逐页保存，显示进度）
 * 
 * 保存所有图片的当前状态（原图、已有的译图、元数据）
 * 
 * @param progressCallback 进度回调
 * @returns 是否成功
 */
export async function preSaveOriginalImages(
    progressCallback?: PreSaveProgressCallback
): Promise<boolean> {
    const imageStore = useImageStore()

    // 检查是否应该启用
    if (!shouldEnableAutoSave()) {
        console.log('[AutoSave] 自动保存未启用或非书架模式，跳过预保存')
        return true
    }

    const sessionPath = getSessionPath()
    if (!sessionPath) {
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

    console.log(`[AutoSave] 预保存开始：${totalImages} 页（逐页保存）`)
    progressCallback?.onStart?.(totalImages)

    try {
        // 导入公共保存函数
        const { saveAllPagesSequentially, saveSessionMeta } = await import('@/api/pageStorage')

        const uiSettings = buildUiSettings()

        // 使用公共函数逐页保存
        const savedCount = await saveAllPagesSequentially(
            sessionPath,
            allImages as unknown as import('@/api/pageStorage').ImageDataForSave[],
            {
                onProgress: (current, total) => {
                    progressCallback?.onProgress?.(current, total)
                }
            }
        )

        // 保存会话元数据
        await saveSessionMeta(sessionPath, {
            ui_settings: uiSettings,
            total_pages: totalImages,
            currentImageIndex: imageStore.currentImageIndex
        })

        // 更新书架章节的图片数量
        const sessionStore = useSessionStore()
        const bookId = sessionStore.currentBookId
        const chapterId = sessionStore.currentChapterId
        if (bookId && chapterId) {
            try {
                const { apiClient } = await import('@/api/client')
                await apiClient.put(`/api/bookshelf/books/${bookId}/chapters/${chapterId}/image-count`, {
                    count: totalImages
                })
                console.log(`[AutoSave] 已更新章节图片数量为 ${totalImages}`)
            } catch (e) {
                console.warn('[AutoSave] 更新章节图片数量失败（非致命）:', e)
            }
        }

        sessionPathCache = sessionPath
        preSaveCompleted = true

        console.log(`[AutoSave] 预保存完成，共保存 ${savedCount}/${totalImages} 页`)
        progressCallback?.onComplete?.()

        return true

    } catch (error) {
        console.error('[AutoSave] 预保存失败:', error)
        const errorMsg = error instanceof Error ? error.message : '预保存失败'
        progressCallback?.onError?.(errorMsg)
        sessionPathCache = null
        preSaveCompleted = false
        return false
    }
}

// ============================================================
// 单页保存阶段
// ============================================================

/**
 * 保存翻译完成的页面
 * 
 * 在每页翻译完成后调用，只保存该页的译图和干净背景
 * 
 * @param pageIndex 页面索引（原始索引，0-based）
 */
export async function saveTranslatedImage(pageIndex: number): Promise<void> {
    // 检查是否应该启用
    if (!shouldEnableAutoSave()) {
        return
    }

    const sessionPath = sessionPathCache || getSessionPath()
    if (!sessionPath) {
        console.warn('[AutoSave] 会话路径不存在，跳过保存')
        return
    }

    const imageStore = useImageStore()
    const img = imageStore.images[pageIndex]
    if (!img) {
        console.warn(`[AutoSave] 页面 ${pageIndex} 不存在`)
        return
    }

    try {
        // 准备要保存的数据
        const saveData: {
            translated?: string
            clean?: string
            meta?: Record<string, unknown>
        } = {}

        // 译图
        if (img.translatedDataURL?.startsWith('data:')) {
            saveData.translated = img.translatedDataURL
        }

        // 干净背景
        if (img.cleanImageData && typeof img.cleanImageData === 'string' && !img.cleanImageData.startsWith('/api/')) {
            saveData.clean = img.cleanImageData.startsWith('data:')
                ? img.cleanImageData
                : `data:image/png;base64,${img.cleanImageData}`
        }

        // 页面元数据（不含图片数据）
        const pageMeta: Record<string, unknown> = {}
        for (const key of Object.keys(img)) {
            if (!['originalDataURL', 'translatedDataURL', 'cleanImageData'].includes(key)) {
                pageMeta[key] = img[key as keyof typeof img]
            }
        }
        saveData.meta = pageMeta

        // 调用保存 API
        const result = await saveTranslatedPage(sessionPath, pageIndex, saveData)

        if (!result.success) {
            throw new Error(result.error || '保存失败')
        }

        // 标记该页已保存
        imageStore.updateImageByIndex(pageIndex, { hasUnsavedChanges: false })

        console.log(`[AutoSave] 页面 ${pageIndex + 1} 已保存`)

    } catch (error) {
        console.error(`[AutoSave] 保存页面 ${pageIndex + 1} 失败:`, error)
        throw error
    }
}

// ============================================================
// 完成阶段
// ============================================================

/**
 * 完成保存
 * 
 * 在所有翻译完成后调用，更新会话元数据
 */
export async function finalizeSave(): Promise<void> {
    // 检查是否应该启用
    if (!shouldEnableAutoSave()) {
        return
    }

    const sessionPath = sessionPathCache || getSessionPath()
    if (!sessionPath || !preSaveCompleted) {
        console.log('[AutoSave] 未执行预保存，跳过完成保存')
        return
    }

    console.log('[AutoSave] 完成保存...')

    const imageStore = useImageStore()
    const sessionStore = useSessionStore()
    const totalImages = imageStore.images.length

    try {
        // 更新会话元数据
        await saveSessionMeta(sessionPath, {
            ui_settings: buildUiSettings(),
            total_pages: totalImages,
            currentImageIndex: imageStore.currentImageIndex
        })

        // 更新书架章节的图片数量
        const bookId = sessionStore.currentBookId
        const chapterId = sessionStore.currentChapterId
        if (bookId && chapterId) {
            try {
                const { apiClient } = await import('@/api/client')
                await apiClient.put(`/api/bookshelf/books/${bookId}/chapters/${chapterId}/image-count`, {
                    count: totalImages
                })
                console.log(`[AutoSave] 已更新章节图片数量为 ${totalImages}`)
            } catch (e) {
                console.warn('[AutoSave] 更新章节图片数量失败（非致命）:', e)
            }
        }

        console.log('[AutoSave] 会话保存完成')

    } catch (error) {
        console.error('[AutoSave] 完成保存失败:', error)
    } finally {
        // 重置状态
        sessionPathCache = null
        preSaveCompleted = false
    }
}

// ============================================================
// 状态管理
// ============================================================

/**
 * 重置保存状态（取消翻译时调用）
 */
export function resetSaveState(): void {
    sessionPathCache = null
    preSaveCompleted = false
    console.log('[AutoSave] 保存状态已重置')
}

/**
 * 检查是否已完成预保存
 */
export function isPreSaveCompleted(): boolean {
    return preSaveCompleted
}

/**
 * 获取当前会话路径
 */
export function getCurrentSessionPath(): string | null {
    return sessionPathCache
}
