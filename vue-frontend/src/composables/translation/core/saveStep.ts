/**
 * 自动保存步骤实现
 * 
 * 核心功能：
 * 1. 预保存阶段：翻译开始前保存所有原始图片
 * 2. 单图保存阶段：每张图片渲染完成后保存译图
 * 3. 完成保存阶段：更新元数据并完成会话
 * 
 * 仅在书架模式下可用（快速翻译模式不支持）
 */

import { useSessionStore } from '@/stores/sessionStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { batchSaveStartApi, batchSaveImageApi, batchSaveCompleteApi } from '@/api/session'

// ============================================================
// 模块状态（翻译流程中共享）
// ============================================================

/** 会话文件夹路径缓存 */
let sessionFolderCache: string | null = null

/** 是否已经执行过预保存 */
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
function getSessionPath(): string | null {
    const sessionStore = useSessionStore()
    const bookId = sessionStore.currentBookId
    const chapterId = sessionStore.currentChapterId

    if (!bookId || !chapterId) {
        return null
    }

    return `bookshelf/${bookId}/${chapterId}`
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
 * 预保存阶段：保存所有原始图片
 * 在翻译开始前调用一次
 * 
 * @param progressCallback 进度回调（可选）
 * @returns 是否成功（失败不阻止翻译流程）
 */
export async function preSaveOriginalImages(
    progressCallback?: PreSaveProgressCallback
): Promise<boolean> {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()

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
    if (allImages.length === 0) {
        console.warn('[AutoSave] 没有图片，跳过预保存')
        progressCallback?.onError?.('没有图片')
        return false
    }

    console.log(`[AutoSave] 预保存开始：${allImages.length} 张原图`)
    progressCallback?.onStart?.(allImages.length)

    try {
        // 1. 构建图片元数据
        const imagesMeta = allImages.map((img, _index) => {
            const meta: Record<string, unknown> = {}
            for (const key of Object.keys(img)) {
                if (!['originalDataURL', 'translatedDataURL', 'cleanImageData'].includes(key)) {
                    meta[key] = img[key as keyof typeof img]
                }
            }
            meta.hasOriginalData = !!img.originalDataURL
            meta.hasTranslatedData = !!img.translatedDataURL
            meta.hasCleanData = !!img.cleanImageData
            return meta
        })

        // 2. 构建 UI 设置
        const { textStyle, targetLanguage, sourceLanguage } = settingsStore.settings
        const uiSettings: Record<string, unknown> = {
            targetLanguage,
            sourceLanguage,
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

        // 3. 调用开始保存 API
        const startResponse = await batchSaveStartApi(sessionPath, {
            ui_settings: uiSettings,
            images_meta: imagesMeta,
            currentImageIndex: imageStore.currentImageIndex
        })

        if (!startResponse.success || !startResponse.session_folder) {
            throw new Error(startResponse.error || '初始化保存失败')
        }

        sessionFolderCache = startResponse.session_folder
        console.log(`[AutoSave] 会话文件夹：${sessionFolderCache}`)

        // 4. 逐张保存原图（带进度回调）
        for (let i = 0; i < allImages.length; i++) {
            const img = allImages[i]
            if (img?.originalDataURL?.startsWith('data:')) {
                try {
                    await batchSaveImageApi(sessionFolderCache, i, 'original', img.originalDataURL)
                    console.log(`[AutoSave] 原图 ${i + 1}/${allImages.length} 已保存`)
                    progressCallback?.onProgress?.(i + 1, allImages.length)
                } catch (err) {
                    console.error(`[AutoSave] 保存原图 ${i + 1} 失败:`, err)
                    // 单张失败不中断整体流程，但仍更新进度
                    progressCallback?.onProgress?.(i + 1, allImages.length)
                }
            } else {
                // 图片没有有效数据，跳过但仍更新进度
                progressCallback?.onProgress?.(i + 1, allImages.length)
            }
        }

        preSaveCompleted = true
        console.log('[AutoSave] 预保存完成')
        progressCallback?.onComplete?.()
        return true

    } catch (error) {
        console.error('[AutoSave] 预保存失败:', error)
        const errorMsg = error instanceof Error ? error.message : '预保存失败'
        progressCallback?.onError?.(errorMsg)
        sessionFolderCache = null
        preSaveCompleted = false
        return false
    }
}

// ============================================================
// 单图保存阶段
// ============================================================

/**
 * 单图保存阶段：保存指定图片的译图和干净背景
 * 在每张图片渲染完成后调用
 * 
 * @param imageIndex 图片索引
 */
export async function saveTranslatedImage(imageIndex: number): Promise<void> {
    // 检查是否应该启用
    if (!shouldEnableAutoSave()) {
        return
    }

    if (!sessionFolderCache) {
        console.warn('[AutoSave] 会话文件夹未初始化，跳过保存')
        return
    }

    const imageStore = useImageStore()
    const img = imageStore.images[imageIndex]
    if (!img) {
        console.warn(`[AutoSave] 图片 ${imageIndex} 不存在`)
        return
    }

    try {
        // 保存译图
        if (img.translatedDataURL?.startsWith('data:')) {
            await batchSaveImageApi(sessionFolderCache, imageIndex, 'translated', img.translatedDataURL)
        }

        // 保存干净背景
        // cleanImageData 可能是纯 Base64（不带 data: 前缀）或带前缀的格式
        if (img.cleanImageData && !img.cleanImageData.startsWith('/api/')) {
            // 如果是纯 Base64，需要添加前缀后保存
            const cleanData = img.cleanImageData.startsWith('data:')
                ? img.cleanImageData
                : `data:image/png;base64,${img.cleanImageData}`
            await batchSaveImageApi(sessionFolderCache, imageIndex, 'clean', cleanData)
        }

        // 标记该图片已保存（清除未保存标记）
        imageStore.updateImageByIndex(imageIndex, { hasUnsavedChanges: false })

    } catch (error) {
        console.error(`[AutoSave] 保存译图 ${imageIndex + 1} 失败:`, error)
        // 向上抛出，让管线捕获并记录失败状态
        throw error
    }
}

// ============================================================
// 完成保存阶段
// ============================================================

/**
 * 完成保存阶段：更新元数据并完成会话
 * 在所有翻译完成后调用
 */
export async function finalizeSave(): Promise<void> {
    // 检查是否应该启用
    if (!shouldEnableAutoSave()) {
        return
    }

    if (!sessionFolderCache || !preSaveCompleted) {
        console.log('[AutoSave] 未执行预保存或会话文件夹不存在，跳过完成保存')
        return
    }

    console.log('[AutoSave] 完成保存...')

    const imageStore = useImageStore()

    try {
        // 收集最终元数据
        const imagesMeta = imageStore.images.map((img) => {
            const meta: Record<string, unknown> = {}
            for (const key of Object.keys(img)) {
                if (!['originalDataURL', 'translatedDataURL', 'cleanImageData'].includes(key)) {
                    meta[key] = img[key as keyof typeof img]
                }
            }
            meta.hasOriginalData = !!img.originalDataURL
            meta.hasTranslatedData = !!img.translatedDataURL
            meta.hasCleanData = !!img.cleanImageData
            return meta
        })

        await batchSaveCompleteApi(sessionFolderCache, imagesMeta)
        console.log('[AutoSave] 会话保存完成')

    } catch (error) {
        console.error('[AutoSave] 完成保存失败:', error)
    } finally {
        // 重置状态
        sessionFolderCache = null
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
    sessionFolderCache = null
    preSaveCompleted = false
    console.log('[AutoSave] 保存状态已重置')
}
