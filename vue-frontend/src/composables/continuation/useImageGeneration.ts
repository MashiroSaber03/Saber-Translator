/**
 * 图片生成Composable
 * 处理页面图片生成、滑动窗口参考图、三视图生成等
 */

import { ref, type Ref } from 'vue'
import type { PageContent } from '@/api/continuation'
import * as continuationApi from '@/api/continuation'
import type { ContinuationState } from './useContinuationState'

interface ImageGenerationComposable {
    isGenerating: Ref<boolean>
    generationProgress: Ref<number>
    batchGenerateImages: (pages: PageContent[], initialStyleRefs?: string[]) => Promise<void>
    regeneratePageImage: (pageNumber: number) => Promise<void>
}

export function useImageGeneration(bookId: Ref<string | undefined>, state: ContinuationState): ImageGenerationComposable {
    const isGenerating = ref(false)
    const generationProgress = ref(0)

    async function resolveStyleReferences(currentPageNumber: number): Promise<string[]> {
        if (!bookId.value) return []

        try {
            const currentAbsolutePage = state.totalOriginalPages.value + currentPageNumber
            const availableResult = await continuationApi.getAvailableImages(
                bookId.value,
                'image',
                currentAbsolutePage
            )

            if (availableResult.success) {
                const merged = [
                    ...(availableResult.original_images || []),
                    ...(availableResult.continuation_images || []),
                ]
                    .filter(image => image.has_image && image.path)
                    .sort((left, right) => left.page_number - right.page_number)
                    .map(image => image.path)

                if (merged.length > 0) {
                    return merged.slice(-state.styleRefPages.value)
                }
            }
        } catch (error) {
            console.warn('获取可用参考图失败，回退到默认逻辑:', error)
        }

        const styleResult = await continuationApi.getStyleReferences(bookId.value, state.styleRefPages.value)
        return styleResult.success && styleResult.images ? styleResult.images : []
    }

    async function batchGenerateImages(pages: PageContent[], initialStyleRefs?: string[]) {
        if (!bookId.value || pages.length === 0) return

        isGenerating.value = true
        generationProgress.value = 0

        const totalPages = pages.length
        let completedPages = 0

        try {
            // 确定初始画风参考图
            let styleRefs: string[]
            if (initialStyleRefs && initialStyleRefs.length > 0) {
                // 使用用户选择的初始参考图
                styleRefs = [...initialStyleRefs]
            } else {
                const firstPendingPage = pages.find(page => !page.image_url)?.page_number ?? 1
                styleRefs = await resolveStyleReferences(firstPendingPage)
            }

            for (const page of pages) {
                if (page.image_url) {
                    completedPages++
                    generationProgress.value = Math.round((completedPages / totalPages) * 100)
                    continue
                }

                state.showMessage(`正在生成第 ${page.page_number}/${totalPages} 页图片...`, 'info')

                try {
                    const result = await continuationApi.generatePageImage(
                        bookId.value,
                        page.page_number,
                        page,
                        styleRefs,
                        undefined,
                        state.styleRefPages.value
                    )

                    if (result.success && result.image_path) {
                        page.image_url = result.image_path
                        page.status = 'generated'

                        // Update styleRefs滑动窗口
                        if (styleRefs.length >= state.styleRefPages.value) {
                            styleRefs.shift()
                        }
                        styleRefs.push(result.image_path)
                    } else {
                        page.status = 'failed'
                    }
                } catch (error) {
                    console.error(`生成第 ${page.page_number} 页失败:`, error)
                    page.status = 'failed'
                }

                completedPages++
                generationProgress.value = Math.round((completedPages / totalPages) * 100)
            }

            await continuationApi.savePages(bookId.value, pages)
            state.showMessage(`图片生成完成 (${completedPages}/${totalPages})`, 'success')
        } catch (error) {
            state.showMessage('批量生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
        } finally {
            isGenerating.value = false
            generationProgress.value = 0
        }
    }


    async function regeneratePageImage(pageNumber: number) {
        if (!bookId.value) return

        const page = state.pages.value.find(p => p.page_number === pageNumber)
        if (!page) return

        try {
            page.status = 'generating'

            const styleRefs = await resolveStyleReferences(pageNumber)

            const result = await continuationApi.regeneratePageImage(
                bookId.value,
                pageNumber,
                page,
                styleRefs,
                undefined,
                state.styleRefPages.value
            )

            if (result.success && result.image_path) {
                if (page.image_url) {
                    page.previous_url = page.image_url
                }
                page.image_url = result.image_path
                page.status = 'generated'

                await continuationApi.savePages(bookId.value, state.pages.value)
                state.showMessage(`第 ${pageNumber} 页图片已重新生成`, 'success')
            } else {
                page.status = 'failed'
                state.showMessage('重新生成失败: ' + result.error, 'error')
            }
        } catch (error) {
            page.status = 'failed'
            state.showMessage('重新生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
        }
    }

    return {
        isGenerating,
        generationProgress,
        batchGenerateImages,
        regeneratePageImage
    }
}
