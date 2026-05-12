/**
 * 图片生成Composable
 * 处理页面图片生成、滑动窗口参考图、三视图生成等
 */

import { ref, type Ref } from 'vue'
import type { PageContent } from '@/api/continuation'
import * as continuationApi from '@/api/continuation'
import type { ContinuationState } from './useContinuationState'
import { hasUsableStoryContent, isUsableImagePrompt, normalizeImagePrompt } from './promptValidation'

interface ImageGenerationComposable {
    isGenerating: Ref<boolean>
    generationProgress: Ref<number>
    batchGenerateImages: (pages: PageContent[], initialStyleReferenceTokens?: string[]) => Promise<void>
    regeneratePageImage: (pageNumber: number) => Promise<void>
}

export function useImageGeneration(bookId: Ref<string | undefined>, state: ContinuationState): ImageGenerationComposable {
    const isGenerating = ref(false)
    const generationProgress = ref(0)

    async function resolveStyleReferenceTokens(currentPageNumber: number): Promise<string[]> {
        if (!bookId.value) return []

        try {
            const availableResult = await continuationApi.getAvailableImages(
                bookId.value,
                'image'
            )

            if (availableResult.success) {
                const currentAbsolutePage = (availableResult.total_original_pages || 0) + currentPageNumber
                const merged = [
                    ...(availableResult.original_images || []),
                    ...(availableResult.continuation_images || []),
                ]
                    .filter(image => image.has_image && image.path && image.token)
                    .filter(image => image.page_number < currentAbsolutePage)
                    .sort((left, right) => left.page_number - right.page_number)
                    .map(image => image.token)

                if (merged.length > 0) {
                    return merged.slice(-state.styleRefPages.value)
                }
            }
        } catch (error) {
            console.warn('获取可用参考图失败，回退到默认逻辑:', error)
        }

        const styleResult = await continuationApi.getStyleReferences(bookId.value, state.styleRefPages.value)
        return styleResult.success && styleResult.tokens ? styleResult.tokens : []
    }

    async function batchGenerateImages(pages: PageContent[], initialStyleReferenceTokens?: string[]) {
        if (!bookId.value || pages.length === 0) return

        isGenerating.value = true
        generationProgress.value = 0

        const totalPages = pages.length
        let completedPages = 0

        try {
            // 确定初始画风参考图
            let styleReferenceTokens: string[]
            if (initialStyleReferenceTokens && initialStyleReferenceTokens.length > 0) {
                styleReferenceTokens = [...initialStyleReferenceTokens]
            } else {
                const firstPendingPage = pages.find(page => !page.image_url)?.page_number ?? 1
                styleReferenceTokens = await resolveStyleReferenceTokens(firstPendingPage)
            }

            for (const page of pages) {
                if (page.image_url) {
                    completedPages++
                    generationProgress.value = Math.round((completedPages / totalPages) * 100)
                    continue
                }

                page.final_prompt = normalizeImagePrompt(page.final_prompt)
                if (!hasUsableStoryContent(page) && !isUsableImagePrompt(page.final_prompt)) {
                    page.status = 'failed'
                    state.showMessage(`第 ${page.page_number} 页剧情或最终提示词无效，请先完善页面剧情或手动修改最终提示词`, 'error')
                    await continuationApi.savePages(bookId.value, pages)
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
                        styleReferenceTokens,
                        undefined,
                        state.styleRefPages.value
                    )

                    if (result.success && result.image_path) {
                        page.image_url = result.image_path
                        page.status = 'generated'

                        // 推进 token 滑动窗口，后续页自然可以解析到新生成的续写图。
                        const nextToken = `continuation:${page.page_number}`
                        if (styleReferenceTokens.length >= state.styleRefPages.value) {
                            styleReferenceTokens.shift()
                        }
                        styleReferenceTokens.push(nextToken)
                    } else {
                        page.status = 'failed'
                    }
                } catch (error) {
                    console.error(`生成第 ${page.page_number} 页失败:`, error)
                    page.status = 'failed'
                }

                await continuationApi.savePages(bookId.value, pages)

                completedPages++
                generationProgress.value = Math.round((completedPages / totalPages) * 100)
            }
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
            page.final_prompt = normalizeImagePrompt(page.final_prompt)

            if (!hasUsableStoryContent(page) && !isUsableImagePrompt(page.final_prompt)) {
                page.status = 'failed'
                await continuationApi.savePages(bookId.value, state.pages.value)
                state.showMessage(`第 ${pageNumber} 页剧情或最终提示词无效，请先完善页面剧情或手动修改最终提示词`, 'error')
                return
            }

            const styleReferenceTokens = await resolveStyleReferenceTokens(pageNumber)

            const result = await continuationApi.regeneratePageImage(
                bookId.value,
                pageNumber,
                page,
                styleReferenceTokens,
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
