/**
 * 续写面板全局状态管理
 * 使用 provide/inject 模式在组件树中共享状态
 */

import { ref, readonly, type Ref, inject, provide, type InjectionKey } from 'vue'
import type { CharacterProfile, ChapterScript, PageContent } from '@/api/continuation'
import * as continuationApi from '@/api/continuation'

export interface ContinuationState {
    // 状态
    isLoading: Readonly<Ref<boolean>>
    isDataReady: Readonly<Ref<boolean>>
    currentStep: Ref<number>
    errorMessage: Ref<string>
    successMessage: Ref<string>

    // 配置
    pageCount: Ref<number>
    styleRefPages: Ref<number>
    continuationDirection: Ref<string>

    // 数据
    characters: Ref<CharacterProfile[]>
    chapterScript: Ref<ChapterScript | null>
    pages: Ref<PageContent[]>
    imageRefreshKey: Ref<number>
    totalOriginalPages: Ref<number>  // 原作总页数

    // 生成状态
    isGeneratingPages: Ref<boolean>
    isGeneratingPrompts: Ref<boolean>

    // 方法
    initializeData: () => Promise<void>
    resetState: () => Promise<void>
    showMessage: (message: string, type: 'success' | 'error' | 'info') => void

    // URL获取方法
    getCharacterImageUrl: (characterName: string) => string
    getFormImageUrl: (imagePath: string) => string
    getGeneratedImageUrl: (imagePath: string) => string
}

const ContinuationStateKey: InjectionKey<ContinuationState> = Symbol('ContinuationState')

export { ContinuationStateKey }

export function useContinuationState(bookId: Ref<string | undefined>): ContinuationState {
    const isLoading = ref(false)
    const isDataReady = ref(false)
    const currentStep = ref(0)
    const errorMessage = ref('')
    const successMessage = ref('')

    // 配置数据
    const pageCount = ref(10)
    const styleRefPages = ref(3)
    const continuationDirection = ref('')

    // 核心数据
    const characters = ref<CharacterProfile[]>([])
    const chapterScript = ref<ChapterScript | null>(null)
    const pages = ref<PageContent[]>([])
    const totalOriginalPages = ref<number>(0)  // 原作总页数
    // 生成状态
    const isGeneratingPages = ref(false)
    const isGeneratingPrompts = ref(false)

    // 图片刷新key
    const imageRefreshKey = ref(Date.now())

    async function initializeData() {
        if (!bookId.value) return

        isLoading.value = true

        try {
            const result = await continuationApi.prepareContinuation(bookId.value)

            if (result.success && result.saved_data) {
                const data = result.saved_data

                // 加载角色数据
                const charResult = await continuationApi.getCharacters(bookId.value)
                if (charResult.success && charResult.characters) {
                    characters.value = charResult.characters
                }

                chapterScript.value = data.script
                pages.value = data.pages || []

                if (data.config) {
                    pageCount.value = data.config.page_count || 10
                    styleRefPages.value = data.config.style_reference_pages || 3
                    continuationDirection.value = data.config.continuation_direction || ''
                }

                // 获取原作总页数
                try {
                    const availableResult = await continuationApi.getAvailableImages(bookId.value, 'script')
                    if (availableResult.success && availableResult.total_original_pages) {
                        totalOriginalPages.value = availableResult.total_original_pages
                    }
                } catch (e) {
                    console.warn('获取原作总页数失败:', e)
                }

                isDataReady.value = true
            }
        } catch (error) {
            console.error('初始化数据失败:', error)
            showMessage('初始化数据失败', 'error')
        } finally {
            isLoading.value = false
        }
    }

    async function resetState() {
        currentStep.value = 0
        isDataReady.value = false
        characters.value = []
        chapterScript.value = null
        pages.value = []
        pageCount.value = 10
        styleRefPages.value = 3
        continuationDirection.value = ''
        imageRefreshKey.value = Date.now()
        totalOriginalPages.value = 0
    }

    function showMessage(message: string, type: 'success' | 'error' | 'info' = 'info') {
        if (type === 'error') {
            errorMessage.value = message
            successMessage.value = ''
        } else if (type === 'success') {
            successMessage.value = message
            errorMessage.value = ''
        }

        setTimeout(() => {
            errorMessage.value = ''
            successMessage.value = ''
        }, 3000)
    }

    // URL获取方法
    function getCharacterImageUrl(characterName: string): string {
        if (!bookId.value) return ''
        return `/api/manga-insight/${bookId.value}/continuation/characters/${encodeURIComponent(characterName)}/image?t=${imageRefreshKey.value}`
    }

    function getFormImageUrl(imagePath: string): string {
        if (!bookId.value || !imagePath) return ''
        return `/api/manga-insight/file?path=${encodeURIComponent(imagePath)}&t=${imageRefreshKey.value}`
    }

    function getGeneratedImageUrl(imagePath: string): string {
        if (!bookId.value || !imagePath) return ''
        return `/api/manga-insight/${bookId.value}/continuation/generated-image?path=${encodeURIComponent(imagePath)}`
    }

    return {
        // 状态
        isLoading: readonly(isLoading),
        isDataReady: readonly(isDataReady),
        currentStep,
        errorMessage,
        successMessage,

        // 配置
        pageCount,
        styleRefPages,
        continuationDirection,

        // 数据
        characters,
        chapterScript,
        pages,
        imageRefreshKey,
        totalOriginalPages,

        // 生成状态
        isGeneratingPages,
        isGeneratingPrompts,

        // 方法
        initializeData,
        resetState,
        showMessage,

        // URL获取方法
        getCharacterImageUrl,
        getFormImageUrl,
        getGeneratedImageUrl
    }
}

export function provideContinuationState() {
    const bookId = inject<Ref<string>>('bookId')

    if (!bookId) {
        throw new Error('provideContinuationState must be used after bookId is provided')
    }

    const state = useContinuationState(bookId)
    provide(ContinuationStateKey, state)

    return state
}

export function useContinuationStateInject(): ContinuationState {
    const state = inject(ContinuationStateKey)

    if (!state) {
        throw new Error('useContinuationStateInject must be used after provideContinuationState')
    }

    return state
}
