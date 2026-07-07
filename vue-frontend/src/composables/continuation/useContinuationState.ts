import { getCurrentInstance, onBeforeUnmount, ref, readonly, type Ref } from 'vue'
import type { CharacterProfile, ChapterScript, PageContent } from '@/api/continuation'
import * as continuationApi from '@/api/continuation'

export interface ContinuationState {
    isLoading: Readonly<Ref<boolean>>
    isDataReady: Readonly<Ref<boolean>>
    isSyncingAnalysis: Readonly<Ref<boolean>>
    currentStep: Ref<number>
    messageType: Ref<'success' | 'error' | 'info' | ''>
    errorMessage: Ref<string>
    successMessage: Ref<string>
    lastAnalysisSyncAt: Ref<string>

    pageCount: Ref<number>
    styleRefPages: Ref<number>
    continuationDirection: Ref<string>

    characters: Ref<CharacterProfile[]>
    chapterScript: Ref<ChapterScript | null>
    pages: Ref<PageContent[]>
    imageRefreshKey: Ref<number>

    isGeneratingPages: Ref<boolean>

    initializeData: () => Promise<void>
    syncAnalysisData: (source?: 'auto' | 'manual') => Promise<void>
    resetState: () => void
    showMessage: (message: string, type: 'success' | 'error' | 'info') => void

    getCharacterImageUrl: (characterName: string) => string
    getFormImageUrl: (imagePath: string) => string
    getGeneratedImageUrl: (imagePath: string) => string
}

export function useContinuationState(bookId: Ref<string | undefined>): ContinuationState {
    const isLoading = ref(false)
    const isDataReady = ref(false)
    const isSyncingAnalysis = ref(false)
    const currentStep = ref(0)
    const messageType = ref<'success' | 'error' | 'info' | ''>('')
    const errorMessage = ref('')
    const successMessage = ref('')
    const lastAnalysisSyncAt = ref('')
    let messageTimer: ReturnType<typeof setTimeout> | null = null

    const pageCount = ref(10)
    const styleRefPages = ref(3)
    const continuationDirection = ref('')

    const characters = ref<CharacterProfile[]>([])
    const chapterScript = ref<ChapterScript | null>(null)
    const pages = ref<PageContent[]>([])
    const isGeneratingPages = ref(false)
    const imageRefreshKey = ref(Date.now())
    let initializeRequestId = 0
    let syncRequestId = 0
    let isMounted = true

    function resetLoadedContinuationData(): void {
        isDataReady.value = false
        isSyncingAnalysis.value = false
        characters.value = []
        chapterScript.value = null
        pages.value = []
        pageCount.value = 10
        styleRefPages.value = 3
        continuationDirection.value = ''
        lastAnalysisSyncAt.value = ''
        isGeneratingPages.value = false
        imageRefreshKey.value = Date.now()
    }

    function clearMessageTimer(): void {
        if (messageTimer) {
            clearTimeout(messageTimer)
            messageTimer = null
        }
    }

    function setMessageState(message: string, type: 'success' | 'error' | 'info', persistent: boolean): void {
        clearMessageTimer()

        messageType.value = type
        if (type === 'error') {
            errorMessage.value = message
            successMessage.value = ''
        } else {
            successMessage.value = message
            errorMessage.value = ''
        }

        if (!persistent) {
            messageTimer = setTimeout(() => {
                messageType.value = ''
                errorMessage.value = ''
                successMessage.value = ''
                messageTimer = null
            }, 3000)
        }
    }

    async function loadCharactersForBook(
        activeBookId: string,
        persistentError: boolean,
        isCurrentRequest: () => boolean
    ): Promise<boolean> {
        if (!activeBookId) return false

        const charResult = await continuationApi.getCharacters(activeBookId)
        if (!isCurrentRequest()) {
            return false
        }
        if (charResult.success && charResult.characters) {
            characters.value = charResult.characters
            imageRefreshKey.value = Date.now()
            return true
        }

        if (!charResult.success && charResult.error) {
            setMessageState(`加载角色失败：${charResult.error}`, 'error', persistentError)
        }
        return false
    }

    function applySavedContinuationData(data: {
        script: ChapterScript | null
        pages: PageContent[]
        config: {
            page_count?: number
            style_reference_pages?: number
            continuation_direction?: string
        } | null
    }): void {
        chapterScript.value = data.script
        pages.value = data.pages || []

        if (data.config) {
            pageCount.value = data.config.page_count || 10
            styleRefPages.value = data.config.style_reference_pages || 3
            continuationDirection.value = data.config.continuation_direction || ''
        }
    }

    function applyPreparationResult(result: {
        ready?: boolean
        message?: string
        synced_at?: string
    }, persistentMessage: boolean = true): void {
        isDataReady.value = Boolean(result.ready)
        if (result.synced_at) {
            lastAnalysisSyncAt.value = result.synced_at
        }

        if (!result.ready && result.message) {
            setMessageState(result.message, 'error', persistentMessage)
            return
        }

        if (result.ready && messageType.value === 'error' && errorMessage.value && !errorMessage.value.startsWith('加载角色失败：')) {
            messageType.value = ''
            errorMessage.value = ''
        }
    }

    async function initializeData() {
        const activeBookId = bookId.value
        if (!activeBookId) return
        const requestId = ++initializeRequestId
        const isCurrentRequest = () => isMounted && requestId === initializeRequestId && bookId.value === activeBookId

        clearMessageTimer()

        isLoading.value = true
        messageType.value = ''
        errorMessage.value = ''
        successMessage.value = ''
        resetLoadedContinuationData()

        try {
            const result = await continuationApi.prepareContinuation(activeBookId)
            if (!isCurrentRequest()) {
                return
            }

            if (result.success && result.saved_data) {
                const data = result.saved_data
                applySavedContinuationData(data)
                applyPreparationResult(result)
                await loadCharactersForBook(activeBookId, true, isCurrentRequest)
            } else if (!result.success && result.error) {
                setMessageState(result.error, 'error', true)
            }
        } catch {
            if (isCurrentRequest()) {
                setMessageState('初始化数据失败', 'error', true)
            }
        } finally {
            if (isCurrentRequest()) {
                isLoading.value = false
            }
        }
    }

    async function syncAnalysisData(source: 'auto' | 'manual' = 'manual') {
        const activeBookId = bookId.value
        if (!activeBookId) return
        const requestId = ++syncRequestId
        const isCurrentRequest = () => isMounted && requestId === syncRequestId && bookId.value === activeBookId

        const hasContinuationPayload = Boolean(chapterScript.value) || pages.value.length > 0
        isSyncingAnalysis.value = true

        try {
            const result = await continuationApi.syncContinuationAnalysis(activeBookId)
            if (!isCurrentRequest()) {
                return
            }

            if (!result.success) {
                const message = result.error || '分析数据同步失败'
                setMessageState(message, 'error', true)
                return
            }

            applyPreparationResult(result)
            const charactersLoaded = await loadCharactersForBook(activeBookId, true, isCurrentRequest)

            if (!result.ready) {
                return
            }

            if (source === 'manual' && charactersLoaded) {
                const successText = hasContinuationPayload
                    ? '已同步到最新分析数据，现有续写内容已保留'
                    : (result.message || '分析数据同步完成')
                showMessage(successText, 'success')
            }
        } finally {
            if (isCurrentRequest()) {
                isSyncingAnalysis.value = false
            }
        }
    }

    function resetState() {
        clearMessageTimer()
        currentStep.value = 0
        messageType.value = ''
        errorMessage.value = ''
        successMessage.value = ''
        resetLoadedContinuationData()
    }

    function showMessage(message: string, type: 'success' | 'error' | 'info' = 'info') {
        setMessageState(message, type, false)
    }

    function getBookPathSegment(): string | null {
        return bookId.value ? encodeURIComponent(bookId.value) : null
    }

    function getCharacterImageUrl(characterName: string): string {
        const bookPathSegment = getBookPathSegment()
        if (!bookPathSegment) return ''
        return `/api/manga-insight/${bookPathSegment}/continuation/characters/${encodeURIComponent(characterName)}/image?t=${imageRefreshKey.value}`
    }

    function getFormImageUrl(imagePath: string): string {
        if (!bookId.value || !imagePath) return ''
        return `/api/manga-insight/file?path=${encodeURIComponent(imagePath)}&t=${imageRefreshKey.value}`
    }

    function getGeneratedImageUrl(imagePath: string): string {
        const bookPathSegment = getBookPathSegment()
        if (!bookPathSegment || !imagePath) return ''
        return `/api/manga-insight/${bookPathSegment}/continuation/generated-image?path=${encodeURIComponent(imagePath)}`
    }

    if (getCurrentInstance()) {
        onBeforeUnmount(() => {
            isMounted = false
            initializeRequestId += 1
            syncRequestId += 1
            clearMessageTimer()
        })
    }

    return {
        isLoading: readonly(isLoading),
        isDataReady: readonly(isDataReady),
        isSyncingAnalysis: readonly(isSyncingAnalysis),
        currentStep,
        messageType,
        errorMessage,
        successMessage,
        lastAnalysisSyncAt,

        pageCount,
        styleRefPages,
        continuationDirection,

        characters,
        chapterScript,
        pages,
        imageRefreshKey,

        isGeneratingPages,

        initializeData,
        syncAnalysisData,
        resetState,
        showMessage,

        getCharacterImageUrl,
        getFormImageUrl,
        getGeneratedImageUrl
    }
}
