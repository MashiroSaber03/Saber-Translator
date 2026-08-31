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
    initialReferenceTokens: Ref<string[]>

    characters: Ref<CharacterProfile[]>
    hasMoreCharacterForms: Readonly<Ref<boolean>>
    isLoadingMoreCharacterForms: Readonly<Ref<boolean>>
    chapterScript: Ref<ChapterScript | null>
    pages: Ref<PageContent[]>
    imageRefreshKey: Ref<number>

    isGeneratingPages: Ref<boolean>

    initializeData: () => Promise<void>
    loadMoreCharacterForms: () => Promise<void>
    syncAnalysisData: (source?: 'auto' | 'manual') => Promise<void>
    resetState: () => void
    showMessage: (message: string, type: 'success' | 'error' | 'info') => void

    getCharacterImageUrl: (characterName: string) => string
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
    const initialReferenceTokens = ref<string[]>([])

    const characters = ref<CharacterProfile[]>([])
    const hasMoreCharacterForms = ref(false)
    const isLoadingMoreCharacterForms = ref(false)
    const chapterScript = ref<ChapterScript | null>(null)
    const pages = ref<PageContent[]>([])
    const isGeneratingPages = ref(false)
    const imageRefreshKey = ref(Date.now())
    let initializeRequestId = 0
    let syncRequestId = 0
    let formPageRequestId = 0
    let characterFormCursor: number | null = null
    let isMounted = true

    function resetLoadedContinuationData(): void {
        isDataReady.value = false
        isSyncingAnalysis.value = false
        characters.value = []
        characterFormCursor = null
        hasMoreCharacterForms.value = false
        isLoadingMoreCharacterForms.value = false
        formPageRequestId += 1
        chapterScript.value = null
        pages.value = []
        pageCount.value = 10
        styleRefPages.value = 3
        continuationDirection.value = ''
        initialReferenceTokens.value = []
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

        try {
            const charactersResult = await continuationApi.getCharacters(activeBookId)
            if (!isCurrentRequest()) {
                return false
            }
            characters.value = charactersResult.items
            characterFormCursor = charactersResult.nextCursor
            hasMoreCharacterForms.value = characterFormCursor !== null
            imageRefreshKey.value = Date.now()
            return true
        } catch (error) {
            if (isCurrentRequest()) {
                const message = error instanceof Error ? error.message : '网络错误'
                setMessageState(`加载角色失败：${message}`, 'error', persistentError)
            }
            return false
        }
    }

    function mergeCharacterFormPage(nextCharacters: CharacterProfile[]): void {
        const currentByName = new Map(characters.value.map(character => [character.name, character]))
        characters.value = nextCharacters.map(character => {
            const current = currentByName.get(character.name)
            if (!current) return character

            const formIds = new Set(current.forms.map(form => form.form_id))
            return {
                ...character,
                forms: [
                    ...current.forms,
                    ...character.forms.filter(form => !formIds.has(form.form_id)),
                ],
                reference_image: character.reference_image || current.reference_image,
            }
        })
    }

    async function loadMoreCharacterForms(): Promise<void> {
        const activeBookId = bookId.value
        const cursor = characterFormCursor
        if (
            !activeBookId
            || cursor === null
            || !hasMoreCharacterForms.value
            || isLoadingMoreCharacterForms.value
        ) {
            return
        }
        const requestId = ++formPageRequestId
        isLoadingMoreCharacterForms.value = true
        try {
            const charactersResult = await continuationApi.getCharacters(activeBookId, cursor)
            if (!isMounted || requestId !== formPageRequestId || bookId.value !== activeBookId) {
                return
            }
            mergeCharacterFormPage(charactersResult.items)
            characterFormCursor = charactersResult.nextCursor
            hasMoreCharacterForms.value = characterFormCursor !== null
            imageRefreshKey.value = Date.now()
        } catch (error) {
            if (isMounted && requestId === formPageRequestId && bookId.value === activeBookId) {
                const message = error instanceof Error ? error.message : '网络错误'
                showMessage(`加载更多角色形态失败：${message}`, 'error')
            }
        } finally {
            if (isMounted && requestId === formPageRequestId) {
                isLoadingMoreCharacterForms.value = false
            }
        }
    }

    function applySavedContinuationData(data: {
        script: ChapterScript | null
        pages: PageContent[]
        config: {
            page_count?: number
            style_reference_pages?: number
            continuation_direction?: string
        } | null
        reference_tokens?: string[]
    }): void {
        chapterScript.value = data.script
        pages.value = data.pages || []
        initialReferenceTokens.value = [...(data.reference_tokens ?? [])]

        if (data.config) {
            pageCount.value = data.config.page_count ?? 10
            styleRefPages.value = data.config.style_reference_pages ?? 3
            continuationDirection.value = data.config.continuation_direction ?? ''
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
        syncRequestId += 1
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

            applySavedContinuationData(result.saved_data)
            applyPreparationResult(result)
            await loadCharactersForBook(activeBookId, true, isCurrentRequest)
        } catch (error) {
            if (isCurrentRequest()) {
                const message = error instanceof Error ? error.message : '网络错误'
                setMessageState(`初始化数据失败：${message}`, 'error', true)
            }
        } finally {
            if (isCurrentRequest()) {
                isLoading.value = false
            }
        }
    }

    async function syncAnalysisData(source: 'auto' | 'manual' = 'manual') {
        const activeBookId = bookId.value
        if (!activeBookId || isSyncingAnalysis.value) return
        const requestId = ++syncRequestId
        const isCurrentRequest = () => isMounted && requestId === syncRequestId && bookId.value === activeBookId

        const hasContinuationPayload = Boolean(chapterScript.value) || pages.value.length > 0
        isSyncingAnalysis.value = true

        try {
            const result = await continuationApi.syncContinuationAnalysis(activeBookId)
            if (!isCurrentRequest()) {
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
        } catch (error) {
            if (isCurrentRequest()) {
                const message = error instanceof Error ? error.message : '网络错误'
                setMessageState(`同步分析数据失败：${message}`, 'error', true)
            }
        } finally {
            if (isCurrentRequest()) {
                isSyncingAnalysis.value = false
            }
        }
    }

    function resetState() {
        initializeRequestId += 1
        syncRequestId += 1
        formPageRequestId += 1
        clearMessageTimer()
        isLoading.value = false
        currentStep.value = 0
        messageType.value = ''
        errorMessage.value = ''
        successMessage.value = ''
        resetLoadedContinuationData()
    }

    function showMessage(message: string, type: 'success' | 'error' | 'info' = 'info') {
        setMessageState(message, type, false)
    }

    function getCharacterImageUrl(characterName: string): string {
        return characters.value.find(character => character.name === characterName)?.reference_image ?? ''
    }

    if (getCurrentInstance()) {
        onBeforeUnmount(() => {
            isMounted = false
            initializeRequestId += 1
            syncRequestId += 1
            formPageRequestId += 1
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
        initialReferenceTokens,

        characters,
        hasMoreCharacterForms: readonly(hasMoreCharacterForms),
        isLoadingMoreCharacterForms: readonly(isLoadingMoreCharacterForms),
        chapterScript,
        pages,
        imageRefreshKey,

        isGeneratingPages,

        initializeData,
        loadMoreCharacterForms,
        syncAnalysisData,
        resetState,
        showMessage,

        getCharacterImageUrl
    }
}
