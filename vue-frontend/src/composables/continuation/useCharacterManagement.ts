import { watch, type Ref } from 'vue'
import * as continuationApi from '@/api/continuation'
import type { ContinuationState } from './useContinuationState'
import {
    runContinuationMutation,
    type ContinuationMutationOptions,
} from './continuationActionRunner'

export interface CharacterManagementComposable {
    addCharacter: (name: string, aliases: string[], description: string) => Promise<boolean>
    deleteCharacter: (name: string) => Promise<boolean>
    updateCharacterInfo: (name: string, newName: string, aliases: string[]) => Promise<boolean>
    toggleCharacterEnabled: (name: string, enabled: boolean) => Promise<boolean>
    addForm: (charName: string, formName: string, description: string) => Promise<boolean>
    updateForm: (charName: string, formId: string, formName: string, description: string) => Promise<boolean>
    deleteForm: (charName: string, formId: string) => Promise<boolean>
    uploadFormImage: (charName: string, formId: string, file: File) => Promise<boolean>
    deleteFormImage: (charName: string, formId: string) => Promise<boolean>
    toggleFormEnabled: (charName: string, formId: string, enabled: boolean) => Promise<boolean>
    generateOrtho: (
        charName: string,
        formId: string,
        sourceImage: File,
    ) => Promise<string>
    setFormReference: (charName: string, formId: string, imagePath: string) => Promise<void>
}

export function useCharacterManagement(bookId: Ref<string | undefined>, state: ContinuationState): CharacterManagementComposable {
    let bookGeneration = 0
    watch(bookId, () => {
        bookGeneration += 1
    })

    function runBookMutation<T>(
        activeBookId: string,
        options: Omit<ContinuationMutationOptions<T>, 'state' | 'isCurrent'>,
    ): Promise<boolean> {
        const generation = bookGeneration
        return runContinuationMutation({
            ...options,
            state,
            isCurrent: () => bookGeneration === generation && bookId.value === activeBookId,
        })
    }

    async function addCharacter(name: string, aliases: string[], description: string) {
        const activeBookId = bookId.value
        if (!activeBookId) return false

        return runBookMutation(activeBookId, {
            failurePrefix: '添加失败',
            successMessage: '角色添加成功',
            run: () => continuationApi.addCharacter(activeBookId, {
                name,
                aliases,
                description
            }),
            afterSuccess: () => state.initializeData()
        })
    }

    async function deleteCharacter(name: string) {
        const activeBookId = bookId.value
        if (!activeBookId) return false

        return runBookMutation(activeBookId, {
            failurePrefix: '删除失败',
            successMessage: '角色删除成功',
            run: () => continuationApi.deleteCharacter(activeBookId, name),
            afterSuccess: () => state.initializeData()
        })
    }

    async function updateCharacterInfo(name: string, newName: string, aliases: string[]) {
        const activeBookId = bookId.value
        if (!activeBookId) return false

        return runBookMutation(activeBookId, {
            failurePrefix: '更新失败',
            successMessage: '角色信息更新成功',
            run: () => continuationApi.updateCharacterInfo(activeBookId, name, {
                name: newName,
                aliases
            }),
            afterSuccess: () => state.initializeData()
        })
    }

    async function toggleCharacterEnabled(name: string, enabled: boolean) {
        const activeBookId = bookId.value
        if (!activeBookId) return false

        const char = state.characters.value.find(c => c.name === name)
        if (!char) return false

        const previousEnabled = char.enabled
        char.enabled = enabled

        return runBookMutation(activeBookId, {
            failurePrefix: '操作失败',
            run: () => continuationApi.updateCharacterInfo(activeBookId, name, {
                name: char.name,
                aliases: char.aliases || [],
                enabled
            }),
            onFailure: () => {
                char.enabled = previousEnabled
            }
        })
    }

    async function addForm(charName: string, formName: string, description: string) {
        const activeBookId = bookId.value
        if (!activeBookId) return false

        return runBookMutation(activeBookId, {
            failurePrefix: '添加失败',
            successMessage: '形态添加成功',
            run: () => continuationApi.addCharacterForm(activeBookId, charName, {
                form_name: formName,
                description
            }),
            afterSuccess: () => state.initializeData()
        })
    }

    async function updateForm(charName: string, formId: string, formName: string, description: string) {
        const activeBookId = bookId.value
        if (!activeBookId) return false

        return runBookMutation(activeBookId, {
            failurePrefix: '更新失败',
            successMessage: '形态更新成功',
            run: () => continuationApi.updateCharacterForm(activeBookId, charName, formId, {
                form_name: formName,
                description
            }),
            afterSuccess: () => state.initializeData()
        })
    }

    async function deleteForm(charName: string, formId: string) {
        const activeBookId = bookId.value
        if (!activeBookId) return false

        return runBookMutation(activeBookId, {
            failurePrefix: '删除失败',
            successMessage: '形态删除成功',
            run: () => continuationApi.deleteCharacterForm(activeBookId, charName, formId),
            afterSuccess: () => state.initializeData()
        })
    }

    async function uploadFormImage(charName: string, formId: string, file: File) {
        const activeBookId = bookId.value
        if (!activeBookId) return false

        return runBookMutation(activeBookId, {
            failurePrefix: '上传失败',
            successMessage: '图片上传成功',
            run: () => continuationApi.uploadFormImage(activeBookId, charName, formId, file),
            afterSuccess: async () => {
                state.imageRefreshKey.value = Date.now()
                await state.initializeData()
            }
        })
    }

    async function deleteFormImage(charName: string, formId: string) {
        const activeBookId = bookId.value
        if (!activeBookId) return false

        return runBookMutation(activeBookId, {
            failurePrefix: '删除失败',
            successMessage: '图片删除成功',
            run: () => continuationApi.deleteFormImage(activeBookId, charName, formId),
            afterSuccess: async () => {
                state.imageRefreshKey.value = Date.now()
                await state.initializeData()
            }
        })
    }

    async function toggleFormEnabled(charName: string, formId: string, enabled: boolean) {
        const activeBookId = bookId.value
        if (!activeBookId) return false

        const char = state.characters.value.find(c => c.name === charName)
        if (!char) return false

        const form = char.forms?.find(f => f.form_id === formId)
        if (!form) return false

        const previousEnabled = form.enabled
        form.enabled = enabled

        return runBookMutation(activeBookId, {
            failurePrefix: '操作失败',
            run: () => continuationApi.toggleFormEnabled(activeBookId, charName, formId, enabled),
            onFailure: () => {
                form.enabled = previousEnabled
            }
        })
    }

    async function generateOrtho(charName: string, formId: string, sourceImage: File) {
        const activeBookId = bookId.value
        if (!activeBookId) throw new Error('当前未选择漫画')
        return continuationApi.generateFormOrtho(activeBookId, charName, formId, sourceImage)
    }

    async function setFormReference(charName: string, formId: string, imagePath: string) {
        const activeBookId = bookId.value
        if (!activeBookId) return
        const generation = bookGeneration

        await continuationApi.setFormReference(activeBookId, charName, formId, imagePath)
        if (bookGeneration !== generation || bookId.value !== activeBookId) return
        state.imageRefreshKey.value = Date.now()
        await state.initializeData()
    }

    return {
        addCharacter,
        deleteCharacter,
        updateCharacterInfo,
        toggleCharacterEnabled,
        addForm,
        updateForm,
        deleteForm,
        uploadFormImage,
        deleteFormImage,
        toggleFormEnabled,
        generateOrtho,
        setFormReference
    }
}
