import type { Ref } from 'vue'
import * as continuationApi from '@/api/continuation'
import type { ContinuationState } from './useContinuationState'
import { runContinuationMutation, toContinuationActionError } from './continuationActionRunner'

export interface CharacterManagementComposable {
    addCharacter: (name: string, aliases: string[], description: string) => Promise<void>
    deleteCharacter: (name: string) => Promise<void>
    updateCharacterInfo: (name: string, newName: string, aliases: string[]) => Promise<void>
    toggleCharacterEnabled: (name: string, enabled: boolean) => Promise<void>
    addForm: (charName: string, formName: string, description: string) => Promise<void>
    updateForm: (charName: string, formId: string, formName: string, description: string) => Promise<void>
    deleteForm: (charName: string, formId: string) => Promise<void>
    uploadFormImage: (charName: string, formId: string, file: File) => Promise<void>
    deleteFormImage: (charName: string, formId: string) => Promise<void>
    toggleFormEnabled: (charName: string, formId: string, enabled: boolean) => Promise<void>
    generateOrtho: (
        charName: string,
        formId: string,
        sourceImages: File[],
    ) => Promise<continuationApi.UploadImageResponse>
    setFormReference: (charName: string, formId: string, imagePath: string) => Promise<void>
}

export function useCharacterManagement(bookId: Ref<string | undefined>, state: ContinuationState): CharacterManagementComposable {
    async function addCharacter(name: string, aliases: string[], description: string) {
        const activeBookId = bookId.value
        if (!activeBookId) return

        await runContinuationMutation({
            state,
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
        if (!activeBookId) return

        await runContinuationMutation({
            state,
            failurePrefix: '删除失败',
            successMessage: '角色删除成功',
            run: () => continuationApi.deleteCharacter(activeBookId, name),
            afterSuccess: () => state.initializeData()
        })
    }

    async function updateCharacterInfo(name: string, newName: string, aliases: string[]) {
        const activeBookId = bookId.value
        if (!activeBookId) return

        await runContinuationMutation({
            state,
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
        if (!activeBookId) return

        const char = state.characters.value.find(c => c.name === name)
        if (!char) return

        const previousEnabled = char.enabled
        char.enabled = enabled

        await runContinuationMutation({
            state,
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
        if (!activeBookId) return

        const char = state.characters.value.find(c => c.name === charName)
        const existingForms = new Set((char?.forms || []).map(form => form.form_id))
        let nextIndex = 1
        while (existingForms.has(`form_${nextIndex}`)) {
            nextIndex += 1
        }
        const formId = `form_${nextIndex}`

        await runContinuationMutation({
            state,
            failurePrefix: '添加失败',
            successMessage: '形态添加成功',
            run: () => continuationApi.addCharacterForm(activeBookId, charName, {
                form_id: formId,
                form_name: formName,
                description
            }),
            afterSuccess: () => state.initializeData()
        })
    }

    async function updateForm(charName: string, formId: string, formName: string, description: string) {
        const activeBookId = bookId.value
        if (!activeBookId) return

        await runContinuationMutation({
            state,
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
        if (!activeBookId) return

        await runContinuationMutation({
            state,
            failurePrefix: '删除失败',
            successMessage: '形态删除成功',
            run: () => continuationApi.deleteCharacterForm(activeBookId, charName, formId),
            afterSuccess: () => state.initializeData()
        })
    }

    async function uploadFormImage(charName: string, formId: string, file: File) {
        const activeBookId = bookId.value
        if (!activeBookId) return

        const formData = new FormData()
        formData.append('file', file)

        await runContinuationMutation({
            state,
            failurePrefix: '上传失败',
            successMessage: '图片上传成功',
            run: () => continuationApi.uploadFormImage(activeBookId, charName, formId, formData),
            afterSuccess: async () => {
                state.imageRefreshKey.value = Date.now()
                await state.initializeData()
            }
        })
    }

    async function deleteFormImage(charName: string, formId: string) {
        const activeBookId = bookId.value
        if (!activeBookId) return

        await runContinuationMutation({
            state,
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
        if (!activeBookId) return

        const char = state.characters.value.find(c => c.name === charName)
        if (!char) return

        const form = char.forms?.find(f => f.form_id === formId)
        if (!form) return

        const previousEnabled = form.enabled
        form.enabled = enabled

        await runContinuationMutation({
            state,
            failurePrefix: '操作失败',
            run: () => continuationApi.toggleFormEnabled(activeBookId, charName, formId, enabled),
            onFailure: () => {
                form.enabled = previousEnabled
            }
        })
    }

    async function generateOrtho(charName: string, formId: string, sourceImages: File[]) {
        if (!bookId.value) return { success: false, error: 'No book ID' }

        try {
            const result = await continuationApi.generateFormOrtho(bookId.value, charName, formId, sourceImages)
            return result
        } catch (error) {
            return {
                success: false,
                error: toContinuationActionError(error)
            }
        }
    }

    async function setFormReference(charName: string, formId: string, imagePath: string) {
        if (!bookId.value) return

        const result = await continuationApi.setFormReference(bookId.value, charName, formId, imagePath)

        if (result.success) {
            state.imageRefreshKey.value = Date.now()
            await state.initializeData()
        } else {
            throw new Error(result.error)
        }
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
