/**
 * 角色管理Composable
 * 处理角色CRUD、形态管理、图片上传等
 */

import { inject, provide, type Ref, type InjectionKey } from 'vue'
import * as continuationApi from '@/api/continuation'
import { useContinuationStateInject, type ContinuationState } from './useContinuationState'

interface CharacterManagementComposable {
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
    generateOrtho: (charName: string, formId: string, sourceImages: File[]) => Promise<{ success: boolean; image_path?: string; error?: string }>
    setFormReference: (charName: string, formId: string, imagePath: string) => Promise<void>
}

const CharacterManagementKey: InjectionKey<CharacterManagementComposable> = Symbol('CharacterManagement')

export { CharacterManagementKey }

export function useCharacterManagement(bookId: Ref<string | undefined>, state: ContinuationState): CharacterManagementComposable {
    async function addCharacter(name: string, aliases: string[], description: string) {
        if (!bookId.value) return

        try {
            const result = await continuationApi.addCharacter(bookId.value, {
                name,
                aliases,
                description
            })

            if (result.success) {
                await state.initializeData()
                state.showMessage('角色添加成功', 'success')
            } else {
                state.showMessage('添加失败: ' + result.error, 'error')
            }
        } catch (error) {
            state.showMessage('添加失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
        }
    }

    async function deleteCharacter(name: string) {
        if (!bookId.value) return

        try {
            const result = await continuationApi.deleteCharacter(bookId.value, name)

            if (result.success) {
                await state.initializeData()
                state.showMessage('角色删除成功', 'success')
            } else {
                state.showMessage('删除失败: ' + result.error, 'error')
            }
        } catch (error) {
            state.showMessage('删除失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
        }
    }

    async function updateCharacterInfo(name: string, newName: string, aliases: string[]) {
        if (!bookId.value) return

        try {
            const result = await continuationApi.updateCharacterInfo(bookId.value, name, {
                name: newName,
                aliases
            })

            if (result.success) {
                await state.initializeData()
                state.showMessage('角色信息更新成功', 'success')
            } else {
                state.showMessage('更新失败: ' + result.error, 'error')
            }
        } catch (error) {
            state.showMessage('更新失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
        }
    }

    async function toggleCharacterEnabled(name: string, enabled: boolean) {
        if (!bookId.value) return

        // 乐观更新：先立即更新本地状态
        const char = state.characters.value.find(c => c.name === name)
        if (!char) return
        
        const previousEnabled = char.enabled
        char.enabled = enabled

        try {
            // 异步保存到后端
            const result = await continuationApi.updateCharacterInfo(bookId.value, name, {
                name: char.name,
                aliases: char.aliases || [],
                enabled
            })

            if (!result.success) {
                // 如果失败，回滚状态
                char.enabled = previousEnabled
                state.showMessage('操作失败: ' + result.error, 'error')
            }
        } catch (error) {
            // 如果出错，回滚状态
            char.enabled = previousEnabled
            state.showMessage('操作失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
        }
    }

    async function addForm(charName: string, formName: string, description: string) {
        if (!bookId.value) return

        try {
            // 自动生成formId - 基于现有形态数量
            const char = state.characters.value.find(c => c.name === charName)
            const existingFormsCount = char?.forms?.length || 0
            const formId = `form_${existingFormsCount + 1}`

            const result = await continuationApi.addCharacterForm(bookId.value, charName, {
                form_id: formId,
                form_name: formName,
                description
            })

            if (result.success) {
                await state.initializeData()
                state.showMessage('形态添加成功', 'success')
            } else {
                state.showMessage('添加失败: ' + result.error, 'error')
            }
        } catch (error) {
            state.showMessage('添加失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
        }
    }

    async function updateForm(charName: string, formId: string, formName: string, description: string) {
        if (!bookId.value) return

        try {
            const result = await continuationApi.updateCharacterForm(bookId.value, charName, formId, {
                form_name: formName,
                description
            })

            if (result.success) {
                await state.initializeData()
                state.showMessage('形态更新成功', 'success')
            } else {
                state.showMessage('更新失败: ' + result.error, 'error')
            }
        } catch (error) {
            state.showMessage('更新失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
        }
    }

    async function deleteForm(charName: string, formId: string) {
        if (!bookId.value) return

        try {
            const result = await continuationApi.deleteCharacterForm(bookId.value, charName, formId)

            if (result.success) {
                await state.initializeData()
                state.showMessage('形态删除成功', 'success')
            } else {
                state.showMessage('删除失败: ' + result.error, 'error')
            }
        } catch (error) {
            state.showMessage('删除失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
        }
    }

    async function uploadFormImage(charName: string, formId: string, file: File) {
        if (!bookId.value) return

        try {
            const formData = new FormData()
            formData.append('file', file)

            const result = await continuationApi.uploadFormImage(bookId.value, charName, formId, formData)

            if (result.success) {
                state.imageRefreshKey.value = Date.now()
                await state.initializeData()
                state.showMessage('图片上传成功', 'success')
            } else {
                state.showMessage('上传失败: ' + result.error, 'error')
            }
        } catch (error) {
            state.showMessage('上传失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
        }
    }

    async function deleteFormImage(charName: string, formId: string) {
        if (!bookId.value) return

        try {
            const result = await continuationApi.deleteFormImage(bookId.value, charName, formId)

            if (result.success) {
                state.imageRefreshKey.value = Date.now()
                await state.initializeData()
                state.showMessage('图片删除成功', 'success')
            } else {
                state.showMessage('删除失败: ' + result.error, 'error')
            }
        } catch (error) {
            state.showMessage('删除失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
        }
    }

    async function toggleFormEnabled(charName: string, formId: string, enabled: boolean) {
        if (!bookId.value) return

        // 乐观更新：先立即更新本地状态
        const char = state.characters.value.find(c => c.name === charName)
        if (!char) return
        
        const form = char.forms?.find(f => f.form_id === formId)
        if (!form) return
        
        const previousEnabled = form.enabled
        form.enabled = enabled

        try {
            // 异步保存到后端
            const result = await continuationApi.toggleFormEnabled(bookId.value, charName, formId, enabled)

            if (!result.success) {
                // 如果失败，回滚状态
                form.enabled = previousEnabled
                state.showMessage('操作失败: ' + result.error, 'error')
            }
        } catch (error) {
            // 如果出错，回滚状态
            form.enabled = previousEnabled
            state.showMessage('操作失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
        }
    }

    async function generateOrtho(charName: string, formId: string, sourceImages: File[]) {
        if (!bookId.value) return { success: false, error: 'No book ID' }

        try {
            const result = await continuationApi.generateFormOrtho(bookId.value, charName, formId, sourceImages)
            return result
        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : '网络错误'
            }
        }
    }

    async function setFormReference(charName: string, formId: string, imagePath: string) {
        if (!bookId.value) return

        try {
            const result = await continuationApi.setFormReference(bookId.value, charName, formId, imagePath)

            if (result.success) {
                state.imageRefreshKey.value = Date.now()
                await state.initializeData()
            } else {
                throw new Error(result.error)
            }
        } catch (error) {
            throw error
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

export function provideCharacterManagement() {
    const bookId = inject<Ref<string>>('bookId')
    const state = useContinuationStateInject()

    if (!bookId) {
        throw new Error('provideCharacterManagement must be used after bookId is provided')
    }

    const composable = useCharacterManagement(bookId, state)
    provide(CharacterManagementKey, composable)

    return composable
}

export function useCharacterManagementInject(): CharacterManagementComposable {
    const composable = inject(CharacterManagementKey)
    if (!composable) {
        throw new Error('useCharacterManagementInject must be used after provideCharacterManagement')
    }
    return composable
}
