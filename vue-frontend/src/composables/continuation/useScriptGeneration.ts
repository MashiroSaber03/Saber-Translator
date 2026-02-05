/**
 * 脚本生成 Composable
 * 处理续写脚本的生成和编辑
 */

import { ref, type Ref, inject, provide, type InjectionKey } from 'vue'
import * as continuationApi from '@/api/continuation'
import { useContinuationStateInject, type ContinuationState } from './useContinuationState'

interface ScriptGenerationComposable {
    isGenerating: Ref<boolean>
    generateScript: () => Promise<void>
}

const ScriptGenerationKey: InjectionKey<ScriptGenerationComposable> = Symbol('ScriptGenerationKey')

export { ScriptGenerationKey }

export function provideScriptGeneration() {
    const bookId = inject<Ref<string>>('bookId')
    const state = useContinuationStateInject()

    if (!bookId) {
        throw new Error('provideScriptGeneration must be used after bookId is provided')
    }

    const composable = useScriptGeneration(bookId, state)
    provide(ScriptGenerationKey, composable)

    return composable
}

export function useScriptGenerationInject() {
    const composable = inject(ScriptGenerationKey)
    if (!composable) {
        throw new Error('useScriptGenerationInject must be used after provideScriptGeneration')
    }
    return composable
}

export function useScriptGeneration(bookId: Ref<string>, state: ContinuationState): ScriptGenerationComposable {
    const isGenerating = ref(false)

    /**
     * 生成脚本
     */
    async function generateScript(): Promise<void> {
        if (!bookId.value) return

        isGenerating.value = true
        state.errorMessage.value = ''

        try {
            const result = await continuationApi.generateScript(
                bookId.value,
                state.continuationDirection.value,
                state.pageCount.value
            )

            if (result.success && result.script) {
                state.chapterScript.value = result.script

                // 保存配置（脚本后端已自动保存）
                try {
                    await continuationApi.saveConfig(bookId.value, {
                        page_count: state.pageCount.value,
                        style_reference_pages: state.styleRefPages.value,
                        continuation_direction: state.continuationDirection.value
                    })
                    console.log('配置已保存')
                } catch (saveError) {
                    console.error('保存配置失败:', saveError)
                }

                state.showMessage('脚本生成成功', 'success')
            } else {
                state.showMessage('脚本生成失败: ' + result.error, 'error')
            }
        } catch (error) {
            state.showMessage('脚本生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
        } finally {
            isGenerating.value = false
        }
    }

    return {
        isGenerating,
        generateScript
    }
}
