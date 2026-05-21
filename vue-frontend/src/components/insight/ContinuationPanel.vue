<template>
  <div class="continuation-panel">
    <!-- 消息提示 -->
    <div v-if="state.errorMessage.value || state.successMessage.value" class="message" :class="state.messageType.value || (state.errorMessage.value ? 'error' : 'success')">
      {{ state.errorMessage.value || state.successMessage.value }}
    </div>

    <div class="analysis-sync-bar">
      <div class="analysis-sync-meta">
        <span class="analysis-sync-title">分析数据同步</span>
        <span class="analysis-sync-status">{{ analysisSyncStatus }}</span>
      </div>
      <button
        class="btn secondary small analysis-sync-button"
        :disabled="state.isSyncingAnalysis.value || !insightStore.currentBookId"
        @click="handleManualSync"
      >
        {{ state.isSyncingAnalysis.value ? '同步中...' : '🔄 同步分析数据' }}
      </button>
    </div>
    
    <!-- 步骤指示器 -->
    <div class="step-indicator">
      <div 
        v-for="(name, index) in stepNames" 
        :key="index"
        class="step"
        :class="{
          active: state.currentStep.value === index,
          completed: state.currentStep.value > index,
          clickable: canNavigateToStep(index)
        }"
        @click="navigateToStep(index)"
      >
        <span class="step-number">{{ index + 1 }}</span>
        <span class="step-name">{{ name }}</span>
      </div>
    </div>
    
    <!-- 步骤内容 -->
    <div class="step-content">
      <!-- 步骤1: 角色管理 + 设置 -->
      <div v-show="state.currentStep.value === 0" class="step-panel">
        <h3>📝 续写设置</h3>
        
        <div class="form-group">
          <label>续写页数</label>
          <input v-model.number="state.pageCount.value" type="number" min="5" max="50">
          <p class="hint">建议 10-20 页</p>
        </div>
        
        <div class="form-group">
          <label>画风参考页数</label>
          <input v-model.number="state.styleRefPages.value" type="number" min="1" max="10">
          <p class="hint">用于维持画风一致性</p>
        </div>
        
        <div class="form-group">
          <label>续写方向（可选）</label>
          <textarea 
            v-model="state.continuationDirection.value" 
            rows="4" 
            placeholder="例如：延续主线剧情，探索新的冒险..."
          ></textarea>
          <p class="hint">留空将自动根据剧情发展生成</p>
        </div>
        
        <CharacterManagementPanel 
          v-if="insightStore.currentBookId"
          :book-id="insightStore.currentBookId"
          :is-loading="state.isLoading.value"
        />
        
        <div class="actions">
          <button class="btn secondary danger" @click="handleClearAndRestart">🗑️ 清除数据重新开始</button>
          <button class="btn primary" :disabled="!canProceedToScript" @click="goToStep(1)">
            下一步：生成脚本 →
          </button>
        </div>
      </div>
      
      <!-- 步骤2: 脚本生成 -->
      <div v-show="state.currentStep.value === 1" class="step-panel">
        <ScriptGenerationPanel
          :script="state.chapterScript.value"
          :is-generating="isGeneratingScript"
          :is-saving="isSavingScript"
          :book-id="insightStore.currentBookId || ''"
          @generate="handleGenerateScript"
          @update-script="handleScriptUpdate"
          @save-script="handleSaveScript"
          @reset-script="handleResetScript"
        />
        
        <div class="actions">
          <button class="btn secondary" @click="goToStep(0)">← 上一步</button>
          <button class="btn primary" :disabled="!canProceedToPages" @click="goToStep(2)">
            下一步：页面剧情 →
          </button>
        </div>
      </div>
      
      <!-- 步骤3: 页面剧情 -->
      <div v-show="state.currentStep.value === 2" class="step-panel">
        <PageDetailsPanel
          :pages="state.pages.value"
          :is-generating="state.isGeneratingPages.value"
          @generate-details="handleGeneratePageDetails"
          @save-changes="handleSavePageChanges"
          @story-change="handleStoryContentChange"
        />
        
        <div class="actions">
          <button class="btn secondary" @click="goToStep(1)">← 上一步</button>
          <button class="btn primary" :disabled="!canProceedToImages" @click="goToStep(3)">
            下一步：图片生成 →
          </button>
        </div>
      </div>
      
      <!-- 步骤4: 图片生成 / 导出 -->
      <div v-show="state.currentStep.value === 3" class="step-panel">
        <ImageGenerationPanel
          :pages="state.pages.value"
          :is-generating="imageGen.isGenerating.value"
          :progress="imageGen.generationProgress.value"
          :book-id="insightStore.currentBookId || ''"
          @batch-generate="handleBatchGenerate"
          @regenerate="handleRegenerateImage"
          @use-previous="handleUsePrevious"
          @prompt-change="handlePromptChange"
        />
        
        <ExportPanel
          v-if="insightStore.currentBookId"
          :book-id="insightStore.currentBookId"
          :generated-count="generatedPagesCount"
          @clear-and-restart="handleClearAndRestart"
        />

        <div class="actions">
          <button class="btn secondary" @click="goToStep(2)">← 上一步</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, provide, onBeforeUnmount } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import { 
  useContinuationState,
  ContinuationStateKey
} from '@/composables/continuation/useContinuationState'
import { 
  useCharacterManagement,
  CharacterManagementKey
} from '@/composables/continuation/useCharacterManagement'
import { useImageGeneration } from '@/composables/continuation/useImageGeneration'
import CharacterManagementPanel from './continuation/CharacterManagementPanel.vue'
import ScriptGenerationPanel from './continuation/ScriptGenerationPanel.vue'
import PageDetailsPanel from './continuation/PageDetailsPanel.vue'
import ImageGenerationPanel from './continuation/ImageGenerationPanel.vue'
import ExportPanel from './continuation/ExportPanel.vue'
import * as continuationApi from '@/api/continuation'
import { hasUsableStoryContent } from '@/composables/continuation/promptValidation'

const insightStore = useInsightStore()

// 创建响应式的bookId
const bookId = computed(() => insightStore.currentBookId || '')

// 提供所有composables的状态，直接传递bookId
const stateComposable = useContinuationState(bookId)
provide(ContinuationStateKey, stateComposable)

const charMgmtComposable = useCharacterManagement(bookId, stateComposable)
provide(CharacterManagementKey, charMgmtComposable)

const imageGenComposable = useImageGeneration(bookId, stateComposable)

// 使用已创建的composables
const state = stateComposable
const imageGen = imageGenComposable

const stepNames = ['角色设置', '生成脚本', '页面剧情', '图片生成/导出']

const isGeneratingScript = ref(false)
const isSavingScript = ref(false)
const scriptDirty = ref(false)
const lastSavedScriptText = ref('')
let promptSaveTimer: ReturnType<typeof setTimeout> | null = null
let storySaveTimer: ReturnType<typeof setTimeout> | null = null

function resetLocalWorkflowState() {
  isGeneratingScript.value = false
  isSavingScript.value = false
  if (storySaveTimer) {
    clearTimeout(storySaveTimer)
    storySaveTimer = null
  }
}

// 计算属性
const canProceedToScript = computed(() => {
  return state.isDataReady.value && state.characters.value.length > 0
})

const canProceedToPages = computed(() => {
  return state.chapterScript.value !== null
})

const canProceedToImages = computed(() => {
  return state.pages.value.length > 0 && state.pages.value.every(
    p => p.status !== 'failed' && hasUsableStoryContent(p)
  )
})

const generatedPagesCount = computed(() => {
  return state.pages.value.filter(p => p.image_url && p.status === 'generated').length
})

const analysisSyncStatus = computed(() => {
  if (state.isSyncingAnalysis.value) {
    return '正在同步最新分析数据...'
  }

  if (!state.isDataReady.value) {
    return state.errorMessage.value || '缺少故事概要或时间线，暂不可续写'
  }

  if (state.lastAnalysisSyncAt.value) {
    const syncDate = new Date(state.lastAnalysisSyncAt.value)
    if (!Number.isNaN(syncDate.getTime())) {
      return `已同步 ${syncDate.toLocaleString()}`
    }
  }

  return '分析数据已就绪'
})

async function persistContinuationConfig(): Promise<{ success: boolean; error?: string }> {
  if (!insightStore.currentBookId) {
    return { success: false, error: '当前未选择漫画' }
  }

  try {
    const result = await continuationApi.saveConfig(insightStore.currentBookId, {
      page_count: state.pageCount.value,
      style_reference_pages: state.styleRefPages.value,
      continuation_direction: state.continuationDirection.value
    })

    if (!result.success) {
      return { success: false, error: result.error || '未知错误' }
    }

    return { success: true }
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : '网络错误'
    }
  }
}

// 步骤导航
function canNavigateToStep(step: number): boolean {
  if (step === 0) return true
  if (step === 1) return canProceedToScript.value
  if (step === 2) return canProceedToPages.value
  if (step === 3) return canProceedToImages.value
  return false
}

function navigateToStep(step: number) {
  if (canNavigateToStep(step)) {
    void goToStep(step)
  }
}

function resolveReachableStep(requestedStep: number): number {
  if (requestedStep <= 0) return 0
  if (!canProceedToScript.value) return 0
  if (requestedStep === 1) return 1
  if (!canProceedToPages.value) return 1
  if (requestedStep === 2) return 2
  return 3
}

// 脚本生成
async function handleGenerateScript(payload: { referenceTokens: string[] | null; referenceImageCount: number }) {
  if (!insightStore.currentBookId) return

  isGeneratingScript.value = true
  state.errorMessage.value = ''

  try {
    // 使用支持自定义参考图的API
    const result = await continuationApi.generateScriptWithRefs(
      insightStore.currentBookId,
      state.continuationDirection.value,
      state.pageCount.value,
      payload.referenceTokens || undefined,
      payload.referenceImageCount
    )
    
    if (result.success && result.script) {
      const hadExistingPages = state.pages.value.length > 0
      state.chapterScript.value = result.script
      lastSavedScriptText.value = result.script.script_text
      scriptDirty.value = false
      if (hadExistingPages) {
        state.pages.value = []
        await persistPages([])
      }

      const baseMessage = hadExistingPages ? '脚本生成成功，旧的页面剧情已清空' : '脚本生成成功'
      const configResult = await persistContinuationConfig()

      if (configResult.success) {
        state.showMessage(baseMessage, 'success')
      } else {
        state.showMessage(`${baseMessage}，但续写配置保存失败：${configResult.error}`, 'info')
      }
    } else {
      state.showMessage('生成失败: ' + result.error, 'error')
    }
  } catch (error) {
    state.showMessage('生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isGeneratingScript.value = false
  }
}

async function handleSaveScript(showSuccessMessage = true): Promise<boolean> {
  if (!insightStore.currentBookId || !state.chapterScript.value) return false

  isSavingScript.value = true
  const shouldInvalidatePages = scriptDirty.value && state.pages.value.length > 0
  try {
    const result = await continuationApi.saveScript(
      insightStore.currentBookId,
      state.chapterScript.value
    )

    if (result.success) {
      lastSavedScriptText.value = state.chapterScript.value.script_text
      scriptDirty.value = false
      if (shouldInvalidatePages) {
        state.pages.value = []
        await persistPages([])
        state.showMessage('脚本已更新，旧的页面剧情已清空，请重新生成。', 'info')
        return true
      }
      if (showSuccessMessage) {
        state.showMessage('脚本已保存', 'success')
      }
      return true
    }

    state.showMessage('脚本保存失败: ' + result.error, 'error')
    return false
  } catch (error) {
    state.showMessage('脚本保存失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
    return false
  } finally {
    isSavingScript.value = false
  }
}

async function persistPages(pages = state.pages.value): Promise<void> {
  if (!insightStore.currentBookId) return
  await continuationApi.savePages(insightStore.currentBookId, pages)
}

function handleScriptUpdate(scriptText: string) {
  if (!state.chapterScript.value) return
  state.chapterScript.value.script_text = scriptText
  scriptDirty.value = scriptText !== lastSavedScriptText.value
}

function handleStoryContentChange() {
  if (storySaveTimer) {
    clearTimeout(storySaveTimer)
  }

  storySaveTimer = setTimeout(async () => {
    if (!insightStore.currentBookId || state.pages.value.length === 0) return

    try {
      await persistPages()
    } catch (error) {
      state.showMessage('页面剧情保存失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
    }
  }, 600)
}

function handleResetScript() {
  if (!state.chapterScript.value) return
  state.chapterScript.value.script_text = lastSavedScriptText.value
  scriptDirty.value = false
}

// 页面剧情生成
async function handleGeneratePageDetails() {
  if (!insightStore.currentBookId || !state.chapterScript.value) return

  if (scriptDirty.value) {
    const saved = await handleSaveScript(false)
    if (!saved) {
      return
    }
  }
  
  state.isGeneratingPages.value = true
  state.errorMessage.value = ''
  
  const totalPages = state.chapterScript.value.page_count || state.pageCount.value
  const workingPages = Array.from({ length: totalPages }, (_, index) => {
    const pageNumber = index + 1
    const existing = state.pages.value.find(page => page.page_number === pageNumber)
    return existing
      ? { ...existing }
      : {
          page_number: pageNumber,
          continuity_text: '',
          story_text: '',
          dialogue_text: '',
          characters: [],
          character_forms: [],
          final_prompt: '',
          image_url: '',
          previous_url: '',
          status: 'pending' as const,
        }
  })
  state.pages.value = [...workingPages]
  
  try {
    for (let i = 1; i <= totalPages; i++) {
      const existingPage = workingPages[i - 1]!
      const alreadyReady = existingPage.status !== 'failed'
        && hasUsableStoryContent(existingPage)

      if (alreadyReady) {
        continue
      }

      state.showMessage(`正在生成第 ${i}/${totalPages} 页剧情...`, 'info')
      
      const detailResult = await continuationApi.generateSinglePageDetails(
        insightStore.currentBookId,
        state.chapterScript.value,
        i
      )
      
      if (!detailResult.success || !detailResult.page) {
        workingPages[i - 1] = {
          page_number: i,
          continuity_text: '',
          story_text: '',
          dialogue_text: '',
          characters: [],
          character_forms: [],
          final_prompt: '',
          image_url: '',
          previous_url: '',
          status: 'failed' as const
        }
        state.pages.value = [...workingPages]
        await persistPages(workingPages)
        continue
      }

      workingPages[i - 1] = {
        ...detailResult.page,
        status: detailResult.page.status || 'pending',
      }

      state.pages.value = [...workingPages]
      await persistPages(workingPages)
    }
    
    state.showMessage(`页面剧情生成完成 (${workingPages.length} 页)`, 'success')
  } catch (error) {
    state.showMessage('生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    state.isGeneratingPages.value = false
  }
}

async function handleSavePageChanges() {
  if (!insightStore.currentBookId || state.pages.value.length === 0) return
  
  try {
    await persistPages()
    state.showMessage('页面数据保存成功', 'success')
  } catch (error) {
    state.showMessage('保存失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  }
}

// 图片生成
async function handleBatchGenerate(initialStyleReferenceTokens: string[] | null) {
  if (!insightStore.currentBookId) return
  await imageGen.batchGenerateImages(state.pages.value, initialStyleReferenceTokens || undefined)
}

async function handleRegenerateImage(pageNumber: number) {
  if (!insightStore.currentBookId) return
  await imageGen.regeneratePageImage(pageNumber)
}

async function handleUsePrevious(pageNumber: number) {
  const page = state.pages.value.find(p => p.page_number === pageNumber)
  if (!page || !page.previous_url) return
  
  const temp = page.image_url
  page.image_url = page.previous_url
  page.previous_url = temp
  
  if (insightStore.currentBookId) {
    await persistPages()
  }
}

// 提示词变更处理
async function handlePromptChange(_pageNumber: number) {
  if (promptSaveTimer) {
    clearTimeout(promptSaveTimer)
  }

  promptSaveTimer = setTimeout(async () => {
    if (!insightStore.currentBookId || state.pages.value.length === 0) return

    try {
      await persistPages()
    } catch (error) {
      state.showMessage('提示词保存失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
    }
  }, 600)
}

async function handleManualSync() {
  await state.syncAnalysisData('manual')
}

// 清空并重新开始
async function handleClearAndRestart() {
  if (!insightStore.currentBookId) return
  
  try {
    if (promptSaveTimer) {
      clearTimeout(promptSaveTimer)
      promptSaveTimer = null
    }
    await continuationApi.clearContinuationData(insightStore.currentBookId)
    state.resetState()
    resetLocalWorkflowState()
    await state.initializeData()
    if (state.isDataReady.value) {
      state.showMessage('续写数据已清空，可重新开始。', 'success')
    }
  } catch (error) {
    state.showMessage('清空失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  }
}

async function goToStep(step: number) {
  if (state.currentStep.value === 0 && step !== 0) {
    const configResult = await persistContinuationConfig()
    if (!configResult.success) {
      state.showMessage(`续写配置保存失败：${configResult.error}`, 'info')
    }
  }

  if (state.currentStep.value === 1 && step !== 1 && scriptDirty.value) {
    const saved = await handleSaveScript(false)
    if (!saved) {
      return
    }
  }

  state.currentStep.value = resolveReachableStep(step)
}

// 生命周期
watch(() => insightStore.currentBookId, (newBookId) => {
  if (promptSaveTimer) {
    clearTimeout(promptSaveTimer)
    promptSaveTimer = null
  }
  resetLocalWorkflowState()

  if (newBookId) {
    state.initializeData()
  } else {
    state.resetState()
  }
}, { immediate: true })

watch(() => insightStore.dataRefreshKey, async (newKey, oldKey) => {
  if (!insightStore.currentBookId || newKey <= 0 || newKey === oldKey) return
  await state.syncAnalysisData('auto')
})

watch(() => state.chapterScript.value, (script) => {
  if (script) {
    lastSavedScriptText.value = script.script_text
    scriptDirty.value = false
  } else {
    lastSavedScriptText.value = ''
    scriptDirty.value = false
  }
}, { immediate: true })

onBeforeUnmount(() => {
  if (promptSaveTimer) {
    clearTimeout(promptSaveTimer)
    promptSaveTimer = null
  }
})
</script>

<style scoped>
.continuation-panel {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.analysis-sync-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 16px;
  margin-bottom: 16px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 12px;
  background: var(--bg-secondary, #f7f7f7);
}

.analysis-sync-meta {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.analysis-sync-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary, #333);
}

.analysis-sync-status {
  font-size: 12px;
  color: var(--text-secondary, #666);
}

/* 消息提示 */
.message {
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 16px;
  font-size: 14px;
}

.message.error {
  background: #fef2f2;
  color: #dc2626;
  border: 1px solid #fecaca;
}

.message.success {
  background: #f0fdf4;
  color: #16a34a;
  border: 1px solid #bbf7d0;
}

.message.info {
  background: #eff6ff;
  color: #2563eb;
  border: 1px solid #bfdbfe;
}

/* 步骤指示器 */
.step-indicator {
  display: flex;
  justify-content: center;
  gap: 8px;
  margin-bottom: 24px;
  padding: 16px;
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 12px;
}

.step {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 20px;
  background: var(--bg-primary, #fff);
  border: 2px solid var(--border-color, #e0e0e0);
  transition: all 0.3s;
}

.step.clickable {
  cursor: pointer;
}

.step.clickable:hover {
  border-color: var(--primary, #6366f1);
}

.step.active {
  background: var(--primary, #6366f1);
  border-color: var(--primary, #6366f1);
  color: white;
}

.step.completed {
  background: #22c55e;
  border-color: #22c55e;
  color: white;
}

.step-number {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: rgb(255,255,255,0.2);
  font-weight: bold;
  font-size: 13px;
}

.step:not(.active, .completed) .step-number {
  background: var(--bg-secondary, #f5f5f5);
}

.step-name {
  font-size: 14px;
  font-weight: 500;
}

/* 步骤内容 */
.step-content {
  background: var(--bg-primary, #fff);
  border-radius: 12px;
  border: 1px solid var(--border-color, #e0e0e0);
}

.step-panel {
  padding: 24px;
}

.step-panel h3 {
  margin: 0 0 20px;
  font-size: 18px;
  font-weight: 600;
}

/* 表单样式 */
.form-group {
  margin-bottom: 16px;
}

.form-group label {
  display: block;
  margin-bottom: 6px;
  font-weight: 500;
  font-size: 14px;
}

.form-group input,
.form-group textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 8px;
  font-size: 14px;
  font-family: inherit;
}

.form-group input:focus,
.form-group textarea:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
}

.hint {
  margin-top: 4px;
  font-size: 12px;
  color: var(--text-secondary, #666);
}

/* 按钮 */
.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn.primary {
  background: var(--primary, #6366f1);
  color: white;
}

.btn.primary:hover:not(:disabled) {
  background: var(--primary-dark, #4f46e5);
}

.btn.secondary {
  background: var(--bg-secondary, #f3f4f6);
  color: var(--text-primary, #333);
  border: 1px solid var(--border-color, #e0e0e0);
}

.btn.secondary:hover:not(:disabled) {
  background: var(--bg-hover, #e5e7eb);
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn.danger {
  background: #fee2e2;
  color: #dc2626;
  border: 1px solid #fecaca;
}

.btn.danger:hover:not(:disabled) {
  background: #fecaca;
  border-color: #fca5a5;
}

.actions {
  display: flex;
  justify-content: space-between;
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid var(--border-color, #e0e0e0);
}
</style>
