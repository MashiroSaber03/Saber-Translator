<template>
  <div class="continuation-panel">
    <!-- 消息提示 -->
    <div v-if="state.errorMessage.value || state.successMessage.value" class="message" :class="state.errorMessage.value ? 'error' : 'success'">
      {{ state.errorMessage.value || state.successMessage.value }}
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
          :is-generating="scriptGen.isGenerating.value"
          @generate="handleGenerateScript"
        />
        
        <div class="actions">
          <button class="btn secondary" @click="goToStep(0)">← 上一步</button>
          <button class="btn primary" :disabled="!canProceedToPages" @click="goToStep(2)">
            下一步：页面详情 →
          </button>
        </div>
      </div>
      
      <!-- 步骤3: 页面详情 -->
      <div v-show="state.currentStep.value === 2" class="step-panel">
        <PageDetailsPanel
          :pages="state.pages.value"
          :is-generating="state.isGeneratingPages.value"
          :regenerating-page="regeneratingPromptPage"
          @generate-details="handleGeneratePageDetails"
          @regenerate-prompt="handleRegeneratePrompt"
          @regenerate-all-prompts="handleRegenerateAllPrompts"
          @save-changes="handleSavePageChanges"
          @data-change="onPageDataChange"
        />
        
        <div class="actions">
          <button class="btn secondary" @click="goToStep(1)">← 上一步</button>
          <button class="btn primary" :disabled="!canProceedToImages" @click="goToStep(3)">
            下一步：图片生成 →
          </button>
        </div>
      </div>
      
      <!-- 步骤4: 图片生成 -->
      <div v-show="state.currentStep.value === 3" class="step-panel">
        <ImageGenerationPanel
          :pages="state.pages.value"
          :is-generating="imageGen.isGenerating.value"
          :progress="imageGen.generationProgress.value"
          @batch-generate="handleBatchGenerate"
          @regenerate="handleRegenerateImage"
          @use-previous="handleUsePrevious"
          @prompt-change="handlePromptChange"
        />
        
        <div class="actions">
          <button class="btn secondary" @click="goToStep(2)">← 上一步</button>
          <button class="btn primary" :disabled="!canProceedToExport" @click="goToStep(4)">
            下一步：导出 →
          </button>
        </div>
      </div>
      
      <!-- 步骤5: 导出 -->
      <div v-show="state.currentStep.value === 4" class="step-panel">
        <ExportPanel
          v-if="insightStore.currentBookId"
          :book-id="insightStore.currentBookId"
          :generated-count="generatedPagesCount"
          @clear-and-restart="handleClearAndRestart"
        />
        
        <div class="actions">
          <button class="btn secondary" @click="goToStep(3)">← 返回生成</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, provide } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import { 
  useContinuationState,
  ContinuationStateKey
} from '@/composables/continuation/useContinuationState'
import { 
  useCharacterManagement,
  CharacterManagementKey
} from '@/composables/continuation/useCharacterManagement'
import { 
  useScriptGeneration,
  ScriptGenerationKey
} from '@/composables/continuation/useScriptGeneration'
import { 
  useImageGeneration,
  ImageGenerationKey
} from '@/composables/continuation/useImageGeneration'
import CharacterManagementPanel from './continuation/CharacterManagementPanel.vue'
import ScriptGenerationPanel from './continuation/ScriptGenerationPanel.vue'
import PageDetailsPanel from './continuation/PageDetailsPanel.vue'
import ImageGenerationPanel from './continuation/ImageGenerationPanel.vue'
import ExportPanel from './continuation/ExportPanel.vue'
import * as continuationApi from '@/api/continuation'

const insightStore = useInsightStore()

// 创建响应式的bookId
const bookId = computed(() => insightStore.currentBookId || '')

// 提供所有composables的状态，直接传递bookId
const stateComposable = useContinuationState(bookId)
provide(ContinuationStateKey, stateComposable)

const charMgmtComposable = useCharacterManagement(bookId, stateComposable)
provide(CharacterManagementKey, charMgmtComposable)

const scriptGenComposable = useScriptGeneration(bookId, stateComposable)
provide(ScriptGenerationKey, scriptGenComposable)

const imageGenComposable = useImageGeneration(bookId, stateComposable)
provide(ImageGenerationKey, imageGenComposable)

// 使用已创建的composables
const state = stateComposable
const scriptGen = scriptGenComposable
const imageGen = imageGenComposable

const stepNames = ['角色设置', '生成脚本', '页面详情', '图片生成', '导出']

const regeneratingPromptPage = ref<number | null>(null)

// 计算属性
const canProceedToScript = computed(() => {
  return state.characters.value.length > 0
})

const canProceedToPages = computed(() => {
  return state.chapterScript.value !== null
})

const canProceedToImages = computed(() => {
  return state.pages.value.length > 0 && state.pages.value.every(p => p.image_prompt)
})

const canProceedToExport = computed(() => {
  return state.pages.value.some(p => p.image_url && p.status === 'generated')
})

const generatedPagesCount = computed(() => {
  return state.pages.value.filter(p => p.image_url && p.status === 'generated').length
})

// 步骤导航
function canNavigateToStep(step: number): boolean {
  if (step === 0) return true
  if (step === 1) return canProceedToScript.value
  if (step === 2) return canProceedToPages.value
  if (step === 3) return canProceedToImages.value
  if (step === 4) return canProceedToExport.value
  return false
}

function navigateToStep(step: number) {
  if (canNavigateToStep(step)) {
    state.currentStep.value = step
  }
}

function goToStep(step: number) {
  state.currentStep.value = step
}

// 脚本生成
async function handleGenerateScript() {
  if (!insightStore.currentBookId) return
  
  scriptGen.isGenerating.value = true
  state.errorMessage.value = ''
  
  try {
    const result = await continuationApi.generateScript(
      insightStore.currentBookId,
      state.continuationDirection.value,
      state.pageCount.value
    )
    
    if (result.success && result.script) {
      state.chapterScript.value = result.script
      
      // 保存配置
      try {
        await continuationApi.saveConfig(insightStore.currentBookId, {
          page_count: state.pageCount.value,
          style_reference_pages: state.styleRefPages.value,
          continuation_direction: state.continuationDirection.value
        })
      } catch (error) {
        console.error('保存配置失败:', error)
      }
      
      state.showMessage('脚本生成成功', 'success')
    } else {
      state.showMessage('生成失败: ' + result.error, 'error')
    }
  } catch (error) {
    state.showMessage('生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    scriptGen.isGenerating.value = false
  }
}

// 页面详情生成
async function handleGeneratePageDetails() {
  if (!insightStore.currentBookId || !state.chapterScript.value) return
  
  state.isGeneratingPages.value = true
  state.isGeneratingPrompts.value = true
  state.errorMessage.value = ''
  
  const totalPages = state.chapterScript.value.page_count || state.pageCount.value
  state.pages.value = []
  
  try {
    for (let i = 1; i <= totalPages; i++) {
      state.showMessage(`正在生成第 ${i}/${totalPages} 页详情...`, 'info')
      
      const detailResult = await continuationApi.generateSinglePageDetails(
        insightStore.currentBookId,
        state.chapterScript.value,
        i
      )
      
      if (!detailResult.success || !detailResult.page) {
        state.pages.value.push({
          page_number: i,
          characters: [],
          description: `生成失败: ${detailResult.error || '未知错误'}`,
          dialogues: [],
          image_prompt: '',
          image_url: '',
          previous_url: '',
          status: 'failed' as const
        })
        continue
      }
      
      state.showMessage(`正在生成第 ${i}/${totalPages} 页提示词...`, 'info')
      
      const promptResult = await continuationApi.generateSingleImagePrompt(
        insightStore.currentBookId,
        detailResult.page,
        i
      )
      
      if (promptResult.success && promptResult.page) {
        state.pages.value.push(promptResult.page)
      } else {
        const pageWithError = { ...detailResult.page }
        pageWithError.image_prompt = `提示词生成失败: ${promptResult.error || '未知错误'}`
        state.pages.value.push(pageWithError)
      }
    }
    
    await continuationApi.savePages(insightStore.currentBookId, state.pages.value)
    state.showMessage(`页面详情和提示词生成完成 (${state.pages.value.length} 页)`, 'success')
  } catch (error) {
    state.showMessage('生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    state.isGeneratingPages.value = false
    state.isGeneratingPrompts.value = false
  }
}

async function handleRegeneratePrompt(pageNumber: number) {
  if (!insightStore.currentBookId) return
  
  const page = state.pages.value.find(p => p.page_number === pageNumber)
  if (!page) return
  
  regeneratingPromptPage.value = pageNumber
  
  try {
    const result = await continuationApi.generateSingleImagePrompt(
      insightStore.currentBookId,
      page,
      pageNumber
    )
    
    if (result.success && result.page) {
      page.image_prompt = result.page.image_prompt
      await continuationApi.savePages(insightStore.currentBookId, state.pages.value)
      state.showMessage(`第 ${pageNumber} 页提示词已更新`, 'success')
    } else {
      state.showMessage('生成失败: ' + result.error, 'error')
    }
  } catch (error) {
    state.showMessage('生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    regeneratingPromptPage.value = null
  }
}

// 批量重新生成所有提示词
async function handleRegenerateAllPrompts() {
  if (!insightStore.currentBookId || state.pages.value.length === 0) return
  
  if (!confirm('确定要重新生成所有页面的提示词吗？这将覆盖现有的提示词。')) {
    return
  }
  
  state.isGeneratingPages.value = true
  state.errorMessage.value = ''
  
  try {
    // 调用批量生成提示词API
    const result = await continuationApi.generateImagePrompts(
      insightStore.currentBookId,
      state.pages.value
    )
    
    if (result.success && result.pages) {
      // 更新提示词
      state.pages.value = result.pages
      await continuationApi.savePages(insightStore.currentBookId, state.pages.value)
      state.showMessage('所有提示词已重新生成', 'success')
    } else {
      state.showMessage('生成失败: ' + result.error, 'error')
    }
  } catch (error) {
    state.showMessage('生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    state.isGeneratingPages.value = false
  }
}

async function handleSavePageChanges() {
  if (!insightStore.currentBookId || state.pages.value.length === 0) return
  
  try {
    await continuationApi.savePages(insightStore.currentBookId, state.pages.value)
    state.showMessage('页面数据保存成功', 'success')
  } catch (error) {
    state.showMessage('保存失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  }
}

function onPageDataChange() {
  console.log('页面数据已修改')
}

// 图片生成
async function handleBatchGenerate() {
  if (!insightStore.currentBookId) return
  await imageGen.batchGenerateImages(state.pages.value)
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
    await continuationApi.savePages(insightStore.currentBookId, state.pages.value)
  }
}

// 提示词变更处理
async function handlePromptChange(pageNumber: number, prompt: string) {
  // 直接更新state中的数据（v-model已经做了，这里只需要标记需要保存）
  console.log(`页面 ${pageNumber} 提示词已修改`)
  // 可以在这里添加防抖保存逻辑，或者让用户手动保存
}

// 清空并重新开始
async function handleClearAndRestart() {
  if (!insightStore.currentBookId) return
  
  try {
    await continuationApi.clearContinuationData(insightStore.currentBookId)
    await state.resetState()
    state.currentStep.value = 0
    state.showMessage('数据已清空', 'success')
  } catch (error) {
    state.showMessage('清空失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  }
}

// 生命周期
onMounted(() => {
  if (insightStore.currentBookId) {
    state.initializeData()
  }
})

watch(() => insightStore.currentBookId, (newBookId) => {
  if (newBookId) {
    state.initializeData()
  }
})
</script>

<style scoped>
.continuation-panel {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
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
  background: rgba(255,255,255,0.2);
  font-weight: bold;
  font-size: 13px;
}

.step:not(.active):not(.completed) .step-number {
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
  margin: 0 0 20px 0;
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
