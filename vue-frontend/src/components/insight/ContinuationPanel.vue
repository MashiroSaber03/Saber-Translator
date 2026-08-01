<template>
  <ProductWorkspacePanel variant="wizard" aria-label="续写工作区">
    <div class="continuation-panel">
      <ProductStatusBanner
        v-if="workflowMessage"
        class="continuation-panel__message"
        :tone="workflowMessageTone"
        aria-live="polite"
      >
        {{ workflowMessage }}
      </ProductStatusBanner>
      <div class="continuation-panel__sync-bar">
        <div class="continuation-panel__sync-meta">
          <span class="continuation-panel__sync-title">分析数据同步</span>
          <span class="continuation-panel__sync-status">{{ analysisSyncStatus }}</span>
        </div>
        <UiButton
          variant="secondary"
          class="continuation-panel__sync-button"
          :disabled="state.isSyncingAnalysis.value || !insightStore.currentBookId"
          size="sm"
          @click="handleManualSync"
        >
          <span v-if="state.isSyncingAnalysis.value">同步中...</span>
          <template v-else>
            <UiIcon name="refresh" />
            <span>同步分析数据</span>
          </template>
        </UiButton>
      </div>
      <ProductWizardSteps
        class="continuation-panel__steps"
        :steps="wizardSteps"
        :active-index="state.currentStep.value"
        aria-label="续写步骤"
        @select="navigateToStep"
      />
      <div class="continuation-panel__step-content">
        <div v-show="state.currentStep.value === 0" class="continuation-panel__step-panel">
          <ProductSectionHeader title="续写设置" icon-name="file-text" />
          <UiFormGrid>
            <UiField
              variant="settings"
              label="续写页数"
              hint="建议 10-20 页"
              control-id="continuationPageCount"
            >
              <UiNumberField
                v-model="state.pageCount.value"
                input-id="continuationPageCount"
                :min="5"
                :max="50"
              />
            </UiField>
            <UiField
              variant="settings"
              label="画风参考页数"
              hint="用于维持画风一致性"
              control-id="continuationStyleRefPages"
            >
              <UiNumberField
                v-model="state.styleRefPages.value"
                input-id="continuationStyleRefPages"
                :min="1"
                :max="10"
              />
            </UiField>
          </UiFormGrid>
          <UiField
            variant="settings"
            label="续写方向（可选）"
            hint="留空将自动根据剧情发展生成"
            control-id="continuationDirection"
          >
            <UiTextarea
              id="continuationDirection"
              v-model="state.continuationDirection.value"
              rows="4"
              variant="panel"
              placeholder="例如：延续主线剧情，探索新的冒险..."
            />
          </UiField>
          <CharacterManagementPanel
            v-if="insightStore.currentBookId"
            :book-id="insightStore.currentBookId"
            :character-management="charMgmt"
            :is-loading="state.isLoading.value"
            :state="state"
          />
          <ProductActionRow aria-label="续写设置操作" divider justify="between">
            <UiButton variant="danger" @click="requestClearAndRestart">
              <UiIcon name="trash" />
              <span>清除数据重新开始</span>
            </UiButton>
            <UiButton variant="primary" :disabled="!canProceedToScript" @click="goToStep(1)">
              <span>下一步：生成脚本</span>
              <UiIcon name="chevron-right" />
            </UiButton>
          </ProductActionRow>
        </div>
        <div v-show="state.currentStep.value === 1" class="continuation-panel__step-panel">
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
          <ProductActionRow aria-label="脚本生成步骤操作" divider justify="between">
            <UiButton variant="secondary" @click="goToStep(0)">
              <UiIcon name="chevron-left" />
              <span>上一步</span>
            </UiButton>
            <UiButton variant="primary" :disabled="!canProceedToPages" @click="goToStep(2)">
              <span>下一步：页面剧情</span>
              <UiIcon name="chevron-right" />
            </UiButton>
          </ProductActionRow>
        </div>
        <div v-show="state.currentStep.value === 2" class="continuation-panel__step-panel">
          <PageDetailsPanel
            :pages="state.pages.value"
            :is-generating="state.isGeneratingPages.value"
            @generate-details="handleGeneratePageDetails"
            @save-changes="handleSavePageChanges"
            @story-change="handleStoryContentChange"
          />
          <ProductActionRow aria-label="页面剧情步骤操作" divider justify="between">
            <UiButton variant="secondary" @click="goToStep(1)">
              <UiIcon name="chevron-left" />
              <span>上一步</span>
            </UiButton>
            <UiButton variant="primary" :disabled="!canProceedToImages" @click="goToStep(3)">
              <span>下一步：图片生成</span>
              <UiIcon name="chevron-right" />
            </UiButton>
          </ProductActionRow>
        </div>
        <div v-show="state.currentStep.value === 3" class="continuation-panel__step-panel">
          <ImageGenerationPanel
            :pages="state.pages.value"
            :is-generating="imageGen.isGenerating.value"
            :progress="imageGen.generationProgress.value"
            :book-id="insightStore.currentBookId || ''"
            :state="state"
            @batch-generate="handleBatchGenerate"
            @regenerate="handleRegenerateImage"
            @use-previous="handleUsePrevious"
            @prompt-change="handlePromptChange"
          />
          <ExportPanel
            v-if="insightStore.currentBookId"
            :book-id="insightStore.currentBookId"
            :generated-count="generatedPagesCount"
            :state="state"
            @clear-and-restart="clearAndRestart"
          />
          <ProductActionRow aria-label="图片生成步骤操作" divider justify="start">
            <UiButton variant="secondary" @click="goToStep(2)">
              <UiIcon name="chevron-left" />
              <span>上一步</span>
            </UiButton>
          </ProductActionRow>
        </div>
      </div>
    </div>
  </ProductWorkspacePanel>
</template>
<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import ProductWorkspacePanel from '@/components/product/ProductWorkspacePanel.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductWizardSteps, {
  type ProductWizardStep,
} from '@/components/product/ProductWizardSteps.vue'
import { ref, computed, watch, onBeforeUnmount } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import { useContinuationState } from '@/composables/continuation/useContinuationState'
import { useCharacterManagement } from '@/composables/continuation/useCharacterManagement'
import { useImageGeneration } from '@/composables/continuation/useImageGeneration'
import CharacterManagementPanel from './continuation/CharacterManagementPanel.vue'
import ScriptGenerationPanel from './continuation/ScriptGenerationPanel.vue'
import PageDetailsPanel from './continuation/PageDetailsPanel.vue'
import ImageGenerationPanel from './continuation/ImageGenerationPanel.vue'
import ExportPanel from './continuation/ExportPanel.vue'
import * as continuationApi from '@/api/continuation'
import type { PageContent } from '@/api/continuation'
import { hasUsableStoryContent } from '@/composables/continuation/promptValidation'
import { confirmProductAction } from '@/composables/useProductConfirm'
import type { PageStoryField, PageStoryValue } from './continuation/pageStoryTypes'
const insightStore = useInsightStore()
const taskCenterStore = useTaskCenterStore()
const bookId = computed(() => insightStore.currentBookId || '')
const stateComposable = useContinuationState(bookId)
const charMgmtComposable = useCharacterManagement(bookId, stateComposable)
const imageGenComposable = useImageGeneration(bookId, stateComposable)
const state = stateComposable
const charMgmt = charMgmtComposable
const imageGen = imageGenComposable
const stepNames = ['角色设置', '生成脚本', '页面剧情', '图片生成/导出']
const isGeneratingScript = ref(false)
const isSavingScript = ref(false)
const scriptDirty = ref(false)
const lastSavedScriptText = ref('')
let promptSaveTimer: ReturnType<typeof setTimeout> | null = null
let storySaveTimer: ReturnType<typeof setTimeout> | null = null

function clearPromptSaveTimer() {
  if (promptSaveTimer) {
    clearTimeout(promptSaveTimer)
    promptSaveTimer = null
  }
}

function clearStorySaveTimer() {
  if (storySaveTimer) {
    clearTimeout(storySaveTimer)
    storySaveTimer = null
  }
}

function clearPendingAutosaves() {
  clearPromptSaveTimer()
  clearStorySaveTimer()
}

function resetLocalWorkflowState() {
  isGeneratingScript.value = false
  isSavingScript.value = false
  clearStorySaveTimer()
}
const canProceedToScript = computed(() => {
  return state.isDataReady.value && state.characters.value.length > 0
})
const canProceedToPages = computed(() => {
  return state.chapterScript.value !== null
})
const canProceedToImages = computed(() => {
  return (
    state.pages.value.length > 0 &&
    state.pages.value.every(p => p.status !== 'failed' && hasUsableStoryContent(p))
  )
})
const generatedPagesCount = computed(() => {
  return state.pages.value.filter(p => p.image_url && p.status === 'generated').length
})
const workflowMessage = computed(() => state.errorMessage.value || state.successMessage.value)
const workflowMessageTone = computed<'success' | 'danger' | 'info'>(() => {
  const messageType = state.messageType.value || (state.errorMessage.value ? 'error' : 'success')
  return messageType === 'error' ? 'danger' : messageType || 'info'
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
async function persistContinuationConfig(): Promise<string | null> {
  if (!insightStore.currentBookId) {
    return '当前未选择漫画'
  }
  try {
    await continuationApi.saveConfig(insightStore.currentBookId, {
      page_count: state.pageCount.value,
      style_reference_pages: state.styleRefPages.value,
      continuation_direction: state.continuationDirection.value,
    })
    return null
  } catch (error) {
    return error instanceof Error ? error.message : '网络错误'
  }
}
function canNavigateToStep(step: number): boolean {
  if (step === 0) return true
  if (step === 1) return canProceedToScript.value
  if (step === 2) return canProceedToPages.value
  if (step === 3) return canProceedToImages.value
  return false
}
const wizardSteps = computed<ProductWizardStep[]>(() => {
  return stepNames.map((label, index) => ({
    label,
    disabled: !canNavigateToStep(index),
  }))
})
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
async function handleGenerateScript(payload: {
  referenceTokens: string[] | null
  referenceImageCount: number
}) {
  if (!insightStore.currentBookId) return
  isGeneratingScript.value = true
  state.errorMessage.value = ''
  try {
    const jobId = await continuationApi.generateScriptWithRefs(
      insightStore.currentBookId,
      state.continuationDirection.value,
      state.pageCount.value,
      payload.referenceTokens || undefined,
      payload.referenceImageCount
    )
    state.showMessage('脚本生成任务已进入任务中心，关闭浏览器也会继续运行', 'info')
    await taskCenterStore.waitForJob(jobId)
    await state.initializeData()
    lastSavedScriptText.value = state.chapterScript.value?.script_text ?? ''
    scriptDirty.value = false
    state.showMessage('脚本生成成功，旧页面剧情已标记为需要重新生成', 'success')
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
    const savedScript = await continuationApi.saveScript(
      insightStore.currentBookId,
      state.chapterScript.value
    )
    state.chapterScript.value = savedScript
    lastSavedScriptText.value = savedScript.script_text
    scriptDirty.value = false
    if (shouldInvalidatePages) {
      state.pages.value = []
      await persistPages([])
      state.showMessage('脚本已更新，已有页面剧情已清空，请重新生成。', 'info')
      return true
    }
    if (showSuccessMessage) {
      state.showMessage('脚本已保存', 'success')
    }
    return true
  } catch (error) {
    state.showMessage(
      '脚本保存失败: ' + (error instanceof Error ? error.message : '网络错误'),
      'error'
    )
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
function applyPageStoryEdit(page: PageContent, field: PageStoryField, value: PageStoryValue) {
  if (field === 'characters') {
    page.characters = Array.isArray(value) ? value : []
    return
  }
  page[field] = typeof value === 'string' ? value : ''
}
function handleStoryContentChange(
  pageNumber: number,
  field: PageStoryField,
  value: PageStoryValue
) {
  const page = state.pages.value.find(item => item.page_number === pageNumber)
  if (!page) return
  applyPageStoryEdit(page, field, value)

  if (storySaveTimer) {
    clearTimeout(storySaveTimer)
  }
  storySaveTimer = setTimeout(async () => {
    if (!insightStore.currentBookId || state.pages.value.length === 0) return
    try {
      await persistPages()
    } catch (error) {
      state.showMessage(
        '页面剧情保存失败: ' + (error instanceof Error ? error.message : '网络错误'),
        'error'
      )
    }
  }, 600)
}
function handleResetScript() {
  if (!state.chapterScript.value) return
  state.chapterScript.value.script_text = lastSavedScriptText.value
  scriptDirty.value = false
}
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
  try {
    const jobId = await continuationApi.generateAllPageDetails(insightStore.currentBookId)
    state.showMessage('页面剧情任务已整体进入任务中心，关闭浏览器也会继续运行', 'info')
    await taskCenterStore.waitForJob(jobId)
    await state.initializeData()
    state.showMessage(`页面剧情生成完成 (${state.pages.value.length} 页)`, 'success')
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
  if (insightStore.currentBookId) {
    await continuationApi.activatePageImageVersion(
      insightStore.currentBookId,
      pageNumber,
      page.previous_url
    )
    await state.initializeData()
  }
}
async function handlePromptChange(pageNumber: number, prompt: string) {
  const page = state.pages.value.find(item => item.page_number === pageNumber)
  if (!page) return
  page.final_prompt = prompt

  if (promptSaveTimer) {
    clearTimeout(promptSaveTimer)
  }
  promptSaveTimer = setTimeout(async () => {
    if (!insightStore.currentBookId || state.pages.value.length === 0) return
    try {
      await persistPages()
    } catch (error) {
      state.showMessage(
        '提示词保存失败: ' + (error instanceof Error ? error.message : '网络错误'),
        'error'
      )
    }
  }, 600)
}
async function handleManualSync() {
  await state.syncAnalysisData('manual')
}
async function clearAndRestart() {
  if (!insightStore.currentBookId) return
  try {
    clearPendingAutosaves()
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

async function requestClearAndRestart() {
  const confirmed = await confirmProductAction({
    title: '清空续写数据',
    message: '确定要清空所有续写数据并重新开始吗？此操作不可恢复。',
    confirmText: '清空',
    cancelText: '取消',
    tone: 'danger',
  })
  if (!confirmed) return
  await clearAndRestart()
}
async function goToStep(step: number) {
  if (state.currentStep.value === 0 && step !== 0) {
    const configError = await persistContinuationConfig()
    if (configError) {
      state.showMessage(`续写配置保存失败：${configError}`, 'info')
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
watch(
  () => insightStore.currentBookId,
  newBookId => {
    clearPendingAutosaves()
    resetLocalWorkflowState()
    if (newBookId) {
      state.initializeData()
    } else {
      state.resetState()
    }
  },
  { immediate: true }
)
watch(
  () => insightStore.dataRefreshKey,
  async (newKey, previousKey) => {
    if (!insightStore.currentBookId || newKey <= 0 || newKey === previousKey) return
    await state.syncAnalysisData('auto')
  }
)
watch(
  () => state.chapterScript.value,
  script => {
    if (script) {
      lastSavedScriptText.value = script.script_text
      scriptDirty.value = false
    } else {
      lastSavedScriptText.value = ''
      scriptDirty.value = false
    }
  },
  { immediate: true }
)
onBeforeUnmount(() => {
  clearPendingAutosaves()
})
</script>

<style scoped>
.continuation-panel {
  --continuation-panel-sync-background: var(--color-surface-quiet);

  width: 100%;
  min-width: 0;
  min-height: 100%;
}

.continuation-panel__sync-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 16px;
  margin-bottom: 16px;
  border: 1px solid var(--color-border-muted);
  border-radius: 12px;
  background: var(--continuation-panel-sync-background);
}

.continuation-panel__sync-meta {
  display: flex;
  flex: 1 1 240px;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.continuation-panel__sync-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-default);
}

.continuation-panel__sync-status {
  font-size: 12px;
  color: var(--color-text-supporting);
}

.continuation-panel__message {
  margin-bottom: 16px;
}

.continuation-panel__steps {
  margin-bottom: 24px;
}

.continuation-panel__step-content {
  min-width: 0;
  overflow: hidden;
  background: var(--color-surface-base);
  border-radius: 12px;
  border: 1px solid var(--color-border-muted);
}

.continuation-panel__step-panel {
  min-width: 0;
  padding: 24px;
}

@media (--breakpoint-sm-down) {
  .continuation-panel__sync-bar {
    align-items: stretch;
  }

  .continuation-panel__sync-button {
    justify-content: center;
    width: 100%;
  }
}
</style>
