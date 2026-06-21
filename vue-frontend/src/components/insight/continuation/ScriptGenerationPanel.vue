<template>
  <div class="script-panel">
    <h3>📝 生成续写脚本</h3>

    <div class="script-editor" v-if="script">
      <div class="script-header">
        <h4>{{ script.chapter_title }}</h4>
        <span class="script-meta">共 {{ script.page_count }} 页 · {{ script.generated_at }}</span>
      </div>

      <UiTextarea
        v-model="scriptText"
        class="script-textarea"
        rows="15"
        placeholder="脚本将在此显示..."
        @update:model-value="handleScriptInput"
      />

      <div class="script-actions">
        <UiButton variant="secondary" @click="$emit('reset-script')" size="sm">↺ 重置</UiButton>
        <UiButton variant="secondary" :disabled="!script || isSaving" @click="handleSave" size="sm">
          {{ isSaving ? '保存中...' : '💾 保存' }}
        </UiButton>
      </div>
    </div>

    <div v-else class="no-script">
      <p>点击下方按钮生成续写脚本</p>
    </div>

    <!-- 参考图配置区域 -->
    <div class="reference-config">
      <div class="config-row">
        <label>VLM参考图数:</label>
        <UiInput
          type="number"
          v-model.number="refCount"
          min="1"
          max="10"
          class="ref-count-input"
        />
        <UiButton
          variant="secondary"
          class="ref-btn"
          @click="openReferenceSelector" size="sm"
        >
          📷 参考图 ({{ getDisplayRefCount() }})
        </UiButton>
      </div>
    </div>

    <UiButton
      variant="primary"
      block
      :disabled="isGenerating"
      @click="handleGenerate"
    >
      {{ isGenerating ? '生成中...' : '🎯 生成脚本' }}
    </UiButton>

    <!-- 参考图选择器 -->
    <ReferenceImageSelector
      v-model:visible="selectorVisible"
      mode="script"
      :max-count="refCount"
      :original-images="availableOriginalImages"
      :continuation-images="[]"
      :character-forms="[]"
      :initial-selection="selectedReferenceTokens"
      :book-id="bookId"
      @confirm="handleSelectorConfirm"
    />
  </div>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { ref, watch } from 'vue'
import type { ChapterScript, MangaImageInfo } from '@/api/continuation'
import { getAvailableImages } from '@/api/continuation'
import ReferenceImageSelector from './ReferenceImageSelector.vue'

const props = defineProps<{
  script: ChapterScript | null
  isGenerating: boolean
  isSaving?: boolean
  bookId: string
}>()

const emit = defineEmits<{
  'generate': [payload: { referenceTokens: string[] | null; referenceImageCount: number }]
  'update-script': [scriptText: string]
  'save-script': []
  'reset-script': []
}>()

const scriptText = ref('')
const refCount = ref(5)
const selectorVisible = ref(false)
const selectedReferenceTokens = ref<string[]>([])
const availableOriginalImages = ref<MangaImageInfo[]>([])

watch(() => props.script?.script_text, (newScriptText) => {
  scriptText.value = newScriptText || ''
}, { immediate: true })

watch(() => props.script, (newScript) => {
  if (!newScript) {
    selectedReferenceTokens.value = []
    selectorVisible.value = false
  }
})

// 加载可用图片列表
async function loadAvailableImages() {
  if (!props.bookId) return

  try {
    const response = await getAvailableImages(props.bookId, 'script')
    if (response.success && response.original_images) {
      availableOriginalImages.value = response.original_images
    }
  } catch (error) {
    console.error('加载可用图片失败:', error)
  }
}

// 打开参考图选择器
function openReferenceSelector() {
  // 确保已加载图片列表
  if (availableOriginalImages.value.length === 0) {
    loadAvailableImages()
  }
  selectorVisible.value = true
}

// 选择器确认
function handleSelectorConfirm(tokens: string[]) {
  selectedReferenceTokens.value = tokens
}

// 获取显示的参考图数量
function getDisplayRefCount(): number {
  // 如果用户已手动选择，显示选择的数量
  if (selectedReferenceTokens.value.length > 0) {
    return selectedReferenceTokens.value.length
  }
  // 否则显示配置的默认数量
  return refCount.value
}

// 生成脚本
function handleGenerate() {
  // 如果用户选择了参考图，传递选择的路径；否则传null使用自动逻辑
  const refs = selectedReferenceTokens.value.length > 0 ? selectedReferenceTokens.value : null
  emit('generate', {
    referenceTokens: refs,
    referenceImageCount: refCount.value,
  })
}

function handleScriptInput(value: string) {
  emit('update-script', value)
}

function handleSave() {
  emit('save-script')
}

// 监听 bookId 变化
watch(() => props.bookId, (newBookId) => {
  if (newBookId) {
    loadAvailableImages()
    selectedReferenceTokens.value = []
    refCount.value = 5
  } else {
    refCount.value = 5
    availableOriginalImages.value = []
    selectedReferenceTokens.value = []
    selectorVisible.value = false
  }
}, { immediate: true })
</script>

<style scoped>
.script-panel {
  padding: 24px;

  --ui-button-padding: 10px 20px;
  --ui-button-radius: 8px;
  --ui-button-font-size: 14px;
  --ui-button-primary-background: var(--color-surface-brand);
  --ui-button-primary-color: white;
  --ui-button-primary-shadow: none;
  --ui-button-primary-hover-background: var(--color-surface-brand-strong);
  --ui-button-primary-hover-transform: none;
  --ui-button-primary-hover-shadow: none;
  --ui-button-secondary-background: var(--color-surface-muted);
  --ui-button-secondary-color: var(--color-text-default, var(--color-text-default));
  --ui-button-secondary-border: 1px solid var(--color-border-muted, var(--color-border-default));
  --ui-button-secondary-hover-background: var(--color-surface-hover);
  --ui-button-secondary-hover-border-color: var(--color-border-muted, var(--color-border-default));
  --ui-button-sm-padding: 6px 12px;
  --ui-button-sm-font-size: 13px;
  --ui-button-disabled-opacity: 0.5;
}

.script-panel h3 {
  margin: 0 0 20px;
  font-size: 18px;
  font-weight: 600;
}

.script-editor {
  margin-bottom: 20px;
}

.script-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.script-header h4 {
  margin: 0;
  font-size: 16px;
}

.script-meta {
  font-size: 13px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.script-textarea {
  width: 100%;
  padding: 16px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 8px;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.6;
  resize: vertical;
}

.script-textarea:focus {
  outline: none;
  border-color: var(--color-border-brand);
}

.script-actions {
  margin-top: 12px;
}

.no-script {
  text-align: center;
  padding: 40px 20px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.no-script p {
  margin: 0;
}

/* 参考图配置区域 */
.reference-config {
  margin-bottom: 16px;
  padding: 12px 16px;
  background: var(--color-surface-subtle);
  border-radius: 8px;
}

.config-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.config-row label {
  font-size: 14px;
  color: var(--color-text-default, var(--color-text-default));
  white-space: nowrap;
}

.ref-count-input {
  width: 60px;
  padding: 6px 10px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 6px;
  font-size: 14px;
  text-align: center;
}

.ref-count-input:focus {
  outline: none;
  border-color: var(--color-border-brand);
}

.ref-btn {
  margin-left: auto;
}
</style>
