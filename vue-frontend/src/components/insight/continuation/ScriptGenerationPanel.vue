<template>
  <div class="script-panel">
    <h3>📝 生成续写脚本</h3>

    <div class="script-editor" v-if="script">
      <div class="script-header">
        <h4>{{ script.chapter_title }}</h4>
        <span class="script-meta">共 {{ script.page_count }} 页 · {{ script.generated_at }}</span>
      </div>

      <textarea
        v-model="scriptText"
        class="script-textarea"
        rows="15"
        placeholder="脚本将在此显示..."
      ></textarea>

      <div class="script-actions">
        <button class="btn secondary small" @click="scriptText = script!.script_text">↺ 重置</button>
      </div>
    </div>

    <div v-else class="no-script">
      <p>点击下方按钮生成续写脚本</p>
    </div>

    <!-- 参考图配置区域 -->
    <div class="reference-config">
      <div class="config-row">
        <label>VLM参考图数:</label>
        <input
          type="number"
          v-model.number="refCount"
          min="1"
          max="10"
          class="ref-count-input"
        />
        <button
          class="btn secondary small ref-btn"
          @click="openReferenceSelector"
        >
          📷 参考图 ({{ getDisplayRefCount() }})
        </button>
      </div>
    </div>

    <button
      class="btn primary"
      :disabled="isGenerating"
      @click="handleGenerate"
    >
      {{ isGenerating ? '生成中...' : '🎯 生成脚本' }}
    </button>

    <!-- 参考图选择器 -->
    <ReferenceImageSelector
      v-model:visible="selectorVisible"
      mode="script"
      :max-count="refCount"
      :original-images="availableOriginalImages"
      :continuation-images="[]"
      :character-forms="[]"
      :initial-selection="selectedRefImages"
      :book-id="bookId"
      @confirm="handleSelectorConfirm"
      @cancel="handleSelectorCancel"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import type { ChapterScript, MangaImageInfo } from '@/api/continuation'
import { getAvailableImages } from '@/api/continuation'
import ReferenceImageSelector from './ReferenceImageSelector.vue'

const props = defineProps<{
  script: ChapterScript | null
  isGenerating: boolean
  bookId: string
}>()

const emit = defineEmits<{
  'generate': [referenceImages: string[] | null]
}>()

const scriptText = ref('')
const refCount = ref(5)
const selectorVisible = ref(false)
const selectedRefImages = ref<string[]>([])
const availableOriginalImages = ref<MangaImageInfo[]>([])

watch(() => props.script, (newScript) => {
  if (newScript) {
    scriptText.value = newScript.script_text
  }
}, { immediate: true })

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
function handleSelectorConfirm(paths: string[]) {
  selectedRefImages.value = paths
}

// 选择器取消
function handleSelectorCancel() {
  // 不做任何操作，保持之前的选择
}

// 获取显示的参考图数量
function getDisplayRefCount(): number {
  // 如果用户已手动选择，显示选择的数量
  if (selectedRefImages.value.length > 0) {
    return selectedRefImages.value.length
  }
  // 否则显示配置的默认数量
  return refCount.value
}

// 生成脚本
function handleGenerate() {
  // 如果用户选择了参考图，传递选择的路径；否则传null使用自动逻辑
  const refs = selectedRefImages.value.length > 0 ? selectedRefImages.value : null
  emit('generate', refs)
}

// 组件挂载时加载可用图片
onMounted(() => {
  if (props.bookId) {
    loadAvailableImages()
  }
})

// 监听 bookId 变化
watch(() => props.bookId, (newBookId) => {
  if (newBookId) {
    loadAvailableImages()
    selectedRefImages.value = []
  }
})
</script>

<style scoped>
.script-panel {
  padding: 24px;
}

.script-panel h3 {
  margin: 0 0 20px 0;
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
  color: var(--text-secondary, #666);
}

.script-textarea {
  width: 100%;
  padding: 16px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 8px;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.6;
  resize: vertical;
}

.script-textarea:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
}

.script-actions {
  margin-top: 12px;
}

.no-script {
  text-align: center;
  padding: 40px 20px;
  color: var(--text-secondary, #666);
}

.no-script p {
  margin: 0;
}

/* 参考图配置区域 */
.reference-config {
  margin-bottom: 16px;
  padding: 12px 16px;
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 8px;
}

.config-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.config-row label {
  font-size: 14px;
  color: var(--text-primary, #333);
  white-space: nowrap;
}

.ref-count-input {
  width: 60px;
  padding: 6px 10px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 6px;
  font-size: 14px;
  text-align: center;
}

.ref-count-input:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
}

.ref-btn {
  margin-left: auto;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  width: 100%;
}

.btn.primary {
  background: var(--primary, #6366f1);
  color: white;
}

.btn.primary:hover:not(:disabled) {
  background: var(--primary-dark, #4f46e5);
}

.btn.primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn.secondary {
  background: var(--bg-secondary, #f3f4f6);
  color: var(--text-primary, #333);
  border: 1px solid var(--border-color, #e0e0e0);
}

.btn.secondary:hover {
  background: var(--bg-hover, #e5e7eb);
}

.btn.small {
  padding: 6px 12px;
  font-size: 13px;
  width: auto;
}
</style>
