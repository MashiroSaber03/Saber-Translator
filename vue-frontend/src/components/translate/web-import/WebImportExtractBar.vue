<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'
import type { WebImportEngine, WebImportState } from '@/types/webImport'

type SupportStatus = {
  message: string
  tone: 'info' | 'success' | 'neutral'
}

const props = withDefaults(defineProps<{
  checkingSupport: boolean
  focusRequestId?: number
  galleryDLAvailable: boolean
  galleryDLSupported: boolean
  isProcessing: boolean
  selectedEngine: WebImportEngine
  status: WebImportState['status']
  urlInput: string
}>(), {
  focusRequestId: 0,
})

const emit = defineEmits<{
  (event: 'extract'): void
  (event: 'update:selectedEngine', value: WebImportEngine): void
  (event: 'update:urlInput', value: string): void
}>()

const engineOptions: Array<{ label: string; value: WebImportEngine }> = [
  { label: '自动选择', value: 'auto' },
  { label: 'Gallery-DL', value: 'gallery-dl' },
  { label: 'AI Agent', value: 'ai-agent' },
]
const engineValues = engineOptions.map(option => option.value)
const sourceUrlInputRef = ref<{ focus: () => void } | null>(null)

const canExtract = computed(() => !props.isProcessing && props.urlInput.trim().length > 0)
const supportStatus = computed<SupportStatus | null>(() => {
  if (!props.urlInput.trim() || props.isProcessing) return null
  if (props.checkingSupport) {
    return {
      message: '正在检查网站支持情况...',
      tone: 'info',
    }
  }
  if (props.galleryDLSupported) {
    return {
      message: '该网站支持 Gallery-DL 高速下载',
      tone: 'success',
    }
  }
  if (props.galleryDLAvailable) {
    return {
      message: '该网站将使用 AI Agent 模式',
      tone: 'neutral',
    }
  }
  return null
})

function handleSubmit(): void {
  if (canExtract.value) {
    emit('extract')
  }
}

function isWebImportEngine(value: string | number): value is WebImportEngine {
  return typeof value === 'string' && engineValues.some(engine => engine === value)
}

function updateSelectedEngine(value: string | number): void {
  if (!isWebImportEngine(value)) return
  emit('update:selectedEngine', value)
}

watch(
  () => props.focusRequestId,
  async (requestId) => {
    if (requestId <= 0) return
    await nextTick()
    sourceUrlInputRef.value?.focus()
  }
)
</script>

<template>
  <form class="web-import-extract-bar" aria-label="网页导入提取" @submit.prevent="handleSubmit">
    <UiFormGrid class="web-import-extract-bar__form">
      <UiField
        variant="settings"
        label="网页 URL"
        control-id="webImportSourceUrl"
        label-visually-hidden
      >
        <UiInput
          ref="sourceUrlInputRef"
          id="webImportSourceUrl"
          :model-value="urlInput"
          type="url"
          placeholder="输入漫画网页 URL，如 https://example.com/chapter-1"
          :disabled="isProcessing"
          @update:model-value="emit('update:urlInput', String($event))"
        />
      </UiField>

      <UiField
        variant="settings"
        label="提取引擎"
        control-id="webImportEngine"
        label-visually-hidden
      >
        <UiSelect
          id="webImportEngine"
          :model-value="selectedEngine"
          :options="engineOptions"
          :disabled="isProcessing"
          @update:model-value="updateSelectedEngine"
        />
      </UiField>

      <ProductActionRow
        class="web-import-extract-bar__actions"
        aria-label="网页导入提取操作"
        justify="start"
      >
        <UiButton
          type="submit"
          variant="primary"
          class="web-import-extract-bar__submit"
          :disabled="!canExtract"
        >
          <UiSpinner v-if="status === 'extracting'" label="提取中" :decorative="false" />
          <span v-else aria-hidden="true">🔍</span>
          {{ status === 'extracting' ? '提取中...' : '开始提取' }}
        </UiButton>
      </ProductActionRow>
    </UiFormGrid>

    <ProductStatusBanner
      v-if="supportStatus"
      class="web-import-extract-bar__support-status"
      :tone="supportStatus.tone"
      role="status"
      aria-live="polite"
    >
      {{ supportStatus.message }}
    </ProductStatusBanner>

    <ProductStatusBanner
      class="web-import-extract-bar__notice"
      tone="warning"
      role="note"
    >
      <template #icon>⚠️</template>
      请仅爬取您有权访问的内容，并遵守目标网站的使用条款。
    </ProductStatusBanner>
  </form>
</template>

<style scoped>
.web-import-extract-bar {
  margin-bottom: 12px;
}

.web-import-extract-bar__form {
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr));
  align-items: end;
  gap: 11px;
  margin-bottom: 12px;
}

@media (--breakpoint-md-up) {
  .web-import-extract-bar__form {
    grid-template-columns: minmax(0, 1fr) 121px auto;
  }
}

.web-import-extract-bar__actions {
  display: flex;
  align-items: flex-end;
  min-width: 0;
}

.web-import-extract-bar__submit {
  --ui-button-padding: 10px 14px;
}

.web-import-extract-bar__notice {
  --product-status-banner-padding: 10px 14px;
}

.web-import-extract-bar__support-status,
.web-import-extract-bar__notice {
  margin-bottom: 16px;
}

@media (--breakpoint-sm-down) {
  .web-import-extract-bar__form {
    grid-template-columns: 1fr;
  }

  .web-import-extract-bar__actions,
  .web-import-extract-bar__submit {
    width: 100%;
  }
}
</style>
