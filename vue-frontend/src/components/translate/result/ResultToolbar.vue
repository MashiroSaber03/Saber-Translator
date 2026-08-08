<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiInput from '@/components/ui/UiInput.vue'

defineProps<{
  failedImageCount: number
  hasFailedImages: boolean
  hasProcessedImage: boolean
  imageSize: number
  isEditMode: boolean
  processedImageLabel: string
  showOriginal: boolean
}>()

defineEmits<{
  (e: 'retryFailed'): void
  (e: 'toggleEditMode'): void
  (e: 'toggleImageView'): void
  (e: 'updateImageSize', value: string | number | boolean): void
}>()
</script>

<template>
  <ProductActionRow
    class="result-toolbar"
    aria-label="图片查看操作"
    justify="center"
  >
    <UiButton
      v-if="hasProcessedImage"
      class="result-toolbar__control"
      variant="primary"
      @click="$emit('toggleImageView')"
    >
      {{ showOriginal ? `查看${processedImageLabel}` : '查看原图' }}
    </UiButton>

    <UiButton
      class="result-toolbar__control"
      :class="{ 'result-toolbar__control--active': isEditMode }"
      variant="primary"
      @click="$emit('toggleEditMode')"
    >
      {{ isEditMode ? '退出编辑' : '切换编辑模式' }}
    </UiButton>

    <UiField
      class="result-toolbar__image-size"
      variant="settings"
      layout="inline"
      label="图片大小"
      control-id="imageSize"
    >
      <UiInput
        id="imageSize"
        type="range"
        min="50"
        max="200"
        :model-value="imageSize"
        class="result-toolbar__slider"
        @update:model-value="$emit('updateImageSize', $event)"
      />
      <span class="result-toolbar__image-size-value">{{ imageSize }}%</span>
    </UiField>

    <UiButton
      v-if="hasFailedImages"
      class="result-toolbar__retry"
      variant="danger"
      title="重新翻译所有失败的图片"
      @click="$emit('retryFailed')"
    >
      重新翻译失败图片 ({{ failedImageCount }})
    </UiButton>
  </ProductActionRow>
</template>

<style scoped>
.result-toolbar {
  width: 100%;
  margin-bottom: 15px;
  gap: 20px;
}

.result-toolbar__control {
  --ui-button-padding: 10px 18px;
  --ui-button-font-size: 0.95em;
  --ui-button-primary-background: linear-gradient(135deg, var(--color-action-primary-hover) 0%, var(--color-action-primary) 100%);
  --ui-button-primary-hover-background: linear-gradient(135deg, color-mix(in srgb, var(--color-action-primary-hover) 82%, var(--color-overlay-backdrop-solid)) 0%, var(--color-action-primary) 100%);
  --ui-button-primary-shadow: 0 2px 6px color-mix(in srgb, var(--color-action-primary) 20%, transparent);
  --ui-button-primary-hover-shadow: 0 4px 10px color-mix(in srgb, var(--color-action-primary) 30%, transparent);
}

.result-toolbar__control--active {
  --ui-button-primary-background: linear-gradient(135deg, var(--color-surface-success) 0%, var(--color-action-success-strong) 100%);
  --ui-button-primary-hover-background: linear-gradient(135deg, color-mix(in srgb, var(--color-surface-success) 82%, var(--color-overlay-backdrop-solid)) 0%, var(--color-action-success-strong) 100%);
  --ui-button-primary-shadow: 0 2px 6px color-mix(in srgb, var(--color-surface-success) 20%, transparent);
  --ui-button-primary-hover-shadow: 0 4px 10px color-mix(in srgb, var(--color-surface-success) 30%, transparent);
}

.result-toolbar__retry {
  --ui-button-padding: 10px 18px;
  --ui-button-font-size: 0.95em;
  --ui-button-danger-background: linear-gradient(135deg, var(--color-status-warning-hover) 0%, var(--color-status-warning) 100%);
  --ui-button-danger-hover-background: linear-gradient(135deg, color-mix(in srgb, var(--color-status-warning-hover) 82%, var(--color-overlay-backdrop-solid)) 0%, var(--color-status-warning) 100%);
  --ui-button-danger-shadow: 0 2px 6px color-mix(in srgb, var(--color-status-warning) 20%, transparent);
  --ui-button-danger-hover-shadow: 0 4px 10px color-mix(in srgb, var(--color-status-warning) 30%, transparent);
}

.result-toolbar__image-size {
  display: flex;
  align-items: center;
  gap: 10px;
}

.result-toolbar__slider {
  width: 120px;
  cursor: pointer;
}

.result-toolbar__image-size-value {
  min-width: 45px;
  color: var(--color-text-muted);
  font-size: 14px;
  text-align: right;
}
</style>
