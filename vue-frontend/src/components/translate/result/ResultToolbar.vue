<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
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
      variant="secondary"
      @click="$emit('toggleImageView')"
    >
      <UiIcon name="eye" />
      {{ showOriginal ? `查看${processedImageLabel}` : '查看原图' }}
    </UiButton>

    <UiButton
      :variant="isEditMode ? 'primary' : 'secondary'"
      @click="$emit('toggleEditMode')"
    >
      <UiIcon name="pencil" />
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
      variant="danger"
      title="重新翻译所有失败的图片"
      @click="$emit('retryFailed')"
    >
      <UiIcon name="refresh" />
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
