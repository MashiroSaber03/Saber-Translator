<template>
  <ProductRecordCard class="form-tile" :class="{ 'form-tile--disabled': form.enabled === false }">
    <div class="form-tile__image-section">
      <img
        v-if="form.reference_image"
        class="form-tile__image"
        :src="formImageUrl"
        :alt="`${characterName} ${form.form_name}参考图`"
      >
      <ProductEmptyState
        v-else
        class="form-tile__image-empty-state"
        icon-name="camera"
        role="note"
        size="compact"
        title="未上传参考图"
      />
      <ProductFileDropzone
        :input-id="`formTileUpload-${form.form_id}`"
        class="form-tile__upload-overlay"
        :label="`上传 ${characterName} ${form.form_name} 参考图`"
        accept="image/*"
        @select="handleUpload"
      >
        <span class="form-tile__upload-text">{{ form.reference_image ? '更换图片' : '上传图片' }}</span>
      </ProductFileDropzone>
    </div>

    <div class="form-tile__content">
      <div class="form-tile__header">
        <h4 class="form-tile__title">{{ form.form_name }}</h4>
        <ProductChipList
          v-if="form.enabled === false"
          aria-label="形态状态"
          :items="disabledStatusChips"
        />
      </div>
      <p v-if="form.description" class="form-tile__description">{{ form.description }}</p>
    </div>

    <div class="form-tile__actions">
      <ProductActionRow
        class="form-tile__action-row"
        justify="start"
        :aria-label="`${characterName} ${form.form_name}三视图操作`"
      >
        <UiSwitch
          size="sm"
          :model-value="form.enabled !== false"
          :title="form.enabled !== false ? '点击禁用' : '点击启用'"
          :ariaLabel="`启用 ${characterName} ${form.form_name}`"
          @change="$emit('toggle-enabled', $event)"
        />
        <UiIconButton variant="primary" :label="`生成 ${characterName} ${form.form_name} 三视图`" @click="$emit('generate-orthographic')">
          <UiIcon name="palette" size="16" />
        </UiIconButton>
        <UiIconButton v-if="form.reference_image" variant="danger" :label="`删除 ${characterName} ${form.form_name} 参考图`" @click="$emit('delete-image')">
          <UiIcon name="trash" size="16" />
        </UiIconButton>
      </ProductActionRow>
      <ProductActionRow
        class="form-tile__action-row form-tile__action-row--secondary"
        justify="end"
        :aria-label="`${characterName} ${form.form_name}形态管理操作`"
      >
        <UiIconButton variant="plain" size="sm" :label="`编辑 ${characterName} ${form.form_name}`" @click="$emit('edit')">
          <UiIcon name="pencil" size="14" />
        </UiIconButton>
        <UiIconButton variant="danger" size="sm" :label="`删除 ${characterName} ${form.form_name}`" @click="$emit('delete')">
          <UiIcon name="trash" size="14" />
        </UiIconButton>
      </ProductActionRow>
    </div>
  </ProductRecordCard>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiSwitch from '@/components/ui/UiSwitch.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductFileDropzone from '@/components/product/ProductFileDropzone.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import type { CharacterForm } from '@/api/continuation'

const props = defineProps<{
  form: CharacterForm
  characterName: string
  formImageUrl: string
}>()

const emit = defineEmits<{
  'edit': []
  'delete': []
  'upload-image': [file: File]
  'delete-image': []
  'generate-orthographic': []
  'toggle-enabled': [enabled: boolean]
}>()

const disabledStatusChips = computed(() => props.form.enabled === false
  ? [{ id: 'disabled', label: '已禁用', tone: 'warning' as const }]
  : [])

function handleUpload(files: File[]) {
  const file = files[0]
  if (!file) return

  emit('upload-image', file)
}
</script>

<style scoped>
.form-tile {
  --form-tile-upload-text-shadow: var(--shadow-soft);
  --form-tile-upload-overlay-start: var(--color-action-primary);
  --form-tile-upload-overlay-end: var(--color-action-primary-hover);
  --product-record-card-background: var(--color-surface-card);
  --product-record-card-border: var(--color-border-muted);
  --product-record-card-radius: 12px;
  --product-record-card-padding: 0;
  --product-record-card-gap: 0;

  overflow: hidden;
}

.form-tile:hover {
  transform: translateY(-2px);
}

.form-tile--disabled {
  opacity: 0.6;
  filter: grayscale(60%);
}

.form-tile--disabled:hover {
  transform: none;
}

.form-tile__image-section {
  aspect-ratio: 1;
  position: relative;
  background: var(--color-surface-muted);
  overflow: hidden;
}

.form-tile__image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.form-tile__image-empty-state {
  width: 100%;
  height: 100%;
}

.form-tile__upload-overlay {
  --product-file-dropzone-background: linear-gradient(135deg, var(--form-tile-upload-overlay-start), var(--form-tile-upload-overlay-end));
  --product-file-dropzone-background-hover: linear-gradient(135deg, var(--form-tile-upload-overlay-start), var(--form-tile-upload-overlay-end));
  --product-file-dropzone-border: transparent;
  --product-file-dropzone-border-hover: transparent;
  --product-file-dropzone-color: var(--color-text-inverse);
  --product-file-dropzone-padding: 0;
  --product-file-dropzone-radius: 0;

  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.25s ease;
  cursor: pointer;
}

.form-tile__upload-text {
  color: var(--color-text-inverse);
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0;
  text-shadow: 0 1px 2px var(--form-tile-upload-text-shadow);
}

.form-tile__image-section:hover .form-tile__upload-overlay,
.form-tile__image-section:focus-within .form-tile__upload-overlay {
  opacity: 1;
}

.form-tile__content {
  padding: 14px 12px 12px;
  flex: 1;
  display: flex;
  flex-direction: column;
}

.form-tile__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 6px;
}

.form-tile__title {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text-strong);
  flex: 1;
  line-height: 1.3;
}

.form-tile__description {
  margin: 0;
  font-size: 11px;
  color: var(--color-text-supporting);
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.form-tile__actions {
  padding: 10px 12px;
  background: var(--color-surface-muted);
  border-top: 1px solid var(--color-border-muted);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-tile__action-row {
  gap: 6px;
}

.form-tile__action-row--secondary {
  padding-top: 2px;
}

</style>
