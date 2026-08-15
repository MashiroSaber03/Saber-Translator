<script setup lang="ts">
import { useId } from 'vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'

interface ProductSelectableImageItem {
  alt: string
  disabled?: boolean
  id: string | number
  label: string
  selected: boolean
  src: string
}

withDefaults(defineProps<{
  ariaLabel?: string
  items: ProductSelectableImageItem[]
}>(), {
  ariaLabel: '图片选择列表',
})

defineEmits<{
  toggle: [id: string | number]
}>()

const inputIdPrefix = useId()

function inputId(index: number): string {
  return `${inputIdPrefix}-${index}`
}
</script>

<template>
  <div class="product-selectable-image-grid" role="list" :aria-label="ariaLabel">
    <div
      v-for="(item, index) in items"
      :key="item.id"
      class="product-selectable-image-grid__item"
      :class="{
        'product-selectable-image-grid__item--selected': item.selected,
        'product-selectable-image-grid__item--disabled': item.disabled,
      }"
      role="listitem"
    >
      <span class="product-selectable-image-grid__checkbox">
        <UiCheckbox
          :input-id="inputId(index)"
          :aria-label="`选择${item.label}`"
          :disabled="item.disabled"
          :model-value="item.selected"
          @change="$emit('toggle', item.id)"
        />
      </span>
      <label class="product-selectable-image-grid__body" :for="inputId(index)">
        <span class="product-selectable-image-grid__preview">
          <img class="product-selectable-image-grid__preview-image" :src="item.src" :alt="item.alt" loading="lazy">
        </span>
        <span class="product-selectable-image-grid__label">{{ item.label }}</span>
      </label>
    </div>
  </div>
</template>

<style scoped>
.product-selectable-image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 12px;
  max-height: 300px;
  padding: 4px;
  overflow-y: auto;
}

.product-selectable-image-grid__item {
  position: relative;
  overflow: hidden;
  border: 2px solid var(--color-border-muted);
  border-radius: 8px;
  cursor: pointer;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.product-selectable-image-grid__item:hover,
.product-selectable-image-grid__item--selected {
  border-color: var(--color-action-primary);
}

.product-selectable-image-grid__item--selected {
  box-shadow: 0 0 0 2px var(--color-focus-brand-subtle);
}

.product-selectable-image-grid__item--disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.product-selectable-image-grid__checkbox {
  position: absolute;
  top: 6px;
  left: 6px;
  z-index: var(--z-local);
}

.product-selectable-image-grid__body {
  display: block;
  cursor: inherit;
}

.product-selectable-image-grid__preview {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  overflow: hidden;
  background: var(--color-surface-subtle);
  aspect-ratio: 3/4;
}

.product-selectable-image-grid__preview-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.product-selectable-image-grid__label {
  display: block;
  padding: 6px;
  background: var(--color-surface-base);
  color: var(--color-text-supporting);
  font-size: 12px;
  text-align: center;
}
</style>
