<script setup lang="ts">
import { computed, ref } from 'vue'

import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'

export interface ProductThumbnailGridItem {
  alt: string
  ariaLabel?: string
  cornerLabel?: string
  disabled?: boolean
  disabledTitle?: string
  fallbackLabel?: string
  id: string | number
  interactive?: boolean
  label: string
  marked?: boolean
  selected?: boolean
  selectedBadge?: string | number
  src: string
}

const props = withDefaults(defineProps<{
  ariaLabel?: string
  columns?: number
  items: ProductThumbnailGridItem[]
}>(), {
  ariaLabel: '缩略图列表',
  columns: undefined,
})

const emit = defineEmits<{
  select: [id: string | number]
}>()

const failedImageKeys = ref<Set<string>>(new Set())

const gridStyle = computed(() => {
  if (!props.columns) return undefined
  return {
    '--product-thumbnail-grid-column-count': String(props.columns),
  }
})

function imageKey(item: ProductThumbnailGridItem): string {
  return `${item.id}:${item.src}`
}

function hasImageFailed(item: ProductThumbnailGridItem): boolean {
  return failedImageKeys.value.has(imageKey(item))
}

function markImageFailed(item: ProductThumbnailGridItem): void {
  failedImageKeys.value = new Set([...failedImageKeys.value, imageKey(item)])
}

function selectItem(item: ProductThumbnailGridItem): void {
  if (item.disabled || item.interactive === false) return
  emit('select', item.id)
}

function itemAriaLabel(item: ProductThumbnailGridItem): string {
  return item.ariaLabel ?? `选择${item.label}`
}
</script>

<template>
  <div
    class="product-thumbnail-grid"
    :class="{ 'product-thumbnail-grid--fixed-columns': columns }"
    role="list"
    :aria-label="ariaLabel"
    :style="gridStyle"
  >
    <div
      v-for="item in items"
      :key="item.id"
      class="product-thumbnail-grid__slot"
      role="listitem"
    >
      <UiButton
        v-if="item.interactive !== false"
        variant="toolbar"
        class="product-thumbnail-grid__item"
        :class="{
          'product-thumbnail-grid__item--selected': item.selected,
          'product-thumbnail-grid__item--marked': item.marked,
        }"
        :data-product-thumbnail-id="item.id"
        :aria-label="itemAriaLabel(item)"
        :aria-pressed="item.selected ? 'true' : 'false'"
        :disabled="item.disabled"
        :title="item.disabledTitle"
        @click="selectItem(item)"
      >
        <span class="product-thumbnail-grid__preview">
          <img
            v-if="item.src && !hasImageFailed(item)"
            class="product-thumbnail-grid__preview-image"
            :src="item.src"
            :alt="item.alt"
            loading="lazy"
            @error="markImageFailed(item)"
          >
          <span v-else class="product-thumbnail-grid__fallback" aria-hidden="true">
            <span v-if="item.fallbackLabel" class="product-thumbnail-grid__fallback-label">{{ item.fallbackLabel }}</span>
            <UiIcon v-else name="image" size="18" />
          </span>
        </span>
        <span v-if="item.selectedBadge" class="product-thumbnail-grid__selected-badge">
          {{ item.selectedBadge }}
        </span>
        <span v-if="item.cornerLabel" class="product-thumbnail-grid__corner-badge">
          {{ item.cornerLabel }}
        </span>
        <span
          v-if="item.disabled"
          class="product-thumbnail-grid__disabled-overlay"
          aria-hidden="true"
        ></span>
        <span class="product-thumbnail-grid__label">{{ item.label }}</span>
      </UiButton>
      <div
        v-else
        class="product-thumbnail-grid__item product-thumbnail-grid__item--static"
        :class="{
          'product-thumbnail-grid__item--selected': item.selected,
          'product-thumbnail-grid__item--marked': item.marked,
        }"
        :data-product-thumbnail-id="item.id"
        role="img"
        :aria-label="item.alt"
      >
        <span class="product-thumbnail-grid__preview">
          <img
            v-if="item.src && !hasImageFailed(item)"
            class="product-thumbnail-grid__preview-image"
            :src="item.src"
            :alt="item.alt"
            loading="lazy"
            @error="markImageFailed(item)"
          >
          <span v-else class="product-thumbnail-grid__fallback" aria-hidden="true">
            <span v-if="item.fallbackLabel" class="product-thumbnail-grid__fallback-label">{{ item.fallbackLabel }}</span>
            <UiIcon v-else name="image" size="18" />
          </span>
        </span>
        <span v-if="item.selectedBadge" class="product-thumbnail-grid__selected-badge">
          {{ item.selectedBadge }}
        </span>
        <span v-if="item.cornerLabel" class="product-thumbnail-grid__corner-badge">
          {{ item.cornerLabel }}
        </span>
        <span class="product-thumbnail-grid__label">{{ item.label }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.product-thumbnail-grid {
  --product-thumbnail-grid-column-count: 4;
  --product-thumbnail-grid-min-size: 64px;
  --product-thumbnail-grid-aspect-ratio: 3 / 4;

  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(var(--product-thumbnail-grid-min-size), 1fr));
  gap: 6px;
}

.product-thumbnail-grid--fixed-columns {
  grid-template-columns: repeat(var(--product-thumbnail-grid-column-count), minmax(0, 1fr));
}

.product-thumbnail-grid__slot {
  min-width: 0;
}

.product-thumbnail-grid__item {
  position: relative;
  display: block;
  width: 100%;
  overflow: hidden;
  aspect-ratio: var(--product-thumbnail-grid-aspect-ratio);
  border: 2px solid transparent;
  border-radius: 6px;
  background: var(--color-surface-subtle);
  color: inherit;
  text-align: left;
  transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
}

.product-thumbnail-grid__item:not(.product-thumbnail-grid__item--static):hover {
  border-color: var(--color-action-primary-soft);
  transform: scale(1.02);
}

.product-thumbnail-grid__item--static {
  cursor: default;
}

.product-thumbnail-grid__item--selected {
  border-color: var(--color-action-primary);
  box-shadow: 0 0 0 2px var(--color-focus-brand-subtle);
}

.product-thumbnail-grid__item--marked::after {
  content: '';
  position: absolute;
  top: 3px;
  right: 3px;
  width: 12px;
  height: 12px;
  border: 1.5px solid var(--color-surface-base);
  border-radius: 50%;
  background: var(--color-status-success);
}

.product-thumbnail-grid__preview,
.product-thumbnail-grid__preview-image,
.product-thumbnail-grid__fallback {
  position: absolute;
  inset: 0;
}

.product-thumbnail-grid__preview-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: center;
  background: var(--color-surface-subtle);
}

.product-thumbnail-grid__fallback {
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text-muted);
  background: var(--color-surface-muted);
}

.product-thumbnail-grid__fallback-label {
  padding: 0 6px;
  color: var(--color-text-supporting);
  font-size: 12px;
  font-weight: 600;
  text-align: center;
}

.product-thumbnail-grid__selected-badge,
.product-thumbnail-grid__corner-badge {
  position: absolute;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text-inverse);
  font-weight: 700;
  line-height: 1;
}

.product-thumbnail-grid__selected-badge {
  top: 6px;
  left: 6px;
  min-width: 24px;
  height: 24px;
  padding: 0 6px;
  border-radius: 999px;
  background: var(--color-action-primary);
  box-shadow: 0 2px 6px var(--shadow-medium);
  font-size: 12px;
}

.product-thumbnail-grid__corner-badge {
  top: 6px;
  right: 6px;
  padding: 3px 6px;
  border-radius: 999px;
  background: var(--color-status-info);
  font-size: 10px;
}

.product-thumbnail-grid__disabled-overlay {
  position: absolute;
  inset: 0;
  background: var(--color-surface-raised);
  cursor: not-allowed;
}

.product-thumbnail-grid__label {
  position: absolute;
  right: 0;
  bottom: 0;
  left: 0;
  padding: 2px 4px;
  background: linear-gradient(transparent, var(--color-overlay-backdrop-strong));
  color: var(--color-text-inverse);
  font-size: 10px;
  text-align: center;
}
</style>
