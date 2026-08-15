<script setup lang="ts">
import { computed } from 'vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductSearchField from '@/components/product/ProductSearchField.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import type { TagData } from '@/types/api'

const props = defineProps<{
  availableTags: TagData[]
  filter: string
  showCreateNewTagOption: boolean
}>()

const emit = defineEmits<{
  (event: 'update:filter', value: string): void
  (event: 'add', tagName: string): void
  (event: 'submit'): void
}>()

const availableTagItems = computed<ProductChipItem[]>(() => props.availableTags.map(tag => ({
  id: tag.name,
  label: tag.name,
  ariaLabel: `添加标签 ${tag.name}`,
  iconName: 'plus',
  interactive: true,
  tone: 'custom',
  backgroundColor: tag.color,
  borderColor: tag.color,
  textColor: 'var(--color-text-inverse)',
})))

function addExistingTag(id: string | number): void {
  emit('add', String(id))
}
</script>

<template>
  <div class="quick-tag-picker__input-wrapper">
    <ProductSearchField
      :model-value="filter"
      placeholder="输入标签名称进行搜索或创建..."
      aria-label="搜索或创建标签"
      :show-icon="false"
      autofocus
      @update:model-value="$emit('update:filter', String($event))"
      @search="$emit('submit')"
      @clear="$emit('update:filter', '')"
    />
  </div>

  <div class="quick-tag-picker__list">
    <ProductChipList
      v-if="availableTagItems.length"
      aria-label="可添加标签"
      :items="availableTagItems"
      @select="addExistingTag"
    />

    <ProductRecordCard
      v-if="showCreateNewTagOption"
      as="button"
      class="quick-tag-picker__item quick-tag-picker__item--new"
      :aria-label="`创建并添加标签 ${filter.trim()}`"
      @click="$emit('add', filter.trim())"
    >
      <span class="quick-tag-picker__content">
        <UiIcon name="plus" size="16" class="quick-tag-picker__icon" />
        <span>创建并添加 "{{ filter.trim() }}"</span>
      </span>
    </ProductRecordCard>

    <ProductStatusBanner
      v-if="availableTags.length === 0 && !showCreateNewTagOption"
      class="quick-tag-picker__empty-state"
      tone="neutral"
      icon-name="tags"
      role="note"
    >
      {{ filter ? '未找到匹配的标签' : '所有标签已添加或暂无标签' }}
    </ProductStatusBanner>
  </div>
</template>

<style scoped>
.quick-tag-picker__input-wrapper {
  --product-search-field-input-padding: 12px 16px;
  --product-search-field-radius: 8px;
  --ui-input-min-height: 44px;

  margin-bottom: 16px;
}

.quick-tag-picker__list {
  --quick-tag-picker-new-background-start: var(--color-focus-brand-soft);
  --quick-tag-picker-new-background-end: color-mix(in srgb, var(--color-action-brand-strong) 10%, transparent);
  --quick-tag-picker-new-background-hover-start: var(--color-focus-brand-subtle);
  --quick-tag-picker-new-background-hover-end: color-mix(in srgb, var(--color-action-brand-strong) 18%, transparent);
  --quick-tag-picker-new-border: var(--shadow-action-brand);
  --quick-tag-picker-new-border-hover: color-mix(in srgb, var(--color-action-brand) 60%, transparent);

  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 260px;
  overflow-y: auto;
}

.quick-tag-picker__item {
  --product-record-card-background: var(--color-surface-interactive-hover);
  --product-record-card-border: transparent;
  --product-record-card-accent: var(--color-border-brand-gradient);
  --product-record-card-padding: 12px 16px;
  --product-record-card-shadow-hover: none;

  color: inherit;
}

.quick-tag-picker__content {
  display: flex;
  align-items: center;
  width: 100%;
  gap: 12px;
}

.quick-tag-picker__item--new {
  --product-record-card-background: linear-gradient(135deg, var(--quick-tag-picker-new-background-start) 0%, var(--quick-tag-picker-new-background-end) 100%);
  --product-record-card-border: var(--quick-tag-picker-new-border);
  --product-record-card-accent: var(--quick-tag-picker-new-border-hover);
}

.quick-tag-picker__item--new:hover {
  --product-record-card-background: linear-gradient(135deg, var(--quick-tag-picker-new-background-hover-start) 0%, var(--quick-tag-picker-new-background-hover-end) 100%);
  --product-record-card-border: var(--quick-tag-picker-new-border-hover);
}

.quick-tag-picker__icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  color: var(--color-action-brand);
}

.quick-tag-picker__empty-state {
  --product-status-banner-align-items: center;
  --product-status-banner-justify-content: center;
  --product-status-banner-padding: 24px 16px;
  --product-status-banner-border: 0;
  --product-status-banner-background: transparent;
  --product-status-banner-icon-display: none;
  --product-status-banner-body-color: var(--color-text-supporting);
  --product-status-banner-text-align: center;

  color: var(--color-text-supporting);
  font-style: italic;
}
</style>
