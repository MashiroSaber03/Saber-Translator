<script setup lang="ts">
import { useId } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'

export type ProductLogTone = 'neutral' | 'info' | 'success' | 'warning' | 'danger' | 'accent'

export interface ProductLogItem {
  detail?: string
  id?: string | number
  message: string
  timestamp?: string
  tone?: ProductLogTone
}

withDefaults(defineProps<{
  activeHint?: string
  ariaLabel?: string
  emptyText?: string
  expanded: boolean
  items: ProductLogItem[]
  title: string
}>(), {
  activeHint: '',
  ariaLabel: '日志',
  emptyText: '暂无日志。',
})

defineEmits<{
  toggle: []
}>()

const contentId = useId()
</script>

<template>
  <section class="product-log-panel">
    <UiButton
      variant="secondary"
      block
      size="sm"
      type="button"
      class="product-log-panel__header"
      :class="{ 'product-log-panel__header--expanded': expanded }"
      :aria-expanded="expanded ? 'true' : 'false'"
      :aria-controls="contentId"
      @click="$emit('toggle')"
    >
      <UiIcon class="product-log-panel__toggle" :name="expanded ? 'chevron-down' : 'chevron-right'" size="14" />
      <span>{{ title }}</span>
      <span v-if="activeHint" class="product-log-panel__hint">{{ activeHint }}</span>
    </UiButton>
    <div v-if="expanded" :id="contentId" class="product-log-panel__content" role="log" :aria-label="ariaLabel">
      <div
        v-for="(item, index) in items"
        :key="item.id ?? index"
        class="product-log-panel__item"
        :class="`product-log-panel__item--${item.tone ?? 'neutral'}`"
      >
        <span v-if="item.timestamp" class="product-log-panel__time">[{{ item.timestamp }}]</span>
        <span class="product-log-panel__message">{{ item.message }}</span>
        <pre v-if="item.detail" class="product-log-panel__detail">{{ item.detail }}</pre>
      </div>
      <div v-if="items.length === 0" class="product-log-panel__empty">{{ emptyText }}</div>
    </div>
  </section>
</template>

<style scoped>
.product-log-panel {
  margin-bottom: 16px;
}

.product-log-panel__header {
  width: 100%;
  justify-content: flex-start;
  text-align: left;
}

.product-log-panel__header--expanded {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}

.product-log-panel__toggle {
  color: var(--color-text-supporting);
}

.product-log-panel__hint {
  margin-left: auto;
  color: var(--color-action-primary);
  font-weight: 400;
  font-size: 13px;
}

.product-log-panel__content {
  max-height: 200px;
  padding: 12px;
  overflow-y: auto;
  border: 1px solid var(--color-border-muted);
  border-top: 0;
  border-radius: 0 0 8px 8px;
  background: var(--color-overlay-backdrop-solid);
  font-family: var(--font-mono);
  font-size: 12px;
}

.product-log-panel__item {
  padding: 2px 0;
  color: var(--color-text-inverse);
}

.product-log-panel__time {
  margin-right: 8px;
  color: var(--color-text-subtle);
}

.product-log-panel__message,
.product-log-panel__detail {
  white-space: pre-wrap;
  word-break: break-word;
}

.product-log-panel__detail {
  margin: 6px 0 0;
  color: var(--color-text-inverse);
  font: inherit;
}

.product-log-panel__item--info .product-log-panel__message {
  color: var(--color-status-info);
}

.product-log-panel__item--warning .product-log-panel__message {
  color: var(--color-status-warning);
}

.product-log-panel__item--success .product-log-panel__message {
  color: var(--color-status-success);
}

.product-log-panel__item--accent .product-log-panel__message {
  color: var(--color-action-brand);
}

.product-log-panel__item--danger .product-log-panel__message {
  color: var(--color-status-error);
}

.product-log-panel__empty {
  color: var(--color-text-inverse);
}
</style>
