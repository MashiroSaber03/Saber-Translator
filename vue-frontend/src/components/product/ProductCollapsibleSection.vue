<script setup lang="ts">
import { computed, useId } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import type { UiIconName } from '@/components/ui/iconRegistry'

const props = withDefaults(defineProps<{
  ariaLabel?: string
  expanded: boolean
  hint?: string
  iconName?: UiIconName
  title: string
}>(), {
  ariaLabel: undefined,
  hint: '',
  iconName: undefined,
})

const emit = defineEmits<{
  'update:expanded': [value: boolean]
  toggle: [value: boolean]
}>()

const toggleIconName = computed<UiIconName>(() => props.expanded ? 'chevron-down' : 'chevron-right')
const bodyId = useId()

function toggle(): void {
  const nextExpanded = !props.expanded
  emit('update:expanded', nextExpanded)
  emit('toggle', nextExpanded)
}
</script>

<template>
  <section class="product-collapsible-section">
    <UiButton
      variant="toolbar"
      type="button"
      class="product-collapsible-section__header"
      :aria-expanded="expanded ? 'true' : 'false'"
      :aria-label="ariaLabel"
      :aria-controls="bodyId"
      @click="toggle"
    >
      <UiIcon class="product-collapsible-section__toggle" :name="toggleIconName" size="14" />
      <span class="product-collapsible-section__title">
        <UiIcon v-if="iconName" :name="iconName" size="16" />
        <slot name="title">{{ title }}</slot>
      </span>
      <span v-if="hint" class="product-collapsible-section__hint">{{ hint }}</span>
    </UiButton>

    <div v-if="expanded" :id="bodyId" class="product-collapsible-section__body">
      <slot />
    </div>
  </section>
</template>

<style scoped>
.product-collapsible-section {
  overflow: hidden;
  border: 1px solid var(--color-border-muted, var(--color-border-soft));
  border-radius: 8px;
  background: var(--color-surface-base);
}

.product-collapsible-section__header {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 12px 14px;
  border: 0;
  background: var(--color-surface-quiet);
  color: inherit;
  font: inherit;
  text-align: left;
  user-select: none;
  transition: background 0.2s ease;
}

.product-collapsible-section__header:hover {
  background: var(--color-surface-hover);
}

.product-collapsible-section__toggle {
  flex: 0 0 auto;
  color: var(--color-text-supporting, var(--color-text-subtle));
}

.product-collapsible-section__title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  color: var(--color-text-default);
  font-size: 14px;
  font-weight: 600;
}

.product-collapsible-section__hint {
  margin-left: auto;
  color: var(--color-text-supporting, var(--color-text-muted));
  font-size: 12px;
  white-space: nowrap;
}

.product-collapsible-section__body {
  padding: 16px;
  background: var(--color-surface-base);
}

@media (--breakpoint-sm-down) {
  .product-collapsible-section__header {
    align-items: flex-start;
  }

  .product-collapsible-section__hint {
    white-space: normal;
  }
}
</style>
