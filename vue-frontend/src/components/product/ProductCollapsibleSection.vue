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
  textToggle?: boolean
  title: string
}>(), {
  ariaLabel: undefined,
  hint: '',
  iconName: undefined,
  textToggle: false,
})

const emit = defineEmits<{
  'update:expanded': [value: boolean]
  toggle: [value: boolean]
}>()

const toggleIconName = computed<UiIconName>(() => props.expanded ? 'chevron-down' : 'chevron-right')
const toggleText = computed(() => props.expanded ? '▼' : '▶')
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
      <span
        v-if="textToggle"
        class="product-collapsible-section__toggle product-collapsible-section__toggle-text"
        aria-hidden="true"
      >{{ toggleText }}</span>
      <UiIcon v-else class="product-collapsible-section__toggle" :name="toggleIconName" size="14" />
      <span class="product-collapsible-section__title">
        <span v-if="$slots.icon" class="product-collapsible-section__title-icon-text" aria-hidden="true"><slot name="icon" /></span>
        <UiIcon v-else-if="iconName" :name="iconName" size="16" />
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
  border: var(--product-collapsible-section-border, 1px solid var(--color-border-muted, var(--color-border-soft)));
  border-radius: var(--product-collapsible-section-radius, 8px);
  background: var(--product-collapsible-section-background, var(--color-surface-base));
}

.product-collapsible-section__header {
  display: flex;
  align-items: center;
  gap: var(--product-collapsible-section-header-gap, 8px);
  width: 100%;
  margin: var(--product-collapsible-section-header-margin, 0);
  padding: var(--product-collapsible-section-header-padding, 12px 14px);
  border: var(--product-collapsible-section-header-border, 0);
  border-bottom: var(--product-collapsible-section-header-border-bottom, 0);
  background: var(--product-collapsible-section-header-background, var(--color-surface-quiet));
  color: inherit;
  font: inherit;
  text-align: left;
  user-select: none;
  transition: background 0.2s ease;
}

.product-collapsible-section__header:hover {
  background: var(--product-collapsible-section-header-hover-background, var(--color-surface-hover));
}

.product-collapsible-section__toggle {
  order: var(--product-collapsible-section-toggle-order, initial);
  flex: 0 0 auto;
  margin-left: var(--product-collapsible-section-toggle-margin-left, 0);
  color: var(--product-collapsible-section-toggle-color, var(--color-text-supporting, var(--color-text-subtle)));
}

.product-collapsible-section__toggle-text {
  font-size: 10px;
  line-height: 1;
}

.product-collapsible-section__title-icon-text {
  font-size: 1em;
  line-height: 1;
}

.product-collapsible-section__title {
  order: var(--product-collapsible-section-title-order, initial);
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  color: var(--product-collapsible-section-title-color, var(--color-text-default));
  font-size: var(--product-collapsible-section-title-font-size, 14px);
  font-weight: var(--product-collapsible-section-title-font-weight, 600);
}

.product-collapsible-section__hint {
  order: var(--product-collapsible-section-hint-order, initial);
  margin-left: var(--product-collapsible-section-hint-margin-left, auto);
  color: var(--color-text-supporting, var(--color-text-muted));
  font-size: 12px;
  white-space: nowrap;
}

.product-collapsible-section__body {
  padding: var(--product-collapsible-section-body-padding, 16px);
  background: var(--product-collapsible-section-body-background, var(--color-surface-base));
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
