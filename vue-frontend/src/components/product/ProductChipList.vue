<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import type { UiIconName } from '@/components/ui/iconRegistry'

export interface ProductChipItem {
  ariaLabel?: string
  backgroundColor?: string
  borderColor?: string
  disabled?: boolean
  iconName?: UiIconName
  id: string | number
  interactive?: boolean
  label: string
  selected?: boolean
  textColor?: string
  tone?: 'neutral' | 'primary' | 'success' | 'warning' | 'danger' | 'inverse' | 'custom'
}

withDefaults(defineProps<{
  ariaLabel?: string
  items: ProductChipItem[]
  label?: string
  labelIconName?: UiIconName
}>(), {
  ariaLabel: undefined,
  label: '',
  labelIconName: undefined,
})

const emit = defineEmits<{
  select: [id: string | number]
}>()

function chipToneClass(item: ProductChipItem): string {
  return `product-chip-list__chip--${item.tone ?? 'neutral'}`
}

function chipStyle(item: ProductChipItem): Record<string, string> | undefined {
  if (item.tone !== 'custom') return undefined

  return {
    '--product-chip-list-custom-background': item.backgroundColor ?? 'var(--color-action-primary)',
    '--product-chip-list-custom-border': item.borderColor ?? 'transparent',
    '--product-chip-list-custom-text': item.textColor ?? 'var(--color-text-inverse)',
  }
}

function selectItem(item: ProductChipItem): void {
  if (!item.interactive || item.disabled) return
  emit('select', item.id)
}

function chipPressed(item: ProductChipItem): string | undefined {
  if (item.selected === undefined) return undefined
  return item.selected ? 'true' : 'false'
}
</script>

<template>
  <div class="product-chip-list" role="list" :aria-label="ariaLabel">
    <span v-if="label" class="product-chip-list__label">
      <UiIcon v-if="labelIconName" :name="labelIconName" size="14" />
      <span>{{ label }}</span>
    </span>

    <span
      v-for="item in items"
      :key="item.id"
      class="product-chip-list__slot"
      role="listitem"
    >
      <UiButton
        v-if="item.interactive"
        variant="toolbar"
        class="product-chip-list__chip product-chip-list__chip--interactive"
        :class="[
          chipToneClass(item),
          { 'product-chip-list__chip--selected': item.selected },
        ]"
        :style="chipStyle(item)"
        :aria-label="item.ariaLabel"
        :aria-pressed="chipPressed(item)"
        :disabled="item.disabled"
        @click="selectItem(item)"
      >
        <UiIcon v-if="item.iconName" :name="item.iconName" size="13" />
        <span>{{ item.label }}</span>
      </UiButton>
      <span
        v-else
        class="product-chip-list__chip"
        :class="[
          chipToneClass(item),
          { 'product-chip-list__chip--selected': item.selected },
        ]"
        :style="chipStyle(item)"
      >
        <UiIcon v-if="item.iconName" :name="item.iconName" size="13" />
        <span>{{ item.label }}</span>
      </span>
    </span>
  </div>
</template>

<style scoped>
.product-chip-list {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  color: var(--product-chip-list-text, var(--color-text-supporting));
  font-size: 12px;
}

.product-chip-list__label {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: var(--product-chip-list-label-text, var(--product-chip-list-text, var(--color-text-supporting)));
  font-weight: 500;
}

.product-chip-list__slot {
  display: inline-flex;
}

.product-chip-list__chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 9px;
  border: 1px solid transparent;
  border-radius: 999px;
  font: inherit;
  font-weight: var(--product-chip-list-chip-font-weight, inherit);
  line-height: 1.35;
  transition: border-color 0.2s ease, background 0.2s ease, color 0.2s ease;
}

.product-chip-list__chip--interactive {
  cursor: pointer;
}

.product-chip-list__chip--neutral {
  border-color: var(--product-chip-list-neutral-border, var(--color-border-muted));
  background: var(--product-chip-list-neutral-background, var(--color-surface-muted));
  color: var(--product-chip-list-neutral-text, var(--color-text-supporting));
}

.product-chip-list__chip--primary {
  border-color: var(--product-chip-list-primary-border, transparent);
  background: var(--product-chip-list-primary-background, var(--color-action-primary));
  color: var(--product-chip-list-primary-text, var(--color-text-inverse));
}

.product-chip-list__chip--success {
  background: var(--color-status-success);
  color: var(--color-text-inverse);
}

.product-chip-list__chip--warning {
  background: var(--color-status-warning-surface-soft);
  color: var(--color-text-default);
}

.product-chip-list__chip--danger {
  background: var(--color-surface-danger-soft);
  color: var(--color-status-error);
}

.product-chip-list__chip--inverse {
  border-color: var(--product-chip-list-inverse-border, var(--color-overlay-inverse-emphasis));
  background: var(--product-chip-list-inverse-background, var(--color-overlay-inverse-muted));
  color: var(--product-chip-list-inverse-text, var(--color-text-inverse));
}

.product-chip-list__chip--custom {
  border-color: var(--product-chip-list-custom-border);
  background: var(--product-chip-list-custom-background);
  color: var(--product-chip-list-custom-text);
}

.product-chip-list__chip--interactive:hover {
  border-color: var(--color-action-primary);
  background: var(--color-action-primary);
  color: var(--color-text-inverse);
}

.product-chip-list__chip--interactive.product-chip-list__chip--selected {
  border-color: var(--product-chip-list-custom-border, var(--color-action-primary));
  background: var(--product-chip-list-custom-background, var(--color-action-primary));
  color: var(--product-chip-list-custom-text, var(--color-text-inverse));
}

.product-chip-list__chip--interactive.product-chip-list__chip--selected:hover {
  border-color: var(--product-chip-list-custom-border, var(--color-action-primary));
  background: var(--product-chip-list-custom-background, var(--color-action-primary));
  color: var(--product-chip-list-custom-text, var(--color-text-inverse));
}
</style>
