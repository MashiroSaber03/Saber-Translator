<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import type { UiIconName } from '@/components/ui/iconRegistry'

export type ProductChoiceCardItem = {
  id: string
  label: string
  description?: string
  iconName?: UiIconName
  disabled?: boolean
}

withDefaults(defineProps<{
  ariaLabel: string
  items: ProductChoiceCardItem[]
  modelValue: string
}>(), {})

const emit = defineEmits<{
  'update:modelValue': [id: string]
  select: [id: string]
}>()

function selectItem(item: ProductChoiceCardItem): void {
  if (item.disabled) return
  emit('update:modelValue', item.id)
  emit('select', item.id)
}
</script>

<template>
  <div class="product-choice-card-grid" role="radiogroup" :aria-label="ariaLabel">
    <UiButton
      v-for="item in items"
      :key="item.id"
      variant="toolbar"
      class="product-choice-card-grid__item"
      :class="{
        'product-choice-card-grid__item--selected': item.id === modelValue,
        'product-choice-card-grid__item--disabled': item.disabled,
      }"
      role="radio"
      :aria-checked="item.id === modelValue ? 'true' : 'false'"
      :disabled="item.disabled"
      @click="selectItem(item)"
    >
      <UiIcon
        v-if="item.iconName"
        class="product-choice-card-grid__icon"
        :name="item.iconName"
        size="42"
        stroke-width="1.5"
      />
      <span class="product-choice-card-grid__label">{{ item.label }}</span>
      <span v-if="item.description" class="product-choice-card-grid__description">{{ item.description }}</span>
    </UiButton>
  </div>
</template>

<style scoped>
.product-choice-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: var(--product-choice-card-grid-gap, 16px);
}

.product-choice-card-grid__item {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  width: 100%;
  min-height: 150px;
  padding: var(--product-choice-card-grid-item-padding, 22px);
  border: 2px solid var(--product-choice-card-grid-item-border, var(--color-border-muted));
  border-radius: var(--product-choice-card-grid-item-radius, 8px);
  background: var(--product-choice-card-grid-item-background, var(--color-surface-base));
  text-align: center;
  transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}

.product-choice-card-grid__item:hover {
  border-color: var(--product-choice-card-grid-item-border-selected, var(--color-border-brand));
  box-shadow: var(--product-choice-card-grid-item-shadow-hover, 0 4px 12px var(--color-focus-brand-soft));
  transform: translateY(-2px);
}

.product-choice-card-grid__item--selected,
.product-choice-card-grid__item--selected:hover {
  border-color: var(--product-choice-card-grid-item-border-selected, var(--color-border-brand));
  background: var(--product-choice-card-grid-item-background-selected, var(--color-focus-brand-soft));
}

.product-choice-card-grid__item--disabled {
  opacity: 0.6;
  transform: none;
}

.product-choice-card-grid__icon {
  color: var(--color-text-brand);
}

.product-choice-card-grid__label {
  font-size: 16px;
  font-weight: 600;
}

.product-choice-card-grid__description {
  color: var(--color-text-supporting);
  font-size: 14px;
  line-height: 1.4;
}
</style>
