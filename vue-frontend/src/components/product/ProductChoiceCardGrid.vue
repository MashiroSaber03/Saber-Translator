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

const props = withDefaults(defineProps<{
  accessibilityLabel: string
  items: ProductChoiceCardItem[]
  modelValue: string
  variant?: 'default' | 'compact'
}>(), {
  variant: 'default',
})

const emit = defineEmits<{
  'update:modelValue': [id: string]
  select: [id: string]
}>()

function selectItem(item: ProductChoiceCardItem): void {
  if (item.disabled) return
  emit('update:modelValue', item.id)
  emit('select', item.id)
}

function choiceTabIndex(item: ProductChoiceCardItem): number {
  if (item.disabled) return -1
  const enabledItems = props.items.filter(candidate => !candidate.disabled)
  const selectedItem = enabledItems.find(candidate => candidate.id === props.modelValue)
  return item.id === (selectedItem ?? enabledItems[0])?.id ? 0 : -1
}

function selectAdjacentItem(event: KeyboardEvent, item: ProductChoiceCardItem): void {
  const enabledItems = props.items.filter(candidate => !candidate.disabled)
  if (enabledItems.length === 0) return
  const currentIndex = Math.max(
    enabledItems.findIndex(candidate => candidate.id === item.id),
    enabledItems.findIndex(candidate => candidate.id === props.modelValue),
    0,
  )
  let nextIndex: number | null = null

  if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
    nextIndex = (currentIndex + 1) % enabledItems.length
  } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
    nextIndex = (currentIndex - 1 + enabledItems.length) % enabledItems.length
  } else if (event.key === 'Home') {
    nextIndex = 0
  } else if (event.key === 'End') {
    nextIndex = enabledItems.length - 1
  }

  if (nextIndex === null) return
  event.preventDefault()
  const nextItem = enabledItems[nextIndex]!
  selectItem(nextItem)
  const group = (event.currentTarget as HTMLElement).closest('[role="radiogroup"]')
  const itemIndex = props.items.findIndex(candidate => candidate.id === nextItem.id)
  group?.querySelectorAll<HTMLElement>('[role="radio"]')[itemIndex]?.focus()
}
</script>

<template>
  <div
    class="product-choice-card-grid"
    :class="`product-choice-card-grid--${variant}`"
    role="radiogroup"
    :aria-label="accessibilityLabel"
  >
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
      :tabindex="choiceTabIndex(item)"
      @click="selectItem(item)"
      @keydown="selectAdjacentItem($event, item)"
    >
      <UiIcon
        v-if="item.iconName"
        class="product-choice-card-grid__icon"
        :name="item.iconName"
        size="42"
        stroke-width="1.5"
      />
      <span class="product-choice-card-grid__heading">
        <span class="product-choice-card-grid__label">{{ item.label }}</span>
        <span
          v-if="variant === 'compact' && item.id === modelValue"
          class="product-choice-card-grid__check"
          aria-hidden="true"
        >✓</span>
      </span>
      <span v-if="item.description" class="product-choice-card-grid__description">{{ item.description }}</span>
    </UiButton>
  </div>
</template>

<style scoped>
.product-choice-card-grid {
  display: grid;
  grid-template-columns: var(--product-choice-card-grid-columns, repeat(auto-fit, minmax(180px, 1fr)));
  gap: var(--product-choice-card-grid-gap, 16px);
}

.product-choice-card-grid__item {
  display: flex;
  flex-direction: column;
  align-items: var(--product-choice-card-grid-item-align-items, center);
  justify-content: var(--product-choice-card-grid-item-justify-content, center);
  gap: 8px;
  width: 100%;
  min-height: var(--product-choice-card-grid-item-min-height, 150px);
  padding: var(--product-choice-card-grid-item-padding, 22px);
  border: var(--product-choice-card-grid-item-border-width, 2px) solid var(--product-choice-card-grid-item-border, var(--color-border-muted));
  border-radius: var(--product-choice-card-grid-item-radius, 8px);
  background: var(--product-choice-card-grid-item-background, var(--color-surface-base));
  text-align: var(--product-choice-card-grid-item-text-align, center);
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
  box-shadow: var(--product-choice-card-grid-item-shadow-selected, none);
}

.product-choice-card-grid__item--disabled {
  opacity: 0.6;
  transform: none;
}

.product-choice-card-grid__icon {
  color: var(--color-text-brand);
}

.product-choice-card-grid__heading {
  display: flex;
  width: var(--product-choice-card-grid-heading-width, auto);
  align-items: center;
  justify-content: var(--product-choice-card-grid-heading-justify-content, center);
  gap: 10px;
}

.product-choice-card-grid__label {
  border-radius: var(--product-choice-card-grid-label-radius, 0);
  padding: var(--product-choice-card-grid-label-padding, 0);
  background: var(--product-choice-card-grid-label-background, transparent);
  color: var(--product-choice-card-grid-label-color, inherit);
  font-size: var(--product-choice-card-grid-label-font-size, 16px);
  font-weight: var(--product-choice-card-grid-label-font-weight, 600);
}

.product-choice-card-grid__check {
  color: var(--product-choice-card-grid-check-color, var(--color-text-link-strong));
  font-weight: 700;
}

.product-choice-card-grid__description {
  margin-top: var(--product-choice-card-grid-description-margin-top, 0);
  color: var(--product-choice-card-grid-description-color, var(--color-text-supporting));
  font-size: var(--product-choice-card-grid-description-font-size, 14px);
  line-height: var(--product-choice-card-grid-description-line-height, 1.4);
  white-space: var(--product-choice-card-grid-description-white-space, normal);
}

@media (--breakpoint-md-down) {
  .product-choice-card-grid--compact {
    grid-template-columns: 1fr;
  }
}
</style>
