<script setup lang="ts">
import { computed } from 'vue'

type ProductRecordCardAs = 'article' | 'button'

const props = withDefaults(defineProps<{
  as?: ProductRecordCardAs
  accent?: boolean
  ariaLabel?: string
  disabled?: boolean
  type?: 'button' | 'submit' | 'reset'
}>(), {
  as: 'article',
  accent: false,
  ariaLabel: undefined,
  disabled: false,
  type: 'button',
})

const emit = defineEmits<{
  click: [event: MouseEvent]
}>()

const cardAttrs = computed(() => {
  if (props.as === 'button') {
    return {
      disabled: props.disabled,
      type: props.type,
    }
  }

  return {}
})

function handleClick(event: MouseEvent): void {
  if (props.disabled) {
    event.preventDefault()
    event.stopPropagation()
    return
  }

  emit('click', event)
}
</script>

<template>
  <component
    :is="as"
    class="product-record-card"
    :class="[
      { 'product-record-card--accent': accent },
      { 'product-record-card--button': as === 'button' },
      { 'product-record-card--disabled': disabled },
    ]"
    v-bind="cardAttrs"
    :aria-label="ariaLabel"
    @click="handleClick"
  >
    <header
      v-if="$slots.icon || $slots.meta || $slots.actions"
      class="product-record-card__header"
    >
      <div v-if="$slots.icon" class="product-record-card__icon">
        <slot name="icon" />
      </div>

      <div v-if="$slots.meta" class="product-record-card__meta">
        <slot name="meta" />
      </div>

      <div v-if="$slots.actions" class="product-record-card__actions">
        <slot name="actions" />
      </div>
    </header>

    <div class="product-record-card__body">
      <slot />
    </div>

    <footer v-if="$slots.footer" class="product-record-card__footer">
      <slot name="footer" />
    </footer>
  </component>
</template>

<style scoped>
.product-record-card {
  --product-record-card-background: var(--color-surface-muted);
  --product-record-card-border: var(--color-border-muted);
  --product-record-card-accent: var(--color-action-primary);
  --product-record-card-shadow-hover: 0 2px 8px var(--color-focus-brand-soft);
  --product-record-card-shadow: none;
  --product-record-card-radius: 8px;
  --product-record-card-padding: 12px;
  --product-record-card-gap: 8px;

  display: flex;
  flex-direction: column;
  gap: var(--product-record-card-gap);
  padding: var(--product-record-card-padding);
  border: 1px solid var(--product-record-card-border);
  border-radius: var(--product-record-card-radius);
  background: var(--product-record-card-background);
  box-shadow: var(--product-record-card-shadow);
  color: inherit;
  font: inherit;
  text-align: left;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.product-record-card:hover {
  border-color: var(--product-record-card-accent);
  box-shadow: var(--product-record-card-shadow-hover);
}

.product-record-card--accent {
  border-left: 3px solid var(--product-record-card-accent);
}

.product-record-card--button {
  width: 100%;
  cursor: pointer;
}

.product-record-card--disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.product-record-card__header {
  display: flex;
  align-items: flex-start;
  gap: var(--product-record-card-gap);
  min-width: 0;
}

.product-record-card__icon {
  display: inline-flex;
  flex: 0 0 auto;
  color: var(--color-text-supporting);
}

.product-record-card__meta {
  flex: 1 1 auto;
  min-width: 0;
  color: var(--color-text-supporting);
  font-size: 12px;
}

.product-record-card__actions {
  display: inline-flex;
  flex: 0 0 auto;
  gap: 4px;
  margin-left: auto;
}

.product-record-card__body,
.product-record-card__footer {
  min-width: 0;
}
</style>
