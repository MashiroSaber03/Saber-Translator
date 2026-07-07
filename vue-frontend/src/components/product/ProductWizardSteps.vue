<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'

export type ProductWizardStep = {
  label: string
  disabled?: boolean
}

const props = withDefaults(defineProps<{
  steps: ProductWizardStep[]
  activeIndex: number
  ariaLabel?: string
}>(), {
  ariaLabel: undefined,
})

const emit = defineEmits<{
  'update:activeIndex': [index: number]
  select: [index: number]
}>()

function isActive(index: number): boolean {
  return index === props.activeIndex
}

function isCompleted(index: number): boolean {
  return index < props.activeIndex
}

function selectStep(index: number): void {
  const step = props.steps[index]
  if (!step || step.disabled) return
  emit('update:activeIndex', index)
  emit('select', index)
}
</script>

<template>
  <nav class="product-wizard-steps" :aria-label="ariaLabel">
    <UiButton
      v-for="(step, index) in props.steps"
      :key="`${index}-${step.label}`"
      type="button"
      variant="toolbar"
      class="product-wizard-steps__step"
      :class="{
        'product-wizard-steps__step--active': isActive(index),
        'product-wizard-steps__step--completed': isCompleted(index),
      }"
      :aria-current="isActive(index) ? 'step' : undefined"
      :disabled="step.disabled"
      @click="selectStep(index)"
    >
      <span class="product-wizard-steps__number">{{ index + 1 }}</span>
      <span class="product-wizard-steps__label">{{ step.label }}</span>
    </UiButton>
  </nav>
</template>

<style scoped>
.product-wizard-steps {
  --product-wizard-steps-background: var(--color-surface-subtle);
  --product-wizard-steps-step-background: var(--color-surface-base);
  --product-wizard-steps-step-border: var(--color-border-muted, var(--color-border-default));
  --product-wizard-steps-step-text: var(--color-text-default);
  --product-wizard-steps-step-active-background: var(--color-surface-brand);
  --product-wizard-steps-step-active-border: var(--color-border-brand);
  --product-wizard-steps-step-active-text: var(--color-text-inverse);
  --product-wizard-steps-step-completed-background: var(--color-status-success);
  --product-wizard-steps-step-completed-border: var(--color-status-success);
  --product-wizard-steps-number-background: var(--color-overlay-inverse-muted);
  --product-wizard-steps-number-idle-background: var(--color-surface-subtle);

  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
  padding: 16px;
  border-radius: 12px;
  background: var(--product-wizard-steps-background);
}

.product-wizard-steps__step {
  display: flex;
  flex: 1 1 160px;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-width: 0;
  padding: 8px 16px;
  border: 2px solid var(--product-wizard-steps-step-border);
  border-radius: 20px;
  background: var(--product-wizard-steps-step-background);
  color: var(--product-wizard-steps-step-text);
  transition: border-color 0.3s, background 0.3s, color 0.3s, opacity 0.3s;
}

.product-wizard-steps__step:not(:disabled):hover {
  border-color: var(--color-border-brand);
}

.product-wizard-steps__step:focus-visible {
  outline: 2px solid var(--color-border-brand);
  outline-offset: 2px;
}

.product-wizard-steps__step:disabled {
  opacity: 0.6;
}

.product-wizard-steps__step--active {
  border-color: var(--product-wizard-steps-step-active-border);
  background: var(--product-wizard-steps-step-active-background);
  color: var(--product-wizard-steps-step-active-text);
}

.product-wizard-steps__step--completed {
  border-color: var(--product-wizard-steps-step-completed-border);
  background: var(--product-wizard-steps-step-completed-background);
  color: var(--product-wizard-steps-step-active-text);
}

.product-wizard-steps__number {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: var(--product-wizard-steps-number-background);
  font-size: 13px;
  font-weight: 700;
}

.product-wizard-steps__step:not(.product-wizard-steps__step--active, .product-wizard-steps__step--completed) .product-wizard-steps__number {
  background: var(--product-wizard-steps-number-idle-background);
}

.product-wizard-steps__label {
  min-width: 0;
  overflow-wrap: anywhere;
  font-size: 14px;
  font-weight: 500;
}

@media (--breakpoint-sm-down) {
  .product-wizard-steps__step {
    flex-basis: 100%;
    justify-content: flex-start;
    border-radius: 12px;
  }
}
</style>
