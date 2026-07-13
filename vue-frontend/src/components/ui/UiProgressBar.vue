<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  value: number
  max?: number
  min?: number
  label?: string
  tone?: 'primary' | 'success' | 'brand'
  size?: 'sm' | 'md' | 'lg'
  striped?: boolean
  animated?: boolean
}>(), {
  max: 100,
  min: 0,
  label: '进度',
  tone: 'primary',
  size: 'sm',
  striped: false,
  animated: false,
})

const boundedValue = computed(() => {
  if (!Number.isFinite(props.value)) return props.min
  return Math.min(props.max, Math.max(props.min, props.value))
})

const percent = computed(() => {
  const range = props.max - props.min
  if (range <= 0) return 0
  return ((boundedValue.value - props.min) / range) * 100
})
</script>

<template>
  <div
    class="ui-progress-bar"
    :class="[
      `ui-progress-bar--tone-${tone}`,
      `ui-progress-bar--size-${size}`,
      {
        'ui-progress-bar--striped': striped,
        'ui-progress-bar--animated': animated,
      }
    ]"
  >
    <div v-if="$slots.default" class="ui-progress-bar__label">
      <slot />
    </div>
    <div
      class="ui-progress-bar__track"
      role="progressbar"
      :aria-label="label"
      :aria-valuemin="min"
      :aria-valuemax="max"
      :aria-valuenow="boundedValue"
    >
      <div class="ui-progress-bar__fill" :style="{ width: `${percent}%` }"></div>
    </div>
  </div>
</template>

<style scoped>
.ui-progress-bar {
  display: grid;
  gap: 8px;
  width: 100%;
}

.ui-progress-bar__label {
  color: var(--color-text-supporting);
  font-size: 13px;
}

.ui-progress-bar__track {
  width: 100%;
  height: var(--ui-progress-bar-height, var(--internal-ui-progress-bar-height, 8px));
  overflow: hidden;
  border-radius: 999px;
  background: var(--ui-progress-bar-track, var(--color-surface-muted));
}

.ui-progress-bar__fill {
  width: 0;
  height: 100%;
  background: var(--ui-progress-bar-fill, var(--internal-ui-progress-bar-fill, linear-gradient(90deg, var(--color-action-primary-soft), var(--color-action-primary-hover))));
  transition: width 0.3s ease;
  position: relative;
}

.ui-progress-bar--size-sm {
  --internal-ui-progress-bar-height: 8px;
}

.ui-progress-bar--size-md {
  --internal-ui-progress-bar-height: 10px;
}

.ui-progress-bar--size-lg {
  --internal-ui-progress-bar-height: 20px;
}

.ui-progress-bar--tone-primary {
  --internal-ui-progress-bar-fill: linear-gradient(90deg, var(--color-action-primary-soft), var(--color-action-primary-hover));
}

.ui-progress-bar--tone-success {
  --internal-ui-progress-bar-fill: linear-gradient(90deg, var(--color-status-success), var(--color-action-success-strong));
}

.ui-progress-bar--tone-brand {
  --internal-ui-progress-bar-fill: linear-gradient(90deg, var(--color-action-brand), var(--color-action-brand-strong));
}

.ui-progress-bar--striped .ui-progress-bar__fill::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background-image: linear-gradient(
    -45deg,
    var(--ui-progress-bar-stripe, var(--color-surface-raised)) 25%,
    transparent 25%,
    transparent 50%,
    var(--ui-progress-bar-stripe, var(--color-surface-raised)) 50%,
    var(--ui-progress-bar-stripe, var(--color-surface-raised)) 75%,
    transparent 75%,
    transparent
  );
  background-size: 30px 30px;
}

.ui-progress-bar--animated .ui-progress-bar__fill::after {
  animation: uiProgressStripeMove 1.5s linear infinite;
}

@keyframes uiProgressStripeMove {
  0% { background-position: 0 0; }
  100% { background-position: 30px 30px; }
}
</style>
