<script setup lang="ts">
withDefaults(defineProps<{
  ariaLabel?: string
  inputId?: string
  modelValue?: boolean
  label?: string
  description?: string
  disabled?: boolean
}>(), {
  ariaLabel: '',
  inputId: '',
  modelValue: false,
  label: '',
  description: '',
  disabled: false,
})

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  change: [value: boolean]
}>()

function handleChange(event: Event) {
  const checked = (event.target as HTMLInputElement).checked
  emit('update:modelValue', checked)
  emit('change', checked)
}
</script>

<template>
  <label
    class="ui-checkbox"
    :class="{
      'ui-checkbox--disabled': disabled,
      'ui-checkbox--checked': modelValue,
      'ui-checkbox--with-content': label || description,
    }"
  >
    <input
      :id="inputId || undefined"
      class="ui-checkbox__input"
      type="checkbox"
      :checked="modelValue"
      :disabled="disabled"
      :aria-label="ariaLabel || undefined"
      @change="handleChange"
    >
    <span class="ui-checkbox__content">
      <span v-if="label" class="ui-checkbox__label">{{ label }}</span>
      <span v-if="description" class="ui-checkbox__description">{{ description }}</span>
    </span>
  </label>
</template>

<style scoped>
.ui-checkbox {
  display: inline-flex;
  align-items: var(--ui-checkbox-align-items, flex-start);
  gap: var(--ui-checkbox-gap, 8px);
  color: var(--ui-checkbox-color, var(--color-text-default));
  cursor: pointer;
}

.ui-checkbox__input {
  width: var(--ui-checkbox-input-width, auto);
  height: var(--ui-checkbox-input-height, auto);
  margin: var(--ui-checkbox-input-margin, 2px 0 0);
  accent-color: var(--ui-checkbox-input-accent-color, auto);
}

.ui-checkbox__content {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.ui-checkbox__label {
  font-weight: var(--ui-checkbox-label-font-weight, 500);
}

.ui-checkbox__description {
  color: var(--color-text-supporting);
  font-size: 0.85em;
  line-height: 1.4;
}

.ui-checkbox--disabled {
  opacity: 0.65;
  cursor: not-allowed;
}

.ui-checkbox--checked {
  border-color: var(--ui-checkbox-checked-border-color, currentColor);
  background: var(--ui-checkbox-checked-background, transparent);
  color: var(--ui-checkbox-checked-color, var(--ui-checkbox-color, var(--color-text-default)));
}
</style>
