<script setup lang="ts">
withDefaults(defineProps<{
  label?: string
  hint?: string
  description?: string
  error?: string
  forId?: string
  controlId?: string
  required?: boolean
}>(), {
  label: '',
  hint: '',
  description: '',
  error: '',
  forId: '',
  controlId: '',
  required: false,
})
</script>

<template>
  <div class="ui-field" :class="{ 'ui-field--invalid': Boolean(error), 'ui-field--required': required }">
    <label v-if="label" class="ui-field__label" :for="controlId || forId || undefined">
      {{ label }}
      <span v-if="required" class="ui-field__required" aria-hidden="true">*</span>
    </label>
    <slot />
    <p v-if="description || hint" class="ui-field__hint">{{ description || hint }}</p>
    <p v-if="error" class="ui-field__error">{{ error }}</p>
  </div>
</template>

<style scoped>
.ui-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 18px;
}

.ui-field__label {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--color-text-default);
}

.ui-field__required {
  color: var(--color-status-error, var(--color-text-danger));
  margin-left: 2px;
}

.ui-field__hint,
.ui-field__error {
  margin: 0;
  font-size: 0.82rem;
  line-height: 1.45;
}

.ui-field__hint {
  color: var(--color-text-supporting);
}

.ui-field__error {
  color: var(--color-status-error, var(--color-text-danger));
}
</style>
