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

.ui-field.ui-settings-field {
  --ui-input-padding: 10px 12px;
  --ui-input-background: var(--color-surface-card, var(--color-surface-plain));
  --ui-input-color: var(--color-text-strong);
  --ui-input-font-size: 0.95em;
  --ui-input-control-margin: 0;
  --ui-select-padding: 10px 12px;
  --ui-select-background: var(--color-surface-card, var(--color-surface-plain));
  --ui-select-color: var(--color-text-strong);
  --ui-select-font-size: 0.95em;
  --ui-textarea-padding: 10px 12px;
  --ui-textarea-background: var(--color-surface-card, var(--color-surface-plain));
  --ui-textarea-color: var(--color-text-strong);
  --ui-textarea-font-size: 0.95em;

  display: block;
  gap: 0;
  margin-bottom: 15px;
}

.ui-field.ui-settings-field:last-child {
  margin-bottom: 0;
}

.ui-field.ui-settings-field--checkbox {
  display: block;
}

.ui-field.ui-settings-field :slotted(label:not(.ui-checkbox-label)) {
  display: block;
  margin-bottom: 6px;
  font-size: 0.95em;
  font-weight: 500;
}

.ui-field :slotted(.ui-checkbox-label) {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 5px;
  margin-bottom: 6px;
  cursor: pointer;
}

.ui-field :slotted(.ui-form-hint) {
  margin-top: 6px;
  color: var(--color-text-supporting, var(--color-text-secondary));
  font-size: 0.85em;
  line-height: 1.45;
}

.ui-field :slotted(.ui-form-hint--error),
.ui-field :slotted(.error-hint) {
  color: var(--color-status-error, var(--color-text-danger));
}
</style>
