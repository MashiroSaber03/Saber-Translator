<script setup lang="ts">
defineProps<{
  label: string
  required?: boolean
  hint?: string
  error?: string
}>()
</script>

<template>
  <div class="continuation-dialog-field">
    <label class="continuation-dialog-field__label">
      {{ label }}
      <span v-if="required" class="continuation-dialog-field__required">*</span>
    </label>
    <slot />
    <p v-if="error" class="continuation-dialog-field__error">{{ error }}</p>
    <p v-if="hint" class="continuation-dialog-field__hint">{{ hint }}</p>
  </div>
</template>

<style scoped>
.continuation-dialog-field {
  margin-bottom: 0;
}

.continuation-dialog-field__label {
  display: block;
  margin-bottom: 6px;
  font-size: 14px;
  font-weight: 600;
}

.continuation-dialog-field__required {
  color: var(--color-text-danger);
}

.continuation-dialog-field__hint {
  margin: 6px 0 0;
  color: var(--color-text-supporting, var(--color-text-subtle));
  font-size: 12px;
}

.continuation-dialog-field__error {
  margin: 6px 0 0;
  color: var(--color-text-danger);
  font-size: 12px;
}

:slotted(.continuation-dialog__form-input) {
  font: inherit;
}
</style>
