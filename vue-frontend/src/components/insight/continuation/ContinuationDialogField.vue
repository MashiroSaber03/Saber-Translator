<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'

defineProps<{
  label: string
  controlId?: string
  required?: boolean
  hint?: string
  error?: string
}>()
</script>

<template>
  <UiField
    class="continuation-dialog-field"
    variant="dialog"
    :label="label"
    :control-id="controlId"
    :required="required"
    :hint="hint"
    :error="error"
  >
    <slot />
  </UiField>
</template>

<style scoped>
.continuation-dialog-field {
  margin-bottom: 0;
}

:slotted(.continuation-dialog__form-input) {
  --ui-textarea-min-height: 0;

  font: inherit;
}
</style>
