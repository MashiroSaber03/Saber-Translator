<script setup lang="ts">
withDefaults(defineProps<{
  label?: string
  hint?: string
  description?: string
  error?: string
  forId?: string
  controlId?: string
  required?: boolean
  variant?: 'default' | 'settings' | 'dialog' | 'editor'
  tone?: 'default' | 'inverse'
  control?: 'default' | 'checkbox'
  layout?: 'stacked' | 'inline'
  labelVisuallyHidden?: boolean
}>(), {
  label: '',
  hint: '',
  description: '',
  error: '',
  forId: '',
  controlId: '',
  required: false,
  variant: 'default',
  tone: 'default',
  control: 'default',
  layout: 'stacked',
  labelVisuallyHidden: false,
})
</script>

<template>
  <div
    class="ui-field"
    :class="[
      `ui-field--${variant}`,
      `ui-field--tone-${tone}`,
      `ui-field--control-${control}`,
      `ui-field--layout-${layout}`,
      {
        'ui-field--invalid': Boolean(error),
        'ui-field--required': required,
      },
    ]"
  >
    <div
      v-if="label || $slots['label-actions']"
      class="ui-field__header"
      :class="{ 'ui-field__header--visually-hidden': labelVisuallyHidden }"
    >
      <label v-if="label" class="ui-field__label" :for="controlId || forId || undefined">
        {{ label }}
        <span v-if="required" class="ui-field__required" aria-hidden="true">*</span>
      </label>
      <div v-if="$slots['label-actions']" class="ui-field__label-actions">
        <slot name="label-actions" />
      </div>
    </div>
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
  color: var(--ui-field-label-color, var(--color-text-default));
  font-size: var(--ui-field-label-font-size, 0.9rem);
  font-weight: var(--ui-field-label-font-weight, 600);
}

.ui-field__header--visually-hidden {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.ui-field--tone-inverse .ui-field__label {
  color: var(--ui-field-inverse-label-color, color-mix(in srgb, var(--color-text-inverse) 70%, transparent));
}

.ui-field__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.ui-field__label-actions {
  display: inline-flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
  flex-shrink: 0;
}

.ui-field__required {
  color: var(--color-status-error, var(--color-text-danger));
  margin-left: 2px;
}

.ui-field__hint,
.ui-field__error {
  margin: 0;
  font-size: var(--ui-field-message-font-size, 0.82rem);
  line-height: var(--ui-field-message-line-height, 1.45);
}

.ui-field__hint {
  color: var(--ui-field-hint-color, var(--color-text-supporting));
}

.ui-field__error {
  color: var(--color-status-error, var(--color-text-danger));
}

.ui-field--settings {
  --ui-input-padding: 10px 12px;
  --ui-input-background: var(--color-surface-card, var(--color-surface-base));
  --ui-input-color: var(--color-text-strong);
  --ui-input-font-size: 0.95em;
  --ui-input-control-margin: 0;
  --ui-textarea-padding: 10px 12px;
  --ui-textarea-background: var(--color-surface-card, var(--color-surface-base));
  --ui-textarea-color: var(--color-text-strong);
  --ui-textarea-font-size: 0.95em;

  display: block;
  gap: 0;
  margin-bottom: 15px;
}

.ui-field--settings:last-child {
  margin-bottom: 0;
}

.ui-field--settings > .ui-field__header {
  margin-bottom: var(--ui-field-settings-header-margin-bottom, 6px);
}

.ui-field--control-checkbox {
  display: block;
}

.ui-field--settings :slotted(label:not(.ui-checkbox)) {
  display: block;
  margin-bottom: 6px;
  font-size: 0.95em;
  font-weight: 500;
}

.ui-field--settings :slotted(.ui-checkbox--with-content) {
  display: flex;
  align-items: center;
}

.ui-field--dialog {
  --ui-input-min-height: 46px;
  --ui-input-padding: 10px 12px;
  --ui-input-border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  --ui-input-radius: 8px;
  --ui-input-background: var(--color-surface-base);
  --ui-input-focus-border: var(--color-border-brand);
  --ui-input-line-height: inherit;
  --ui-textarea-min-height: 98px;
  --ui-textarea-padding: 10px 12px;
  --ui-textarea-border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  --ui-textarea-radius: 8px;
  --ui-textarea-background: var(--color-surface-base);
  --ui-textarea-focus-border: var(--color-border-brand);
  --ui-textarea-line-height: inherit;

  display: block;
  margin-bottom: 0;
}

.ui-field--dialog > .ui-field__header {
  margin-bottom: var(--ui-field-dialog-header-margin-bottom, 6px);
}

.ui-field--dialog .ui-field__label {
  display: block;
  font-size: 14px;
  font-weight: 600;
}

.ui-field--dialog .ui-field__required {
  color: var(--color-text-danger);
}

.ui-field--dialog > .ui-field__hint,
.ui-field--dialog > .ui-field__error {
  margin: 6px 0 0;
  font-size: 12px;
}

.ui-field--editor {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 0;
}

.ui-field--editor > .ui-field__header {
  margin-bottom: 0;
}

.ui-field--editor .ui-field__label {
  color: var(--ui-field-editor-label-color, var(--color-text-secondary));
  font-size: var(--ui-field-editor-label-font-size, 11px);
  font-weight: var(--ui-field-editor-label-font-weight, 600);
  letter-spacing: 0;
}

.ui-field.ui-field--layout-inline {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 10px;
  margin-bottom: 0;
}

.ui-field--layout-inline > .ui-field__header {
  flex-shrink: 0;
  margin-bottom: 0;
}

.ui-field--layout-inline .ui-field__label {
  color: var(--ui-field-inline-label-color, var(--color-text-muted));
  font-size: var(--ui-field-inline-label-font-size, 14px);
  font-weight: var(--ui-field-inline-label-font-weight, 500);
}

.ui-field :slotted(.ui-form-hint) {
  margin-top: 6px;
  color: var(--color-text-supporting, var(--color-text-secondary));
  font-size: 0.85em;
  line-height: 1.45;
}

.ui-field :slotted(.ui-form-hint--error) {
  color: var(--color-status-error, var(--color-text-danger));
}
</style>
