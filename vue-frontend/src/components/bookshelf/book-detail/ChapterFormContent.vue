<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'

defineProps<{
  modelValue: string
}>()

defineEmits<{
  (event: 'update:modelValue', value: string): void
  (event: 'save'): void
}>()
</script>

<template>
  <div class="chapter-form-field">
    <label for="chapterTitleInput">章节名称 <span class="required">*</span></label>
    <UiInput
      id="chapterTitleInput"
      :model-value="modelValue"
      class="chapter-form-input"
      type="text"
      autocomplete="off"
      placeholder="例如：第1话、序章"
      @update:model-value="$emit('update:modelValue', String($event))"
      @keydown.enter="$emit('save')"
    />
  </div>
</template>

<style scoped>
.chapter-form-field {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chapter-form-field label {
  display: block;
  color: var(--color-text-default);
  font-weight: 500;
  font-size: 0.95rem;
  line-height: 1.4;
}

.required {
  color: var(--color-text-danger-strong);
}

.chapter-form-input {
  width: 100%;
  min-height: 42px;
  padding: 10px 12px;
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  outline: none;
  background: var(--color-surface-card);
  color: var(--color-text-default);
  font-size: 0.95rem;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.chapter-form-input:focus {
  border-color: var(--color-border-brand-gradient);
  box-shadow: 0 0 0 3px var(--book-detail-modal-shadow-strong);
}

.chapter-form-input::placeholder {
  color: var(--color-text-supporting);
}
</style>
