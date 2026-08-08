<script setup lang="ts">
import { computed } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'

const props = withDefaults(defineProps<{
  disabled?: boolean
  inputLabel?: string
  modelValue: string
  placeholder?: string
  rows?: number | string
  showSubmitIcon?: boolean
  submitLabel?: string
}>(), {
  disabled: false,
  inputLabel: undefined,
  placeholder: '',
  rows: 1,
  showSubmitIcon: true,
  submitLabel: '发送',
})

const emit = defineEmits<{
  (event: 'submit'): void
  (event: 'update:modelValue', value: string): void
}>()

const valueModel = computed({
  get: () => props.modelValue,
  set: value => emit('update:modelValue', value),
})

const submitDisabled = computed(() => props.disabled || !props.modelValue.trim())

function submit(): void {
  if (submitDisabled.value) return
  emit('submit')
}

function handleKeydown(event: KeyboardEvent): void {
  if (event.key !== 'Enter' || event.shiftKey) return
  event.preventDefault()
  submit()
}
</script>

<template>
  <div class="product-composer">
    <UiTextarea
      v-model="valueModel"
      class="product-composer__input"
      :placeholder="placeholder"
      :rows="rows"
      :disabled="disabled"
      :aria-label="inputLabel || placeholder || undefined"
      @keydown="handleKeydown"
    />
    <UiButton
      variant="primary"
      class="product-composer__submit"
      :aria-label="submitLabel"
      :disabled="submitDisabled"
      @click="submit"
    >
      <UiIcon v-if="showSubmitIcon" name="send" />
      <span>{{ submitLabel }}</span>
    </UiButton>
  </div>
</template>

<style scoped>
.product-composer {
  --ui-textarea-min-height: 48px;
  --ui-textarea-radius: 12px;
  --ui-textarea-padding: 12px 14px;
  --ui-button-primary-background: var(--color-action-primary);
  --ui-button-primary-hover-background: var(--color-action-primary-hover);
  --ui-button-primary-shadow: none;
  --ui-button-primary-hover-shadow: none;
  --ui-button-primary-hover-transform: none;

  display: flex;
  align-items: flex-end;
  gap: 12px;
}

.product-composer__input {
  flex: 1 1 auto;
  resize: vertical;
}

.product-composer__submit {
  flex: 0 0 auto;
  min-height: 48px;
  padding-inline: 18px;
  border-radius: 12px;
}

@media (--breakpoint-sm-down) {
  .product-composer {
    flex-direction: column;
    align-items: stretch;
  }

  .product-composer__submit {
    width: 100%;
  }
}
</style>
