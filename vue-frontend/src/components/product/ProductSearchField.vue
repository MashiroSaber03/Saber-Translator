<script setup lang="ts">
import { computed, ref } from 'vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiInput from '@/components/ui/UiInput.vue'

const props = withDefaults(defineProps<{
  ariaLabel?: string
  autofocus?: boolean
  clearLabel?: string
  clearable?: boolean
  disabled?: boolean
  modelValue?: string
  placeholder?: string
  size?: 'lg' | 'md' | 'sm' | 'xs'
}>(), {
  ariaLabel: '搜索',
  autofocus: false,
  clearLabel: '清除搜索',
  clearable: true,
  disabled: false,
  modelValue: '',
  placeholder: '',
  size: 'md',
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
  clear: []
  search: [value: string]
}>()

const inputRef = ref<InstanceType<typeof UiInput> | null>(null)
const hasValue = computed(() => props.modelValue.trim().length > 0)

function handleUpdate(value: string | number | boolean): void {
  emit('update:modelValue', String(value))
}

function handleSearch(): void {
  emit('search', props.modelValue)
}

function handleClear(): void {
  emit('update:modelValue', '')
  emit('clear')
}

function focus(): void {
  inputRef.value?.focus()
}

defineExpose({ focus })
</script>

<template>
  <div class="product-search-field">
    <UiIcon class="product-search-field__icon" name="search" size="16" />
    <UiInput
      ref="inputRef"
      class="product-search-field__input"
      type="search"
      :model-value="modelValue"
      :placeholder="placeholder"
      :disabled="disabled"
      :size="size"
      :aria-label="ariaLabel"
      :autofocus="autofocus || undefined"
      @update:model-value="handleUpdate"
      @keydown.enter="handleSearch"
    />
    <UiIconButton
      v-if="clearable && hasValue"
      class="product-search-field__clear"
      variant="soft"
      size="sm"
      type="button"
      :disabled="disabled"
      :label="clearLabel"
      :title="clearLabel"
      @click="handleClear"
    >
      <UiIcon name="x" size="14" />
    </UiIconButton>
  </div>
</template>

<style scoped>
.product-search-field {
  position: relative;
  display: flex;
  align-items: center;
  width: 100%;
}

.product-search-field__icon {
  position: absolute;
  left: 12px;
  z-index: var(--z-local);
  color: var(--product-search-field-icon, var(--color-text-supporting));
  pointer-events: none;
}

.product-search-field__input {
  --ui-input-padding: var(--product-search-field-input-padding, 9px 40px 9px 36px);
  --ui-input-radius: var(--product-search-field-radius, 10px);
  --ui-input-background: var(--product-search-field-background, var(--color-surface-input, var(--color-surface-card)));
  --ui-input-border: var(--product-search-field-border, 1px solid var(--color-border-muted));
  --ui-input-focus-border: var(--product-search-field-focus-border, var(--color-action-primary));
  --ui-input-focus-shadow: var(--product-search-field-focus-shadow, var(--color-focus-brand-subtle));
}

.product-search-field__clear {
  position: absolute;
  right: 5px;
  top: calc(50% - 15px);
}
</style>
