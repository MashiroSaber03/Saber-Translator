<script setup lang="ts">
import { computed, ref } from 'vue'
import UiIcon from './UiIcon.vue'
import UiIconButton from './UiIconButton.vue'
import UiInput from './UiInput.vue'

const props = withDefaults(defineProps<{
  modelValue?: string
  inputId?: string
  placeholder?: string
  autocomplete?: string
  disabled?: boolean
  readonly?: boolean
  showLabel?: string
  hideLabel?: string
}>(), {
  modelValue: '',
  inputId: undefined,
  placeholder: '',
  autocomplete: 'off',
  disabled: false,
  readonly: false,
  showLabel: '显示密钥',
  hideLabel: '隐藏密钥',
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

const isVisible = ref(false)
const inputType = computed(() => (isVisible.value ? 'text' : 'password'))
const toggleLabel = computed(() => (isVisible.value ? props.hideLabel : props.showLabel))
const iconName = computed(() => (isVisible.value ? 'eye-off' : 'eye'))

function handleUpdate(value: string | number): void {
  emit('update:modelValue', String(value))
}

function toggleVisibility(): void {
  isVisible.value = !isVisible.value
}
</script>

<template>
  <div class="ui-password-field">
    <UiInput
      :id="inputId"
      class="ui-password-field__input"
      :model-value="modelValue"
      :type="inputType"
      :placeholder="placeholder"
      :autocomplete="autocomplete"
      :disabled="disabled"
      :readonly="readonly"
      @update:model-value="handleUpdate"
    />
    <UiIconButton
      class="ui-password-field__toggle"
      variant="plain"
      size="sm"
      type="button"
      :label="toggleLabel"
      :disabled="disabled"
      @click="toggleVisibility"
    >
      <UiIcon :name="iconName" />
    </UiIconButton>
  </div>
</template>

<style scoped>
.ui-password-field {
  position: relative;
  display: flex;
  align-items: center;
  width: 100%;
}

.ui-password-field__input {
  padding-right: 44px;
}

.ui-password-field__toggle {
  position: absolute;
  top: 50%;
  right: 6px;
  color: var(--color-text-supporting);
  transform: translateY(-50%);
}
</style>
