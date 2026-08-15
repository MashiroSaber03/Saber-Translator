<template>
  <div class="ui-model-picker" :class="`ui-model-picker--fetch-${fetchAppearance}`">
    <div class="ui-model-picker__entry">
      <UiInput
        :id="inputId"
        class="ui-model-picker__input"
        :model-value="modelValue"
        :type="inputType"
        :placeholder="placeholder"
        :disabled="disabled"
        :readonly="readonly"
        @update:model-value="handleInput"
      />
      <UiButton
        v-if="showFetch"
        :variant="fetchVariant"
        type="button"
        class="ui-model-picker__fetch"
        :title="fetchTitle"
        :disabled="disabled || fetchDisabled || fetching"
        @click="emit('fetch')"
      >
        <UiIcon name="search" size="16" />
        <span>{{ fetching ? fetchingLabel : fetchLabel }}</span>
      </UiButton>
    </div>

    <div v-if="showOptions" class="ui-model-picker__options">
      <UiCombobox
        :input-id="optionsComboboxId"
        aria-label="可选模型"
        :model-value="modelValue"
        :options="options"
        :disabled="disabled"
        fit
        @change="handleSelect"
      />
      <span class="ui-model-picker__count">共 {{ resolvedModelCount }} 个模型</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, useId } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCombobox from '@/components/ui/UiCombobox.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiInput from '@/components/ui/UiInput.vue'
import type { UiSelectOption, UiSelectValue } from '@/components/ui/selectTypes'

const props = withDefaults(defineProps<{
  modelValue: UiSelectValue
  inputId?: string
  inputType?: string
  placeholder?: string
  disabled?: boolean
  readonly?: boolean
  showFetch?: boolean
  fetchAppearance?: 'default' | 'muted'
  fetchVariant?: 'primary' | 'secondary'
  fetchTitle?: string
  fetchLabel?: string
  fetchingLabel?: string
  fetching?: boolean
  fetchDisabled?: boolean
  options?: UiSelectOption[]
  modelCount?: number
}>(), {
  inputId: undefined,
  inputType: 'text',
  placeholder: '',
  disabled: false,
  readonly: false,
  showFetch: true,
  fetchAppearance: 'default',
  fetchVariant: 'secondary',
  fetchTitle: '获取可用模型列表',
  fetchLabel: '获取模型',
  fetchingLabel: '获取中...',
  fetching: false,
  fetchDisabled: false,
  options: () => [],
  modelCount: undefined,
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: UiSelectValue): void
  (e: 'change', value: UiSelectValue): void
  (e: 'fetch'): void
}>()

const generatedOptionsId = useId()

const resolvedModelCount = computed(() => {
  if (props.modelCount !== undefined) return props.modelCount
  return props.options.filter(option => option.value !== '').length
})

const showOptions = computed(() => resolvedModelCount.value > 0 && props.options.length > 0)
const optionsComboboxId = computed(() => props.inputId ? `${props.inputId}Options` : generatedOptionsId)

function handleInput(value: string | number): void {
  emit('update:modelValue', value)
}

function handleSelect(value: UiSelectValue): void {
  emit('update:modelValue', value)
  emit('change', value)
}
</script>

<style scoped>
.ui-model-picker {
  display: grid;
  gap: 10px;
}

.ui-model-picker__entry {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.ui-model-picker__input {
  flex: 1;
  min-width: 0;
}

.ui-model-picker__fetch {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-height: 38px;
  padding: 8px 12px;
  font-size: 13px;
  font-weight: 500;
  line-height: 1;
  white-space: nowrap;
}

.ui-model-picker__fetch.ui-button--primary {
  --ui-button-primary-background: var(--color-action-primary);
  --ui-button-primary-hover-background: var(--color-action-primary-hover);
  --ui-button-primary-shadow: none;
  --ui-button-primary-disabled-background: var(--color-action-primary);
  --ui-button-primary-disabled-opacity: 0.6;
}

.ui-model-picker--fetch-muted .ui-model-picker__fetch.ui-button--primary {
  --ui-button-primary-background: var(--color-surface-muted);
  --ui-button-primary-color: var(--color-text-default);
  --ui-button-primary-hover-background: var(--color-surface-interactive-hover);
  --ui-button-primary-hover-transform: none;
  --ui-button-primary-shadow: none;
  --ui-button-primary-hover-shadow: none;
  --ui-button-primary-disabled-background: var(--color-surface-muted);
  --ui-button-primary-disabled-color: var(--color-text-supporting);
}

.ui-model-picker__fetch:disabled {
  opacity: 0.6;
}

.ui-model-picker__options {
  display: grid;
  gap: 8px;
  padding: 12px;
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  background: var(--color-surface-subtle);
}

.ui-model-picker__count {
  color: var(--color-text-supporting);
  font-size: 12px;
  text-align: right;
}

@media (--breakpoint-sm-down) {
  .ui-model-picker__entry {
    align-items: stretch;
    flex-direction: column;
  }

  .ui-model-picker__fetch {
    width: 100%;
  }
}
</style>
