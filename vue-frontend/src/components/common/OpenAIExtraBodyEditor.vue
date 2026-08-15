<template>
  <UiField
    class="openai-extra-body-editor"
    variant="settings"
    :label="label"
    :control-id="resolvedInputId"
    :hint="errorMessage ? '' : hint"
    :error="errorMessage"
  >
    <template #label-actions>
      <UiButton
        variant="secondary"
        type="button"
        :disabled="disabled || !localText.trim()"
        size="sm"
        @click="formatJson"
      >
        格式化
      </UiButton>
    </template>
    <UiTextarea
      :id="resolvedInputId"
      :model-value="localText"
      :rows="rows"
      :placeholder="placeholder"
      :disabled="disabled"
      :error="Boolean(errorMessage)"
      variant="panel"
      class="extra-body-textarea"
      @update:model-value="handleInput"
    />
  </UiField>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import { deepClone } from '@/utils/deepClone'
import { computed, ref, toRaw, useId, watch } from 'vue'

const props = withDefaults(defineProps<{
  modelValue?: Record<string, unknown>
  label?: string
  hint?: string
  placeholder?: string
  rows?: number
  disabled?: boolean
  inputId?: string
  reservedKeys?: string[]
}>(), {
  label: '附加请求字段(JSON对象):',
  hint: '仅用于新增厂商特需 body 字段，例如 {"thinking":{"type":"disabled"}}',
  placeholder: '{\n  "thinking": {\n    "type": "disabled"\n  }\n}',
  rows: 6,
  disabled: false,
  inputId: '',
  reservedKeys: () => ['model', 'messages', 'temperature', 'response_format', 'stream']
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: Record<string, unknown> | undefined): void
}>()

const localText = ref('')
const errorMessage = ref('')
const generatedInputId = useId()
const resolvedInputId = computed(() => props.inputId || generatedInputId)
let pendingLocalModel: Record<string, unknown> | undefined
let hasPendingLocalModel = false

function formatValue(value?: Record<string, unknown>): string {
  if (!value || Object.keys(value).length === 0) return ''
  return JSON.stringify(value, null, 2)
}

function parseObject(text: string): Record<string, unknown> {
  const parsed = JSON.parse(text)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('必须输入 JSON 对象，不能是数组、字符串或数字')
  }

  const reservedKeys = Object.keys(parsed).filter((key) => props.reservedKeys.includes(key))
  if (reservedKeys.length > 0) {
    throw new Error(`不能覆盖统一模板保留字段: ${reservedKeys.join(', ')}`)
  }
  return parsed as Record<string, unknown>
}

watch(
  () => props.modelValue,
  (value) => {
    if (hasPendingLocalModel && toRaw(value) === pendingLocalModel) {
      hasPendingLocalModel = false
      pendingLocalModel = undefined
      errorMessage.value = ''
      return
    }
    hasPendingLocalModel = false
    pendingLocalModel = undefined
    const formatted = formatValue(value)
    if (formatted !== localText.value) {
      localText.value = formatted
    }
    errorMessage.value = ''
  },
  { immediate: true, deep: true }
)

function emitValue(value: Record<string, unknown> | undefined): void {
  hasPendingLocalModel = true
  pendingLocalModel = value
  emit('update:modelValue', value)
}

function handleInput(nextValue: string): void {
  localText.value = nextValue

  const trimmed = nextValue.trim()
  if (!trimmed) {
    errorMessage.value = ''
    emitValue(undefined)
    return
  }

  try {
    const parsed = parseObject(trimmed)
    errorMessage.value = ''
    emitValue(deepClone(parsed))
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'JSON 解析失败'
  }
}

function formatJson(): void {
  const trimmed = localText.value.trim()
  if (!trimmed) return

  try {
    const parsed = parseObject(trimmed)
    const formatted = JSON.stringify(parsed, null, 2)
    localText.value = formatted
    errorMessage.value = ''
    emitValue(deepClone(parsed))
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'JSON 解析失败'
  }
}
</script>

<style scoped>
.openai-extra-body-editor {
  margin-bottom: 0;
}

.extra-body-textarea {
  font-family: var(--font-mono);
  line-height: 1.5;
  resize: vertical;
}
</style>
