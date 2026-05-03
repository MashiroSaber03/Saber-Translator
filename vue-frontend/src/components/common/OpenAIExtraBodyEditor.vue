<template>
  <div class="openai-extra-body-editor">
    <div class="editor-header">
      <label>{{ label }}</label>
      <button
        type="button"
        class="btn btn-secondary btn-sm"
        :disabled="disabled || !localText.trim()"
        @click="formatJson"
      >
        格式化
      </button>
    </div>
    <textarea
      :value="localText"
      :rows="rows"
      :placeholder="placeholder"
      :disabled="disabled"
      class="extra-body-textarea"
      @input="handleInput"
    ></textarea>
    <div v-if="errorMessage" class="input-error">{{ errorMessage }}</div>
    <div v-else class="input-hint">{{ hint }}</div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'

const props = withDefaults(defineProps<{
  modelValue?: Record<string, unknown>
  label?: string
  hint?: string
  placeholder?: string
  rows?: number
  disabled?: boolean
  reservedKeys?: string[]
}>(), {
  label: '附加请求字段(JSON对象):',
  hint: '仅用于新增厂商特需 body 字段，例如 {"thinking":{"type":"disabled"}}',
  placeholder: '{\n  "thinking": {\n    "type": "disabled"\n  }\n}',
  rows: 6,
  disabled: false,
  reservedKeys: () => ['model', 'messages', 'temperature', 'response_format', 'stream']
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: Record<string, unknown> | undefined): void
}>()

const localText = ref('')
const errorMessage = ref('')

function cloneRecord(value: Record<string, unknown>): Record<string, unknown> {
  return JSON.parse(JSON.stringify(value)) as Record<string, unknown>
}

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
    const formatted = formatValue(value)
    if (formatted !== localText.value) {
      localText.value = formatted
    }
    errorMessage.value = ''
  },
  { immediate: true, deep: true }
)

function handleInput(event: Event): void {
  const nextValue = (event.target as HTMLTextAreaElement).value
  localText.value = nextValue

  const trimmed = nextValue.trim()
  if (!trimmed) {
    errorMessage.value = ''
    emit('update:modelValue', undefined)
    return
  }

  try {
    const parsed = parseObject(trimmed)
    errorMessage.value = ''
    emit('update:modelValue', cloneRecord(parsed))
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
    emit('update:modelValue', cloneRecord(parsed))
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'JSON 解析失败'
  }
}
</script>

<style scoped>
.openai-extra-body-editor {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.editor-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.editor-header label {
  margin-bottom: 0;
}

.extra-body-textarea {
  font-family: Consolas, 'Courier New', monospace;
  line-height: 1.5;
  resize: vertical;
}

.input-error {
  color: #d14343;
  font-size: 12px;
}
</style>
