<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import UiColorSwatchGroup from './UiColorSwatchGroup.vue'
import UiField from './UiField.vue'
import UiInput from './UiInput.vue'
import UiNumberField from './UiNumberField.vue'

const props = defineProps<{ modelValue: string }>()
const emit = defineEmits<{
  'update:modelValue': [value: string]
  validityChange: [valid: boolean]
}>()
const hex = ref(props.modelValue)
const valid = computed(() => /^#?[\da-f]{6}$/i.test(hex.value.trim()))
const channels = [
  { label: '红（R）', shift: 16 },
  { label: '绿（G）', shift: 8 },
  { label: '蓝（B）', shift: 0 },
]
const swatches = [
  { label: '黑色', value: '#000000' },
  { label: '白色', value: '#ffffff' },
  { label: '灰色', value: '#808080' },
  { label: '红色', value: '#ff0000' },
  { label: '橙色', value: '#ff8800' },
  { label: '黄色', value: '#ffff00' },
  { label: '绿色', value: '#00aa00' },
  { label: '蓝色', value: '#0000ff' },
]
const rgb = computed(() => Number.parseInt(props.modelValue.replace('#', ''), 16) || 0)

watch(() => props.modelValue, value => { hex.value = value })
watch(valid, value => emit('validityChange', value), { immediate: true })

function setColor(value: string): void {
  hex.value = value
  emit('update:modelValue', value)
}

function updateHex(value: string | number): void {
  hex.value = String(value)
  if (valid.value) emit('update:modelValue', `#${hex.value.trim().replace('#', '').toLowerCase()}`)
}

function updateChannel(shift: number, value: string | number | null): void {
  if (value === null) return
  const number = Number(value)
  if (!Number.isFinite(number)) return
  const channel = Math.max(0, Math.min(255, Math.round(number)))
  const color = (rgb.value & ~(255 << shift)) | (channel << shift)
  setColor(`#${color.toString(16).padStart(6, '0')}`)
}
</script>

<template>
  <div class="ui-color-picker">
    <div class="ui-color-picker__preview" :style="{ background: modelValue }" role="img" :aria-label="`颜色预览 ${modelValue}`"></div>
    <UiField label="HEX 色值" :error="valid ? undefined : '请输入六位十六进制色值，例如 #123456'">
      <UiInput :model-value="hex" aria-label="HEX 色值" :error="!valid" spellcheck="false" @update:model-value="updateHex" />
    </UiField>
    <div v-for="channel in channels" :key="channel.shift" class="ui-color-picker__channel">
      <span>{{ channel.label }}</span>
      <UiInput
        type="range"
        class="ui-color-picker__range"
        :model-value="(rgb >> channel.shift) & 255"
        :min="0"
        :max="255"
        :step="1"
        :aria-label="`${channel.label}滑块`"
        @update:model-value="updateChannel(channel.shift, $event)"
      />
      <UiNumberField
        :model-value="(rgb >> channel.shift) & 255"
        :min="0"
        :max="255"
        :step="1"
        :aria-label="channel.label"
        @update:model-value="updateChannel(channel.shift, $event)"
      />
    </div>
    <UiColorSwatchGroup class="ui-color-picker__swatches" :model-value="modelValue.toLowerCase()" :options="swatches" aria-label="常用颜色" @update:model-value="setColor" />
  </div>
</template>

<style scoped>
.ui-color-picker {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.ui-color-picker__preview {
  height: 48px;
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
}

.ui-color-picker__channel {
  display: grid;
  grid-template-columns: 54px minmax(60px, 1fr) 94px;
  align-items: center;
  gap: 12px;
  color: var(--color-text-default);
  font-size: 13px;
}

.ui-color-picker__range {
  width: 100%;
  accent-color: var(--color-action-primary);
}

.ui-color-picker__swatches {
  --ui-swatch-border-color: var(--color-border-muted);
}
</style>
