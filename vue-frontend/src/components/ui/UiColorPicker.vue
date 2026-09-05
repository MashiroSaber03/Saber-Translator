<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { hexToHsv, hsvToHex } from '@/utils/colorConversion'
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
const hsv = ref(hexToHsv(props.modelValue))
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

watch(() => props.modelValue, value => {
  hex.value = value
  // A HEX round-trip must not move the cursor or erase hue at white/black.
  if (value.toLowerCase() !== hsvToHex(hsv.value)) syncSpectrum(value)
})
watch(valid, value => emit('validityChange', value), { immediate: true })

function setColor(value: string): void {
  hex.value = value
  syncSpectrum(value)
  emit('update:modelValue', value)
}

function syncSpectrum(value: string): void {
  const next = hexToHsv(value)
  hsv.value = {
    h: next.s ? next.h : hsv.value.h,
    s: next.v ? next.s : hsv.value.s,
    v: next.v,
  }
}

function emitSpectrum(): void {
  hex.value = hsvToHex(hsv.value)
  emit('update:modelValue', hex.value)
}

function updateHue(value: string | number): void {
  hsv.value.h = Number(value)
  emitSpectrum()
}

function updateSpectrum(event: PointerEvent): void {
  const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
  if (!rect.width || !rect.height) return
  hsv.value.s = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
  hsv.value.v = 1 - Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))
  emitSpectrum()
}

function startSpectrum(event: PointerEvent): void {
  if (event.button !== 0 || !event.isPrimary) return
  event.preventDefault()
  const target = event.currentTarget as HTMLElement
  target.focus({ preventScroll: true })
  target.setPointerCapture(event.pointerId)
  updateSpectrum(event)
}

function dragSpectrum(event: PointerEvent): void {
  if ((event.currentTarget as HTMLElement).hasPointerCapture(event.pointerId)) updateSpectrum(event)
}

function endSpectrum(event: PointerEvent): void {
  const target = event.currentTarget as HTMLElement
  if (!target.hasPointerCapture(event.pointerId)) return
  // Down/move already sampled the pointer; layout may have shifted before release.
  target.releasePointerCapture(event.pointerId)
}

function handleSpectrumKey(event: KeyboardEvent): void {
  const step = event.shiftKey ? 0.1 : 0.01
  switch (event.key) {
    case 'ArrowLeft': hsv.value.s = Math.max(0, hsv.value.s - step); break
    case 'ArrowRight': hsv.value.s = Math.min(1, hsv.value.s + step); break
    case 'ArrowDown': hsv.value.v = Math.max(0, hsv.value.v - step); break
    case 'ArrowUp': hsv.value.v = Math.min(1, hsv.value.v + step); break
    case 'Home': hsv.value.s = 0; break
    case 'End': hsv.value.s = 1; break
    default: return
  }
  event.preventDefault()
  emitSpectrum()
}

function updateHex(value: string | number): void {
  hex.value = String(value)
  if (valid.value) setColor(`#${hex.value.trim().replace('#', '').toLowerCase()}`)
}

function updateChannel(shift: number, value: number | null): void {
  if (value === null) return
  const channel = Math.round(value)
  const color = (rgb.value & ~(255 << shift)) | (channel << shift)
  setColor(`#${color.toString(16).padStart(6, '0')}`)
}
</script>

<template>
  <div class="ui-color-picker">
    <div
      class="ui-color-picker__spectrum"
      role="slider"
      aria-label="色盘"
      aria-description="左右键调整饱和度，上下键调整明度"
      aria-valuemin="0"
      aria-valuemax="100"
      :aria-valuenow="Math.round(hsv.s * 100)"
      :aria-valuetext="`饱和度 ${Math.round(hsv.s * 100)}%，明度 ${Math.round(hsv.v * 100)}%`"
      tabindex="0"
      :style="{ backgroundColor: `hsl(${hsv.h} 100% 50%)` }"
      @pointerdown="startSpectrum"
      @pointermove="dragSpectrum"
      @pointerup="endSpectrum"
      @keydown="handleSpectrumKey"
    >
      <span class="ui-color-picker__cursor" :style="{ left: `${hsv.s * 100}%`, top: `${(1 - hsv.v) * 100}%` }" aria-hidden="true"></span>
    </div>
    <UiInput
      type="range"
      class="ui-color-picker__hue"
      :model-value="hsv.h"
      :min="0"
      :max="360"
      :step="1"
      aria-label="色相"
      @update:model-value="updateHue"
    />
    <div class="ui-color-picker__hex-row">
      <div class="ui-color-picker__preview" :style="{ background: modelValue }" role="img" :aria-label="`颜色预览 ${modelValue}`"></div>
      <UiField variant="editor" label="HEX 色值" label-visually-hidden :error="valid ? undefined : '请输入六位十六进制色值，例如 #123456'">
        <UiInput size="sm" :model-value="hex" aria-label="HEX 色值" :error="!valid" spellcheck="false" @update:model-value="updateHex" />
      </UiField>
    </div>
    <div class="ui-color-picker__channels">
      <UiField v-for="channel in channels" :key="channel.shift" variant="editor" :label="channel.label">
        <UiNumberField
          size="xs"
          :model-value="(rgb >> channel.shift) & 255"
          :min="0"
          :max="255"
          :step="1"
          :aria-label="channel.label"
          @update:model-value="updateChannel(channel.shift, $event)"
        />
      </UiField>
    </div>
    <UiColorSwatchGroup class="ui-color-picker__swatches" :model-value="modelValue.toLowerCase()" :options="swatches" aria-label="常用颜色" @update:model-value="setColor" />
  </div>
</template>

<style scoped>
.ui-color-picker {
  --internal-ui-color-picker-value-gradient: linear-gradient(to top, #000, transparent);
  --internal-ui-color-picker-saturation-gradient: linear-gradient(to right, #fff, transparent);
  --internal-ui-color-picker-hue-gradient: linear-gradient(to right, #f00, #ff0, #0f0, #0ff, #00f, #f0f, #f00);
  --internal-ui-color-picker-cursor-border: 2px solid #fff;
  --internal-ui-color-picker-cursor-shadow: 0 0 0 1px #000;

  display: flex;
  flex-direction: column;
  gap: 10px;
  flex-shrink: 0;
}

.ui-color-picker__spectrum {
  position: relative;
  height: 120px;
  border-radius: 6px;
  background-image: var(--internal-ui-color-picker-value-gradient), var(--internal-ui-color-picker-saturation-gradient);
  cursor: crosshair;
  touch-action: none;
  user-select: none;
}

.ui-color-picker__spectrum:focus-visible {
  outline: 2px solid var(--color-border-brand);
  outline-offset: 3px;
}

.ui-color-picker__cursor {
  position: absolute;
  box-sizing: border-box;
  width: 12px;
  height: 12px;
  border: var(--internal-ui-color-picker-cursor-border);
  border-radius: 50%;
  box-shadow: var(--internal-ui-color-picker-cursor-shadow);
  transform: translate(-50%, -50%);
  pointer-events: none;
}

.ui-color-picker__hue {
  height: 16px;
  min-height: 16px;
  padding: 0;
  border: 0;
  border-radius: 4px;
  appearance: none;
  background: var(--internal-ui-color-picker-hue-gradient);
  cursor: pointer;
}

.ui-color-picker__hue::-webkit-slider-runnable-track {
  height: 16px;
  background: transparent;
}

.ui-color-picker__hue::-moz-range-track {
  height: 16px;
  background: transparent;
}

.ui-color-picker__hue::-webkit-slider-thumb {
  box-sizing: border-box;
  width: 10px;
  height: 20px;
  margin-top: -2px;
  border: var(--internal-ui-color-picker-cursor-border);
  border-radius: 3px;
  appearance: none;
  background: transparent;
  box-shadow: var(--internal-ui-color-picker-cursor-shadow);
}

.ui-color-picker__hue::-moz-range-thumb {
  box-sizing: border-box;
  width: 10px;
  height: 20px;
  border: var(--internal-ui-color-picker-cursor-border);
  border-radius: 3px;
  background: transparent;
  box-shadow: var(--internal-ui-color-picker-cursor-shadow);
}

.ui-color-picker__hex-row {
  display: grid;
  grid-template-columns: 36px minmax(0, 1fr);
  gap: 10px;
  align-items: start;
}

.ui-color-picker__preview {
  height: 32px;
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
}

.ui-color-picker__channels {
  --ui-number-field-width: 100%;
  --ui-number-field-input-width: 100%;

  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.ui-color-picker__swatches {
  --ui-swatch-border-color: var(--color-border-muted);
  --ui-swatch-size: 24px;
}
</style>
