<script setup lang="ts">
import { computed, ref } from 'vue'
import UiInput from '@/components/ui/UiInput.vue'
const props = defineProps<{
  page: number
  end: number
  total: number
  direction: 'ltr' | 'rtl'
  mode: 'normal' | 'pages' | 'hidden'
  expanded: boolean
}>()
const emit = defineEmits<{ jump: [page: number] }>()
const hoverPage = ref<number | null>(null)
const percent = computed(() => (props.total ? (props.end / props.total) * 100 : 0))
function hover(event: PointerEvent) {
  const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
  let ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
  if (props.direction === 'rtl') ratio = 1 - ratio
  hoverPage.value = Math.round(ratio * (props.total - 1)) + 1
}
</script>
<template>
  <div
    v-if="total"
    class="reader-progress"
    :class="[`reader-progress--${mode}`, { 'reader-progress--expanded': expanded }]"
    :dir="direction"
  >
    <div class="reader-progress__content">
      <span class="reader-progress__page">{{ page === end ? page : `${page}–${end}` }}</span>
      <div
        class="reader-progress__track"
        :style="{
          '--progress': `${percent}%`,
          '--segment': `${100 / total}%`,
          '--fill-direction': direction === 'rtl' ? 'to left' : 'to right',
        }"
        @pointermove="hover"
        @pointerleave="hoverPage = null"
      >
        <span
          v-if="hoverPage !== null"
          class="reader-progress__preview"
          :style="{ insetInlineStart: `${((hoverPage - 0.5) / total) * 100}%` }"
        >{{ hoverPage }}</span>
        <UiInput
          class="reader-progress__input"
          type="range"
          aria-label="阅读进度"
          :aria-valuetext="`第 ${page} 页，共 ${total} 页`"
          :min="1"
          :max="total"
          :model-value="page"
          @update:model-value="emit('jump', Number($event))"
        />
      </div>
      <span class="reader-progress__page">{{ total }}</span>
    </div>
  </div>
</template>
<style scoped>
.reader-progress {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: var(--z-local-toolbar);
  padding: 18px 16px 6px;
  color: var(--color-text-inverse);
}

.reader-progress__content {
  display: flex;
  align-items: center;
  gap: 12px;
  height: 8px;
  transition: height 0.15s;
}

.reader-progress__track {
  position: relative;
  height: 3px;
  flex: 1;
  background: linear-gradient(
    var(--fill-direction),
    var(--color-action-brand) var(--progress),
    var(--color-overlay-inverse-muted) var(--progress)
  );
  border-radius: 4px;
}

.reader-progress__input {
  position: absolute;
  inset: -12px 0;
  width: 100%;
  opacity: 0;
  cursor: pointer;
}

.reader-progress__page {
  direction: ltr;
  font-size: 12px;
  opacity: 0;
  font-variant-numeric: tabular-nums;
}

.reader-progress__preview {
  direction: ltr;
  position: absolute;
  bottom: 18px;
  transform: translateX(-50%);
  padding: 3px 6px;
  border-radius: 4px;
  background: var(--color-surface-inverse-raised);
  font-size: 12px;
  pointer-events: none;
}

.reader-progress[dir='rtl'] .reader-progress__preview {
  transform: translateX(50%);
}

.reader-progress--hidden:not(:hover, :focus-within, .reader-progress--expanded)
  .reader-progress__content {
  opacity: 0;
}

.reader-progress:hover,
.reader-progress:focus-within,
.reader-progress--expanded {
  background: linear-gradient(transparent, var(--color-overlay-scrim));
}

.reader-progress:hover .reader-progress__content,
.reader-progress:focus-within .reader-progress__content,
.reader-progress--expanded .reader-progress__content {
  height: 28px;
}

.reader-progress:hover .reader-progress__track,
.reader-progress:focus-within .reader-progress__track,
.reader-progress--expanded .reader-progress__track,
.reader-progress--pages .reader-progress__track {
  height: 8px;
}

.reader-progress:hover .reader-progress__page,
.reader-progress:focus-within .reader-progress__page,
.reader-progress--expanded .reader-progress__page,
.reader-progress--pages .reader-progress__page {
  opacity: 1;
}

.reader-progress--pages .reader-progress__track::after {
  content: '';
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: repeating-linear-gradient(
    to right,
    transparent 0,
    transparent calc(var(--segment) - 1px),
    var(--color-surface-inverse) calc(var(--segment) - 1px),
    var(--color-surface-inverse) var(--segment)
  );
}
</style>
