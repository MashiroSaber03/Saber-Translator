<script setup lang="ts">
import { nextTick, onMounted, onUnmounted, ref, type CSSProperties } from 'vue'
import OverlayLayer from '@/components/ui/OverlayLayer.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiColorPicker from '@/components/ui/UiColorPicker.vue'
import type { BubbleColorField } from '@/types/bubble'
import { BUBBLE_COLOR_LABELS } from './bubbleColorFields'

const props = defineProps<{ field: BubbleColorField; color: string; anchor: HTMLElement }>()
const emit = defineEmits<{
  apply: [field: BubbleColorField, color: string]
  pick: [field: BubbleColorField]
  close: []
}>()
const draft = ref(props.color)
const valid = ref(true)
const popover = ref<HTMLElement | null>(null)
const position = ref<CSSProperties>({ visibility: 'hidden' })
const editor = props.anchor.closest('.bubble-editor')
const styleSection = props.anchor.closest('.bubble-editor__style-section')
let resizeObserver: ResizeObserver | undefined

function updatePosition(): void {
  if (!popover.value) return
  const anchor = props.anchor.getBoundingClientRect()
  if (!anchor.width || !anchor.height) return
  const bounds = editor?.getBoundingClientRect()
  const padding = 8
  const gap = 6
  // Keep the popover inside the editor panel so the comic stays unobstructed.
  const leftEdge = Math.max(padding, (bounds?.left ?? 0) + padding)
  const rightEdge = Math.min(window.innerWidth - padding, (bounds?.right ?? window.innerWidth) - padding)
  const topEdge = Math.max(padding, (bounds?.top ?? 0) + padding)
  const bottomEdge = Math.min(window.innerHeight - padding, (bounds?.bottom ?? window.innerHeight) - padding)
  const clip = styleSection?.getBoundingClientRect()
  if (anchor.bottom < Math.max(topEdge, clip?.top ?? topEdge) || anchor.top > Math.min(bottomEdge, clip?.bottom ?? bottomEdge) || anchor.right < leftEdge || anchor.left > rightEdge) {
    emit('close')
    return
  }
  const width = Math.min(288, rightEdge - leftEdge)
  const desiredHeight = popover.value.scrollHeight + 2
  const below = Math.max(0, bottomEdge - anchor.bottom - gap)
  const above = Math.max(0, anchor.top - topEdge - gap)
  const openAbove = below < desiredHeight && above > below
  const height = Math.min(desiredHeight, openAbove ? above : below)
  position.value = {
    width: `${width}px`,
    left: `${Math.max(leftEdge, Math.min(anchor.left, rightEdge - width))}px`,
    top: `${openAbove ? anchor.top - gap - height : anchor.bottom + gap}px`,
    maxHeight: `${height}px`,
  }
}

function close(): void {
  props.anchor.focus({ preventScroll: true })
  emit('close')
}

function apply(): void {
  props.anchor.focus({ preventScroll: true })
  emit('apply', props.field, draft.value)
}

function contains(target: EventTarget | null): boolean {
  return target instanceof Node && (props.anchor.contains(target) || Boolean(popover.value?.contains(target)))
}

function handleOutside(event: PointerEvent): void {
  if (!contains(event.target)) emit('close')
}

function handleFocusOut(event: FocusEvent): void {
  if (!contains(event.relatedTarget)) emit('close')
}

onMounted(() => {
  updatePosition()
  resizeObserver = new ResizeObserver(updatePosition)
  if (popover.value) resizeObserver.observe(popover.value)
  if (editor) resizeObserver.observe(editor)
  if (styleSection) resizeObserver.observe(styleSection)
  document.addEventListener('pointerdown', handleOutside, true)
  window.addEventListener('scroll', updatePosition, true)
  window.addEventListener('resize', updatePosition)
  void nextTick(() => {
    updatePosition()
    popover.value?.focus({ preventScroll: true })
  })
})

onUnmounted(() => {
  resizeObserver?.disconnect()
  document.removeEventListener('pointerdown', handleOutside, true)
  window.removeEventListener('scroll', updatePosition, true)
  window.removeEventListener('resize', updatePosition)
})
</script>

<template>
  <Teleport to="body">
    <OverlayLayer level="popover" passthrough>
      <div
        ref="popover"
        class="edit-color-popover"
        role="dialog"
        :aria-label="BUBBLE_COLOR_LABELS[field]"
        tabindex="-1"
        :style="position"
        @focusout="handleFocusOut"
        @keydown.esc.prevent.stop="close"
      >
        <span class="edit-color-popover__title">{{ BUBBLE_COLOR_LABELS[field] }}</span>
        <UiColorPicker v-model="draft" @validity-change="valid = $event" />
        <div class="edit-color-popover__actions">
          <UiButton variant="secondary" size="sm" @click="emit('pick', field)">从图片取色</UiButton>
          <UiButton variant="secondary" size="sm" @click="close">取消</UiButton>
          <UiButton variant="primary" size="sm" :disabled="!valid" @click="apply">应用颜色</UiButton>
        </div>
      </div>
    </OverlayLayer>
  </Teleport>
</template>

<style scoped>
.edit-color-popover {
  position: absolute;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 288px;
  padding: 12px;
  overflow: auto;
  overscroll-behavior: contain;
  border: 1px solid var(--color-border-muted);
  border-radius: 10px;
  background: var(--color-surface-card);
  color: var(--color-text-default);
  box-shadow: 0 4px 20px var(--shadow-medium);
}

.edit-color-popover__title {
  font-size: 13px;
  font-weight: 600;
}

.edit-color-popover__actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
}
</style>
