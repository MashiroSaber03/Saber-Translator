<script setup lang="ts">
import { computed, onUnmounted, ref } from 'vue'

const props = withDefaults(defineProps<{
  leftPaneWidth?: number
  min?: number
  max?: number
  step?: number
  resizerWidth?: number
  ariaLabel?: string
  resizerLabel?: string
  leftScrollTestId?: string
  rightScrollTestId?: string
}>(), {
  leftPaneWidth: 50,
  min: 30,
  max: 70,
  step: 2,
  resizerWidth: 8,
  ariaLabel: undefined,
  resizerLabel: '调整左右面板宽度',
  leftScrollTestId: undefined,
  rightScrollTestId: undefined,
})

const emit = defineEmits<{
  (event: 'update:leftPaneWidth', value: number): void
  (event: 'resize', value: number): void
}>()

const rootRef = ref<HTMLElement | null>(null)
const resizing = ref(false)

const currentWidth = computed(() => clampPaneWidth(props.leftPaneWidth))
const workspaceStyle = computed(() => ({
  '--product-split-left-track': `${currentWidth.value}fr`,
  '--product-split-right-track': `${100 - currentWidth.value}fr`,
  '--product-split-resizer-width': `${props.resizerWidth}px`,
}))

function clampPaneWidth(value: number): number {
  return Math.min(props.max, Math.max(props.min, value))
}

function emitPaneWidth(value: number) {
  const nextWidth = clampPaneWidth(value)
  emit('update:leftPaneWidth', nextWidth)
  emit('resize', nextWidth)
}

function handlePointerMove(event: PointerEvent) {
  if (!resizing.value || !rootRef.value) return
  const rect = rootRef.value.getBoundingClientRect()
  if (rect.width <= 0) return
  emitPaneWidth(((event.clientX - rect.left) / rect.width) * 100)
}

function stopResize() {
  if (!resizing.value) return
  resizing.value = false
  window.removeEventListener('pointermove', handlePointerMove)
  window.removeEventListener('pointerup', stopResize)
}

function startResize(event: PointerEvent) {
  event.preventDefault()
  resizing.value = true
  window.addEventListener('pointermove', handlePointerMove)
  window.addEventListener('pointerup', stopResize)
}

function handleResizerKeydown(event: KeyboardEvent) {
  let next = currentWidth.value

  switch (event.key) {
    case 'ArrowLeft':
      next -= props.step
      break
    case 'ArrowRight':
      next += props.step
      break
    case 'Home':
      next = props.min
      break
    case 'End':
      next = props.max
      break
    default:
      return
  }

  event.preventDefault()
  emitPaneWidth(next)
}

onUnmounted(stopResize)
</script>

<template>
  <div
    ref="rootRef"
    class="product-split-workspace"
    :class="{ 'product-split-workspace--resizing': resizing }"
    :aria-label="ariaLabel"
    :style="workspaceStyle"
  >
    <section class="product-split-workspace__pane product-split-workspace__pane--left">
      <div class="product-split-workspace__scroll" :data-testid="leftScrollTestId">
        <slot name="left" />
      </div>
    </section>

    <div
      class="product-split-workspace__resizer"
      role="separator"
      tabindex="0"
      :aria-label="resizerLabel"
      aria-orientation="vertical"
      :aria-valuemin="min"
      :aria-valuemax="max"
      :aria-valuenow="Math.round(currentWidth)"
      @pointerdown="startResize"
      @keydown="handleResizerKeydown"
    ></div>

    <section class="product-split-workspace__pane product-split-workspace__pane--right">
      <div class="product-split-workspace__scroll" :data-testid="rightScrollTestId">
        <slot name="right" />
      </div>
    </section>
  </div>
</template>

<style scoped>
.product-split-workspace {
  --product-split-resizer-background: var(--color-focus-brand-soft);
  --product-split-resizer-background-strong: var(--color-action-primary-soft);

  display: grid;
  flex: 1 1 auto;
  grid-template-columns:
    minmax(0, var(--product-split-left-track))
    var(--product-split-resizer-width)
    minmax(0, var(--product-split-right-track));
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}

.product-split-workspace--resizing {
  cursor: col-resize;
  user-select: none;
}

.product-split-workspace__pane {
  min-width: 0;
  min-height: 0;
}

.product-split-workspace__scroll {
  height: 100%;
  min-height: 0;
  overflow-x: hidden;
  overflow-y: auto;
}

.product-split-workspace__pane--left > .product-split-workspace__scroll {
  scrollbar-gutter: stable;
}

.product-split-workspace__resizer {
  width: 100%;
  cursor: col-resize;
  border-radius: 999px;
  background: linear-gradient(
    180deg,
    var(--product-split-resizer-background),
    var(--product-split-resizer-background-strong)
  );
}

.product-split-workspace__resizer:focus-visible {
  outline: 2px solid var(--color-border-brand);
  outline-offset: 3px;
}

@media (--breakpoint-md-down) {
  .product-split-workspace {
    grid-template-columns: 1fr;
    gap: 16px;
  }

  .product-split-workspace__resizer {
    display: none;
  }
}
</style>
