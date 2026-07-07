<script setup lang="ts">
import { ref } from 'vue'

const props = withDefaults(defineProps<{
  ariaLabel?: string
  ariaLive?: 'assertive' | 'off' | 'polite'
  empty?: boolean
  gap?: 'none' | 'sm' | 'md'
  padding?: 'none' | 'sm' | 'md'
  role?: 'feed' | 'list' | 'log' | 'region'
}>(), {
  ariaLabel: undefined,
  ariaLive: undefined,
  empty: false,
  gap: 'none',
  padding: 'md',
  role: undefined,
})

const scrollerEl = ref<HTMLElement | null>(null)

function scrollToBottom(): void {
  if (!scrollerEl.value) return
  scrollerEl.value.scrollTop = scrollerEl.value.scrollHeight
}

defineExpose({ scrollToBottom })
</script>

<template>
  <div
    ref="scrollerEl"
    class="product-scroll-stack"
    :class="[
      `product-scroll-stack--gap-${props.gap}`,
      `product-scroll-stack--padding-${props.padding}`,
      { 'product-scroll-stack--empty': props.empty },
    ]"
    :role="props.role"
    :aria-label="props.ariaLabel"
    :aria-live="props.ariaLive"
  >
    <slot v-if="!props.empty" />
    <slot v-else name="empty" />
  </div>
</template>

<style scoped>
.product-scroll-stack {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  min-height: 0;
  overflow-y: auto;
}

.product-scroll-stack--padding-none {
  padding: 0;
}

.product-scroll-stack--padding-sm {
  padding: 12px;
}

.product-scroll-stack--padding-md {
  padding: 20px;
}

.product-scroll-stack--gap-none {
  gap: 0;
}

.product-scroll-stack--gap-sm {
  gap: 8px;
}

.product-scroll-stack--gap-md {
  gap: 12px;
}

.product-scroll-stack--empty {
  justify-content: center;
}
</style>
