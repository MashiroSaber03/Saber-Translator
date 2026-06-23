<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  variant?: 'default' | 'reader' | 'studio'
  chrome?: 'flow' | 'fixed' | 'immersive'
  viewportMode?: 'page' | 'locked' | 'immersive'
  contentClass?: string
  headerHeight?: string
  headerOffset?: string
  contentPadding?: string
  scrollMode?: 'page' | 'content'
  contentScroll?: 'page' | 'content'
  fullHeight?: boolean
}>(), {
  variant: 'default',
  chrome: 'flow',
  viewportMode: 'page',
  contentClass: '',
  headerHeight: '',
  headerOffset: '',
  contentPadding: '',
  scrollMode: 'page',
  contentScroll: undefined,
  fullHeight: false,
})

const shellStyle = computed(() => ({
  ...(props.headerHeight ? { '--ui-app-shell-header-height': props.headerHeight } : {}),
  ...(props.headerOffset ? { '--ui-app-shell-header-offset': props.headerOffset } : {}),
  ...(props.contentPadding ? { '--ui-app-shell-content-padding': props.contentPadding } : {}),
}))

const effectiveScrollMode = computed(() => props.contentScroll ?? props.scrollMode)
</script>

<template>
  <div
    class="ui-app-shell"
    :class="[
      `ui-app-shell--${variant}`,
      `ui-app-shell--chrome-${chrome}`,
      `ui-app-shell--viewport-${viewportMode}`,
      `ui-app-shell--scroll-${effectiveScrollMode}`,
      { 'ui-app-shell--full-height': fullHeight },
    ]"
    :style="shellStyle"
  >
    <template v-if="!$slots.header && !$slots.overlay && !contentClass">
      <slot />
    </template>
    <template v-else>
      <div v-if="$slots.header" class="ui-app-shell__header">
        <slot name="header" />
      </div>
      <div class="ui-app-shell__content" :class="contentClass">
        <slot />
      </div>
      <div v-if="$slots.overlay" class="ui-app-shell__overlay">
        <slot name="overlay" />
      </div>
    </template>
  </div>
</template>

<style scoped>
.ui-app-shell {
  --ui-app-shell-header-height: auto;
  --ui-app-shell-header-offset: 0;
  --ui-app-shell-content-padding: 0;

  min-height: 100vh;
}

.ui-app-shell--viewport-locked,
.ui-app-shell--full-height {
  height: 100vh;
  min-height: 0;
}

.ui-app-shell--viewport-immersive {
  width: 100%;
  height: 100vh;
  min-height: 0;
  overflow: hidden;
}

.ui-app-shell--viewport-locked,
.ui-app-shell--viewport-immersive {
  display: flex;
  flex-direction: column;
}

.ui-app-shell__header {
  min-height: var(--ui-app-shell-header-height);
}

.ui-app-shell__content {
  min-width: 0;
  padding: var(--ui-app-shell-content-padding);
}

.ui-app-shell--viewport-locked > .ui-app-shell__content,
.ui-app-shell--viewport-immersive > .ui-app-shell__content {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  min-height: 0;
}

.ui-app-shell--chrome-fixed > .ui-app-shell__content {
  min-height: calc(100vh - var(--ui-app-shell-header-offset));
}

.ui-app-shell--chrome-fixed > .ui-app-shell__header {
  position: sticky;
  top: 0;
  z-index: var(--z-app-header);
}

.ui-app-shell--scroll-content {
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.ui-app-shell--scroll-content > .ui-app-shell__content {
  flex: 1 1 auto;
  min-height: 0;
  overflow: auto;
}

.ui-app-shell__overlay {
  position: relative;
  z-index: var(--z-overlay);
}

.ui-app-shell--viewport-locked > .ui-app-shell__overlay,
.ui-app-shell--viewport-immersive > .ui-app-shell__overlay {
  position: fixed;
  inset: 0;
  pointer-events: none;
}

.ui-app-shell--viewport-locked > .ui-app-shell__overlay > *,
.ui-app-shell--viewport-immersive > .ui-app-shell__overlay > * {
  pointer-events: auto;
}
</style>
