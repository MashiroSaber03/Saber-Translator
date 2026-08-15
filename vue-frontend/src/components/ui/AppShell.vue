<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  chrome?: 'flow' | 'fixed'
  viewportMode?: 'page' | 'locked' | 'immersive'
  contentClass?: string
  headerHeight?: string
  contentPadding?: string
}>(), {
  chrome: 'flow',
  viewportMode: 'page',
  contentClass: '',
  headerHeight: '',
  contentPadding: '',
})

const shellStyle = computed(() => ({
  ...(props.headerHeight ? { '--ui-app-shell-header-height': props.headerHeight } : {}),
  ...(props.contentPadding ? { '--ui-app-shell-content-padding': props.contentPadding } : {}),
}))

</script>

<template>
  <div
    class="ui-app-shell"
    :class="[
      `ui-app-shell--chrome-${chrome}`,
      `ui-app-shell--viewport-${viewportMode}`,
    ]"
    :style="shellStyle"
  >
    <template v-if="!$slots.header && !contentClass">
      <slot />
    </template>
    <template v-else>
      <div v-if="$slots.header" class="ui-app-shell__header">
        <slot name="header" />
      </div>
      <div class="ui-app-shell__content" :class="contentClass">
        <slot />
      </div>
    </template>
  </div>
</template>

<style scoped>
.ui-app-shell {
  --ui-app-shell-header-height: auto;
  --ui-app-shell-content-padding: 0;

  min-height: 100vh;
}

.ui-app-shell--viewport-locked {
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
  min-height: calc(100vh - var(--ui-app-shell-header-height));
}

.ui-app-shell--chrome-fixed > .ui-app-shell__header {
  position: sticky;
  top: 0;
  z-index: var(--z-app-header);
}

</style>
