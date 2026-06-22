<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  as?: string
  leftWidth?: string
  rightWidth?: string
  gap?: string
  mode?: 'flow' | 'fixed' | 'overlay'
  mobileBreakpoint?: string
  collapsed?: 'none' | 'left' | 'right' | 'both'
  height?: string
  scrollMode?: 'page' | 'main' | 'panes'
  paneScroll?: boolean
  sidebars?: 'flow' | 'sticky' | 'fixed' | 'overlay'
  sidebarTop?: string
  leftInset?: string
  rightInset?: string
  contentInset?: string
  leftOffset?: string
  rightOffset?: string
  leftTop?: string
  rightTop?: string
  leftHeight?: string
  rightHeight?: string
  leftClass?: string
  mainClass?: string
  rightClass?: string
  mobileMode?: 'stack' | 'drawer'
}>(), {
  as: 'div',
  leftWidth: '',
  rightWidth: '',
  gap: '',
  mode: 'flow',
  mobileBreakpoint: '',
  collapsed: 'none',
  height: '',
  scrollMode: 'page',
  paneScroll: false,
  sidebars: 'flow',
  sidebarTop: '',
  leftInset: '',
  rightInset: '',
  contentInset: '',
  leftOffset: '',
  rightOffset: '',
  leftTop: '',
  rightTop: '',
  leftHeight: '',
  rightHeight: '',
  leftClass: '',
  mainClass: '',
  rightClass: '',
  mobileMode: 'stack',
})

const layoutStyle = computed(() => ({
  ...(props.leftWidth ? { '--ui-sidebar-left-width': props.leftWidth } : {}),
  ...(props.rightWidth ? { '--ui-sidebar-right-width': props.rightWidth } : {}),
  ...(props.gap ? { '--ui-sidebar-gap': props.gap } : {}),
  ...(props.mobileBreakpoint ? { '--ui-sidebar-mobile-breakpoint': props.mobileBreakpoint } : {}),
  ...(props.height ? { '--ui-sidebar-height': props.height } : {}),
  ...(props.sidebarTop ? { '--ui-sidebar-top': props.sidebarTop } : {}),
  ...(props.leftInset ? { '--ui-sidebar-left-inset': props.leftInset } : {}),
  ...(props.rightInset ? { '--ui-sidebar-right-inset': props.rightInset } : {}),
  ...(props.contentInset ? { '--ui-sidebar-content-inset': props.contentInset } : {}),
  ...(props.leftOffset ? { '--ui-sidebar-left-offset': props.leftOffset } : {}),
  ...(props.rightOffset ? { '--ui-sidebar-right-offset': props.rightOffset } : {}),
  ...(props.leftTop ? { '--ui-sidebar-left-top': props.leftTop } : {}),
  ...(props.rightTop ? { '--ui-sidebar-right-top': props.rightTop } : {}),
  ...(props.leftHeight ? { '--ui-sidebar-left-height': props.leftHeight } : {}),
  ...(props.rightHeight ? { '--ui-sidebar-right-height': props.rightHeight } : {}),
}))
</script>

<template>
  <component
    :is="as"
    class="ui-sidebar-layout"
    :class="[
      `ui-sidebar-layout--${mode}`,
      `ui-sidebar-layout--${collapsed}-collapsed`,
      `ui-sidebar-layout--scroll-${scrollMode}`,
      `ui-sidebar-layout--sidebars-${sidebars}`,
      `ui-sidebar-layout--mobile-${mobileMode}`,
      { 'ui-sidebar-layout--pane-scroll': paneScroll },
    ]"
    :style="layoutStyle"
  >
    <template v-if="!$slots.left && !$slots.right">
      <slot />
    </template>
    <template v-else>
      <div v-if="$slots.left" class="ui-sidebar-layout__left" :class="leftClass">
        <slot name="left" />
      </div>
      <div class="ui-sidebar-layout__main" :class="mainClass">
        <slot />
      </div>
      <div v-if="$slots.right" class="ui-sidebar-layout__right" :class="rightClass">
        <slot name="right" />
      </div>
    </template>
  </component>
</template>

<style scoped>
.ui-sidebar-layout {
  --ui-sidebar-left-width: auto;
  --ui-sidebar-right-width: auto;
  --ui-sidebar-gap: 0;
  --ui-sidebar-height: auto;
  --ui-sidebar-top: 0;
  --ui-sidebar-left-inset: 0;
  --ui-sidebar-right-inset: 0;
  --ui-sidebar-content-inset: 0;
  --ui-sidebar-left-offset: 0;
  --ui-sidebar-right-offset: 0;
  --ui-sidebar-left-top: var(--ui-sidebar-top);
  --ui-sidebar-right-top: var(--ui-sidebar-top);
  --ui-sidebar-left-height: auto;
  --ui-sidebar-right-height: auto;

  min-width: 0;
  height: var(--ui-sidebar-height);
  display: flex;
  gap: var(--ui-sidebar-gap);
}

.ui-sidebar-layout__left {
  width: var(--ui-sidebar-left-width);
  min-width: 0;
}

.ui-sidebar-layout__main {
  flex: 1 1 auto;
  min-width: 0;
}

.ui-sidebar-layout--sidebars-fixed > .ui-sidebar-layout__main,
.ui-sidebar-layout--sidebars-overlay > .ui-sidebar-layout__main {
  margin-left: var(--ui-sidebar-left-inset);
  margin-right: var(--ui-sidebar-right-inset);
  padding-inline: var(--ui-sidebar-content-inset);
}

.ui-sidebar-layout__right {
  width: var(--ui-sidebar-right-width);
  min-width: 0;
}

.ui-sidebar-layout--scroll-main,
.ui-sidebar-layout--scroll-panes {
  min-height: 0;
}

.ui-sidebar-layout--scroll-main > .ui-sidebar-layout__main,
.ui-sidebar-layout--scroll-panes > .ui-sidebar-layout__main,
.ui-sidebar-layout--scroll-panes > .ui-sidebar-layout__left,
.ui-sidebar-layout--scroll-panes > .ui-sidebar-layout__right {
  min-height: 0;
  overflow: auto;
}

.ui-sidebar-layout--pane-scroll > .ui-sidebar-layout__left,
.ui-sidebar-layout--pane-scroll > .ui-sidebar-layout__main,
.ui-sidebar-layout--pane-scroll > .ui-sidebar-layout__right {
  min-height: 0;
  overflow: auto;
}

.ui-sidebar-layout--sidebars-sticky > .ui-sidebar-layout__left,
.ui-sidebar-layout--sidebars-sticky > .ui-sidebar-layout__right {
  position: sticky;
  top: var(--ui-sidebar-top);
  align-self: flex-start;
  max-height: calc(100dvh - var(--ui-sidebar-top));
}

.ui-sidebar-layout--sidebars-fixed > .ui-sidebar-layout__left,
.ui-sidebar-layout--sidebars-overlay > .ui-sidebar-layout__left {
  position: fixed;
  top: var(--ui-sidebar-left-top);
  left: var(--ui-sidebar-left-offset);
  height: var(--ui-sidebar-left-height);
  max-height: calc(100dvh - var(--ui-sidebar-left-top));
  overflow: auto;
}

.ui-sidebar-layout--sidebars-fixed > .ui-sidebar-layout__right,
.ui-sidebar-layout--sidebars-overlay > .ui-sidebar-layout__right {
  position: fixed;
  top: var(--ui-sidebar-right-top);
  right: var(--ui-sidebar-right-offset);
  height: var(--ui-sidebar-right-height);
  max-height: calc(100dvh - var(--ui-sidebar-right-top));
  overflow: auto;
}

.ui-sidebar-layout--left-collapsed > .ui-sidebar-layout__left,
.ui-sidebar-layout--both-collapsed > .ui-sidebar-layout__left,
.ui-sidebar-layout--right-collapsed > .ui-sidebar-layout__right,
.ui-sidebar-layout--both-collapsed > .ui-sidebar-layout__right {
  display: none;
}

@media (--breakpoint-md-down) {
  .ui-sidebar-layout--mobile-stack.ui-sidebar-layout--sidebars-fixed,
  .ui-sidebar-layout--mobile-stack.ui-sidebar-layout--sidebars-overlay {
    height: auto;
    flex-direction: column;
  }

  .ui-sidebar-layout--mobile-stack.ui-sidebar-layout--sidebars-fixed > .ui-sidebar-layout__main,
  .ui-sidebar-layout--mobile-stack.ui-sidebar-layout--sidebars-overlay > .ui-sidebar-layout__main {
    order: -1;
  }

  .ui-sidebar-layout--mobile-stack.ui-sidebar-layout--sidebars-fixed > .ui-sidebar-layout__left,
  .ui-sidebar-layout--mobile-stack.ui-sidebar-layout--sidebars-fixed > .ui-sidebar-layout__right,
  .ui-sidebar-layout--mobile-stack.ui-sidebar-layout--sidebars-overlay > .ui-sidebar-layout__left,
  .ui-sidebar-layout--mobile-stack.ui-sidebar-layout--sidebars-overlay > .ui-sidebar-layout__right {
    position: static;
    width: 100%;
    height: auto;
    max-height: none;
    overflow: visible;
  }

  .ui-sidebar-layout--mobile-stack.ui-sidebar-layout--sidebars-fixed > .ui-sidebar-layout__main,
  .ui-sidebar-layout--mobile-stack.ui-sidebar-layout--sidebars-overlay > .ui-sidebar-layout__main {
    margin-left: 0;
    margin-right: 0;
    padding-inline: 0;
  }
}
</style>
