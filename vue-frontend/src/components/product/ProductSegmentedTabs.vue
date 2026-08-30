<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import type { UiIconName } from '@/components/ui/iconRegistry'

export type ProductSegmentedTab = {
  id: string
  label: string
  iconName?: UiIconName
  disabled?: boolean
}

const props = withDefaults(defineProps<{
  tabs: ProductSegmentedTab[]
  activeTab: string
  ariaLabel?: string
  layout?: 'wrap' | 'scroll'
  appearance?: 'segmented' | 'underline' | 'radio'
}>(), {
  ariaLabel: undefined,
  layout: 'wrap',
  appearance: 'segmented',
})

const emit = defineEmits<{
  'update:activeTab': [tabId: string]
  select: [tabId: string]
}>()

function selectTab(tab: ProductSegmentedTab): void {
  if (tab.disabled) return
  emit('update:activeTab', tab.id)
  emit('select', tab.id)
}

function activeTabIndex(tab: ProductSegmentedTab): number {
  if (tab.disabled) return -1
  if (tab.id === props.activeTab) return 0

  const enabledTabs = props.tabs.filter(candidate => !candidate.disabled)
  const hasActiveEnabledTab = enabledTabs.some(candidate => candidate.id === props.activeTab)
  return !hasActiveEnabledTab && enabledTabs[0]?.id === tab.id ? 0 : -1
}

function selectAdjacentTab(event: KeyboardEvent, tab: ProductSegmentedTab): void {
  const enabledTabs = props.tabs.filter(candidate => !candidate.disabled)
  if (enabledTabs.length === 0) return

  const currentIndex = Math.max(
    enabledTabs.findIndex(candidate => candidate.id === tab.id),
    enabledTabs.findIndex(candidate => candidate.id === props.activeTab),
    0
  )
  let nextIndex: number | null = null

  if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
    nextIndex = (currentIndex + 1) % enabledTabs.length
  } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
    nextIndex = (currentIndex - 1 + enabledTabs.length) % enabledTabs.length
  } else if (event.key === 'Home') {
    nextIndex = 0
  } else if (event.key === 'End') {
    nextIndex = enabledTabs.length - 1
  }

  if (nextIndex === null) return
  event.preventDefault()
  const nextTab = enabledTabs[nextIndex]!
  selectTab(nextTab)
  const tablist = (event.currentTarget as HTMLElement).closest('[role="tablist"]')
  const tabIndex = props.tabs.findIndex(candidate => candidate.id === nextTab.id)
  tablist?.querySelectorAll<HTMLElement>('[role="tab"]')[tabIndex]?.focus()
}
</script>

<template>
  <div
    class="product-segmented-tabs"
    :class="[
      `product-segmented-tabs--${props.layout}`,
      `product-segmented-tabs--appearance-${props.appearance}`,
    ]"
    role="tablist"
    :aria-label="ariaLabel"
  >
    <UiButton
      v-for="tab in props.tabs"
      :key="tab.id"
      variant="toolbar"
      class="product-segmented-tabs__tab"
      :class="{ 'product-segmented-tabs__tab--active': tab.id === props.activeTab }"
      role="tab"
      :aria-selected="tab.id === props.activeTab"
      :disabled="tab.disabled"
      :tabindex="activeTabIndex(tab)"
      @click="selectTab(tab)"
      @keydown="selectAdjacentTab($event, tab)"
    >
      <span v-if="$slots.tabIcon" class="product-segmented-tabs__icon-text" aria-hidden="true">
        <slot name="tabIcon" :tab="tab" />
      </span>
      <UiIcon v-else-if="tab.iconName" :name="tab.iconName" size="15" />
      <span>{{ tab.label }}</span>
    </UiButton>
  </div>
</template>

<style scoped>
.product-segmented-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: var(--product-segmented-tabs-gap, 4px);
  padding: var(--product-segmented-tabs-padding, 4px);
  border: 1px solid var(--product-segmented-tabs-border, var(--color-border-muted));
  border-radius: var(--product-segmented-tabs-radius, 8px);
  background: var(--product-segmented-tabs-background, var(--color-surface-muted));
  box-shadow: var(--product-segmented-tabs-shadow, none);
}

.product-segmented-tabs--scroll {
  flex-wrap: nowrap;
  overflow-x: auto;
}

.product-segmented-tabs__tab {
  position: var(--product-segmented-tabs-tab-position, static);
  display: inline-flex;
  align-items: center;
  flex: var(--product-segmented-tabs-tab-flex, 1 1 0);
  justify-content: center;
  min-width: var(--product-segmented-tabs-tab-min-width, max-content);
  gap: var(--product-segmented-tabs-tab-gap, 6px);
  padding: var(--product-segmented-tabs-tab-padding, 7px 12px);
  border: var(--product-segmented-tabs-tab-border, 0);
  border-radius: var(--product-segmented-tabs-tab-radius, var(--radius-control));
  background: var(--product-segmented-tabs-tab-background, transparent);
  color: var(--product-segmented-tabs-text, var(--color-text-supporting));
  font-size: var(--product-segmented-tabs-tab-font-size, inherit);
  font-weight: var(--product-segmented-tabs-tab-font-weight, inherit);
  line-height: var(--product-segmented-tabs-tab-line-height, normal);
  box-shadow: var(--product-segmented-tabs-tab-shadow, none);
}

.product-segmented-tabs--scroll .product-segmented-tabs__tab {
  flex: 0 0 auto;
}

.product-segmented-tabs__tab--active,
.product-segmented-tabs__tab--active:hover {
  border: var(--product-segmented-tabs-active-border, var(--product-segmented-tabs-tab-border, 0));
  background: var(--product-segmented-tabs-active-background, var(--color-surface-base));
  color: var(--product-segmented-tabs-active-text, var(--color-text-default));
  box-shadow: var(--product-segmented-tabs-active-shadow, var(--shadow-soft));
  font-weight: var(--product-segmented-tabs-active-font-weight, var(--product-segmented-tabs-tab-font-weight, inherit));
}

.product-segmented-tabs__icon-text {
  font-size: 15px;
  line-height: 1;
}

.product-segmented-tabs--appearance-underline {
  gap: 0;
  padding: 0;
  border-width: 0 0 1px;
  border-radius: 0;
  background: var(--color-surface-base);
}

.product-segmented-tabs--appearance-underline .product-segmented-tabs__tab {
  border-radius: 0;
  padding: var(--product-segmented-tabs-tab-padding, 12px 16px);
  border-bottom: 3px solid transparent;
  background: transparent;
  box-shadow: none;
}

.product-segmented-tabs--appearance-underline .product-segmented-tabs__tab--active,
.product-segmented-tabs--appearance-underline .product-segmented-tabs__tab--active:hover {
  border-bottom-color: var(--color-action-primary);
  background: transparent;
  color: var(--color-action-primary);
  box-shadow: none;
}

.product-segmented-tabs--appearance-radio .product-segmented-tabs__tab {
  position: relative;
  flex: var(--product-segmented-tabs-radio-tab-flex, 0 0 auto);
  gap: var(--product-segmented-tabs-radio-tab-gap, 6px);
  padding: var(--product-segmented-tabs-radio-tab-padding, 0);
  border: 0;
  background: transparent;
  box-shadow: none;
}

.product-segmented-tabs--appearance-radio .product-segmented-tabs__tab::before {
  box-sizing: border-box;
  flex: 0 0 var(--product-segmented-tabs-radio-size, 12px);
  width: var(--product-segmented-tabs-radio-size, 12px);
  height: var(--product-segmented-tabs-radio-size, 12px);
  border: 1px solid var(--product-segmented-tabs-radio-border, var(--color-text-secondary));
  border-radius: 50%;
  content: '';
}

.product-segmented-tabs--appearance-radio .product-segmented-tabs__tab--active::before {
  border-color: var(--product-segmented-tabs-radio-active-color, var(--color-action-primary));
  background: var(--product-segmented-tabs-radio-active-color, var(--color-action-primary));
  box-shadow: inset 0 0 0 var(--product-segmented-tabs-radio-inner-width, 3px)
    var(--product-segmented-tabs-radio-inner-color, var(--color-surface-base));
}
</style>
