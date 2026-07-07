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
  appearance?: 'segmented' | 'underline'
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
  selectTab(enabledTabs[nextIndex])
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
      <UiIcon v-if="tab.iconName" :name="tab.iconName" size="15" />
      <span>{{ tab.label }}</span>
    </UiButton>
  </div>
</template>

<style scoped>
.product-segmented-tabs {
  --product-segmented-tabs-background: var(--color-surface-muted);
  --product-segmented-tabs-border: var(--color-border-muted);
  --product-segmented-tabs-active-background: var(--color-surface-base);
  --product-segmented-tabs-active-text: var(--color-text-default);
  --product-segmented-tabs-text: var(--color-text-supporting);

  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  padding: 4px;
  border: 1px solid var(--product-segmented-tabs-border);
  border-radius: 8px;
  background: var(--product-segmented-tabs-background);
}

.product-segmented-tabs--scroll {
  flex-wrap: nowrap;
  overflow-x: auto;
}

.product-segmented-tabs__tab {
  flex: 1 1 0;
  justify-content: center;
  min-width: max-content;
  gap: 6px;
  padding: 7px 12px;
  color: var(--product-segmented-tabs-text);
}

.product-segmented-tabs--scroll .product-segmented-tabs__tab {
  flex: 0 0 auto;
}

.product-segmented-tabs__tab--active,
.product-segmented-tabs__tab--active:hover {
  background: var(--product-segmented-tabs-active-background);
  color: var(--product-segmented-tabs-active-text);
  box-shadow: var(--shadow-soft);
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
  padding: 12px 16px;
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
</style>
