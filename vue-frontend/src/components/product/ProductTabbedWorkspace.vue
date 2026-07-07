<script setup lang="ts">
import { computed } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import type { ProductClassValue } from '@/components/product/productClassTypes'
import type { UiIconName } from '@/components/ui/iconRegistry'

export type ProductWorkspaceTab = {
  id: string
  label: string
  iconName?: UiIconName
  disabled?: boolean
}

const props = withDefaults(defineProps<{
  tabs: ProductWorkspaceTab[]
  activeTab: string
  ariaLabel?: string
  panelsClass?: ProductClassValue
}>(), {
  ariaLabel: undefined,
  panelsClass: '',
})

const emit = defineEmits<{
  'update:activeTab': [tabId: string]
  select: [tabId: string]
}>()

const tablistAriaLabel = computed(() => props.ariaLabel ? `${props.ariaLabel}标签` : undefined)

function selectTab(tab: ProductWorkspaceTab): void {
  if (tab.disabled) return
  emit('update:activeTab', tab.id)
  emit('select', tab.id)
}

function activeTabIndex(tab: ProductWorkspaceTab): number {
  if (tab.disabled) return -1
  if (tab.id === props.activeTab) return 0

  const enabledTabs = props.tabs.filter(candidate => !candidate.disabled)
  const hasActiveEnabledTab = enabledTabs.some(candidate => candidate.id === props.activeTab)
  return !hasActiveEnabledTab && enabledTabs[0]?.id === tab.id ? 0 : -1
}

function selectAdjacentTab(event: KeyboardEvent, tab: ProductWorkspaceTab): void {
  const enabledTabs = props.tabs.filter(candidate => !candidate.disabled)
  if (enabledTabs.length === 0) return

  const currentIndex = Math.max(
    enabledTabs.findIndex(candidate => candidate.id === tab.id),
    enabledTabs.findIndex(candidate => candidate.id === props.activeTab),
    0,
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

function tabControlId(tabId: string): string {
  return `product-workspace-tab-${tabId}`
}

function tabPanelId(tabId: string): string {
  return `product-workspace-panel-${tabId}`
}
</script>

<template>
  <section class="product-tabbed-workspace" :aria-label="ariaLabel">
    <div class="product-tabbed-workspace__bar">
      <slot name="beforeTabs" />

      <div class="product-tabbed-workspace__tabs" role="tablist" :aria-label="tablistAriaLabel">
        <UiButton
          v-for="tab in props.tabs"
          :id="tabControlId(tab.id)"
          :key="tab.id"
          variant="toolbar"
          class="product-tabbed-workspace__tab"
          :class="{ 'product-tabbed-workspace__tab--active': tab.id === props.activeTab }"
          role="tab"
          :aria-selected="tab.id === props.activeTab"
          :aria-controls="tabPanelId(tab.id)"
          :disabled="tab.disabled"
          :tabindex="activeTabIndex(tab)"
          @click="selectTab(tab)"
          @keydown="selectAdjacentTab($event, tab)"
        >
          <UiIcon
            v-if="tab.iconName"
            class="product-tabbed-workspace__tab-icon"
            :name="tab.iconName"
            :size="16"
          />
          <span class="product-tabbed-workspace__tab-label">{{ tab.label }}</span>
        </UiButton>
      </div>

      <slot name="afterTabs" />
    </div>

    <div class="product-tabbed-workspace__panels" :class="panelsClass">
      <slot />
    </div>
  </section>
</template>

<style scoped>
.product-tabbed-workspace {
  --product-tabbed-workspace-bar-background: var(--color-surface-muted);
  --product-tabbed-workspace-border: var(--color-border-muted);
  --product-tabbed-workspace-tab-text: var(--color-text-secondary);
  --product-tabbed-workspace-tab-text-active: var(--color-text-inverse);
  --product-tabbed-workspace-tab-background-hover: var(--color-surface-hover);
  --product-tabbed-workspace-tab-background-active: var(--color-action-brand);

  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}

.product-tabbed-workspace__bar {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 4px;
  min-width: 0;
  padding: 12px 16px;
  background: var(--product-tabbed-workspace-bar-background);
  border-bottom: 1px solid var(--product-tabbed-workspace-border);
}

.product-tabbed-workspace__tabs {
  display: flex;
  flex: 1 1 auto;
  min-width: 0;
  gap: 4px;
  overflow-x: auto;
  overflow-y: hidden;
  scrollbar-gutter: stable;
}

.product-tabbed-workspace__tab {
  flex: 0 0 auto;
  gap: 6px;
  padding: 8px 16px;
  color: var(--product-tabbed-workspace-tab-text);
}

.product-tabbed-workspace__tab:hover {
  background: var(--product-tabbed-workspace-tab-background-hover);
}

.product-tabbed-workspace__tab--active,
.product-tabbed-workspace__tab--active:hover {
  color: var(--product-tabbed-workspace-tab-text-active);
  background: var(--product-tabbed-workspace-tab-background-active);
}

.product-tabbed-workspace__tab-icon {
  min-width: 0;
}

.product-tabbed-workspace__tab-label {
  min-width: 0;
  white-space: nowrap;
}

.product-tabbed-workspace__panels {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}
</style>
