<script setup lang="ts">
import ProductCollapsibleSection from '@/components/product/ProductCollapsibleSection.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiSwitch from '@/components/ui/UiSwitch.vue'
import { ref } from 'vue'

defineProps<{
  enabled: boolean
  hasValidPageSelection: boolean
  normalizedSelectedPages: number[]
  supportsPageSelection: boolean
  totalImages: number
  isActive: boolean
  summaryFor: (pages: number[]) => string
}>()

defineEmits<{
  (event: 'open'): void
  (event: 'update:enabled', value: boolean): void
}>()

const isPageSelectionExpanded = ref(false)
</script>

<template>
  <ProductCollapsibleSection
    v-model:expanded="isPageSelectionExpanded"
    title="指定翻译页码"
    class="page-selection-section"
  >
    <div class="page-selection-section__form">
      <div class="page-selection-section__header">
        <div
          class="page-selection-section__enable-control"
          :class="{ 'page-selection-section__enable-control--checked': enabled }"
        >
          <UiSwitch
            :model-value="enabled"
            accessibility-label="启用指定翻译页码"
            size="sm"
            :disabled="totalImages === 0 || !supportsPageSelection"
            @change="$emit('update:enabled', $event)"
          />
          <span>启用</span>
        </div>
        <span class="page-selection-section__total-count">共 {{ totalImages }} 张</span>
      </div>

      <ProductStatusBanner
        v-if="isActive"
        class="page-selection-section__summary-block"
        tone="neutral"
        role="note"
      >
        <span class="page-selection-section__summary-value">
          {{ summaryFor(normalizedSelectedPages) }}
        </span>
        <template #actions>
          <UiButton
            class="page-selection-section__open-button"
            variant="secondary"
            size="sm"
            block
            type="button"
            :disabled="totalImages === 0"
            @click="$emit('open')"
          >
            选择页码
          </UiButton>
        </template>
      </ProductStatusBanner>

      <ProductStatusBanner
        v-if="!supportsPageSelection"
        class="page-selection-section__note"
        tone="warning"
        role="note"
      >
        当前模式不支持指定翻译页码
      </ProductStatusBanner>

      <ProductStatusBanner
        v-if="isActive && !hasValidPageSelection && totalImages > 0"
        class="page-selection-section__error"
        tone="danger"
        role="alert"
      >
        请至少选择一页
      </ProductStatusBanner>
    </div>
  </ProductCollapsibleSection>
</template>

<style scoped>
.page-selection-section.product-collapsible-section {
  --settings-sidebar-page-selection-panel-border: var(--color-border-muted);
  --settings-sidebar-page-selection-panel-background: var(--color-surface-quiet);
  --settings-sidebar-page-selection-muted-text: var(--color-text-supporting);
  --settings-sidebar-page-selection-summary-text: var(--color-text-default);
  --settings-sidebar-page-selection-header-divider: var(--color-border-muted);
  --settings-sidebar-page-selection-toggle-text: var(--color-text-supporting);
  --settings-sidebar-page-selection-enable-border: var(--color-border-muted);
  --settings-sidebar-page-selection-enable-background: var(--color-surface-muted);
  --settings-sidebar-page-selection-enable-text: var(--settings-sidebar-page-selection-muted-text);
  --product-collapsible-section-header-gap: 0;
  --product-collapsible-section-header-margin: 0 0 8px;
  --product-collapsible-section-header-padding: 0 0 8px;
  --product-collapsible-section-header-border: 0;
  --product-collapsible-section-header-border-bottom: 1px solid var(--settings-sidebar-page-selection-header-divider);
  --product-collapsible-section-header-background: transparent;
  --product-collapsible-section-header-hover-background: transparent;
  --product-collapsible-section-title-color: var(--color-text-heading);
  --product-collapsible-section-title-font-size: 17px;
  --product-collapsible-section-title-font-weight: 700;
  --product-collapsible-section-title-order: 1;
  --product-collapsible-section-hint-order: 2;
  --product-collapsible-section-toggle-order: 3;
  --product-collapsible-section-toggle-margin-left: auto;
  --product-collapsible-section-toggle-color: var(--settings-sidebar-page-selection-toggle-text);
  --product-collapsible-section-body-padding: 2px 0 0;
  --product-collapsible-section-body-background: transparent;

  margin: 0 0 12px;
  padding: 12px;
  border: 1px solid var(--settings-sidebar-page-selection-panel-border);
  border-radius: 12px;
  background: var(--settings-sidebar-page-selection-panel-background);
}

.page-selection-section__form {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.page-selection-section__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.page-selection-section__enable-control {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border: 1px solid var(--settings-sidebar-page-selection-enable-border);
  border-radius: 999px;
  background: var(--settings-sidebar-page-selection-enable-background);
  color: var(--settings-sidebar-page-selection-enable-text);
  font-weight: 600;
  font-size: 12px;
}

.page-selection-section__enable-control--checked {
  border-color: var(--color-border-accent);
  background: var(--color-focus-brand-soft);
  color: var(--color-action-primary);
}

.page-selection-section__total-count,
.page-selection-section__note {
  color: var(--settings-sidebar-page-selection-muted-text);
  font-size: 12px;
  font-weight: 500;
}

.page-selection-section__summary-block {
  --product-status-banner-icon-display: none;
  --product-status-banner-padding: 8px 10px;
  --product-status-banner-body-color: var(--settings-sidebar-page-selection-summary-text);
  --product-status-banner-actions-margin-left: 0;
  --product-status-banner-actions-width: 100%;
}

.page-selection-section__summary-value {
  display: block;
  line-height: 1.5;
  word-break: break-word;
}

.page-selection-section__note,
.page-selection-section__error {
  --product-status-banner-padding: 7px 10px;
  --product-status-banner-body-font-size: 12px;
}
</style>
