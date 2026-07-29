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
        <div class="page-selection-section__enable-control">
          <span>启用</span>
          <UiSwitch
            :model-value="enabled"
            accessibility-label="启用指定翻译页码"
            size="sm"
            :disabled="totalImages === 0 || !supportsPageSelection"
            @change="$emit('update:enabled', $event)"
          />
        </div>
        <span class="page-selection-section__total-count">共 {{ totalImages }} 张</span>
      </div>

      <ProductStatusBanner
        v-if="isActive"
        class="page-selection-section__summary"
        tone="neutral"
        role="note"
      >
        <span class="page-selection-section__summary-value">
          {{ summaryFor(normalizedSelectedPages) }}
        </span>
        <template #actions>
          <UiButton
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
  color: var(--settings-sidebar-page-selection-muted-text);
  font-weight: 600;
  font-size: 12px;
}

.page-selection-section__total-count,
.page-selection-section__note {
  color: var(--settings-sidebar-page-selection-muted-text);
  font-size: 12px;
  font-weight: 500;
}

.page-selection-section__summary {
  margin-top: 2px;
}

.page-selection-section__summary-value {
  color: var(--settings-sidebar-page-selection-summary-text);
  font-size: 13px;
  line-height: 1.5;
  word-break: break-word;
}

.page-selection-section__error {
  margin-top: 2px;
  font-weight: 600;
}
</style>
