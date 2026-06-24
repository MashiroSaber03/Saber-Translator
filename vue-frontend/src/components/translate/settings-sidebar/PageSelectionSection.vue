<script setup lang="ts">
import CollapsiblePanel from '@/components/common/CollapsiblePanel.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiInput from '@/components/ui/UiInput.vue'

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
</script>

<template>
  <CollapsiblePanel
    title="指定翻译页码"
    :default-expanded="false"
    variant="settings"
    class="settings-panel"
  >
    <div class="settings-form page-selection-form">
      <div class="range-header-row">
        <label class="page-selection-toggle-compact">
          <UiInput
            class="page-selection-toggle-input"
            type="checkbox"
            :model-value="enabled"
            :disabled="totalImages === 0 || !supportsPageSelection"
            @update:model-value="$emit('update:enabled', Boolean($event))"
          />
          <span>启用</span>
        </label>
        <span class="total-count">共 {{ totalImages }} 张</span>
      </div>

      <div v-if="!supportsPageSelection" class="page-selection-note">当前模式不支持指定翻译页码</div>

      <div v-if="isActive" class="page-selection-summary-block">
        <div class="page-selection-summary-value">
          {{ summaryFor(normalizedSelectedPages) }}
        </div>
        <UiButton
          variant="toolbar"
          type="button"
          class="settings-button secondary-button page-selection-open-btn"
          :disabled="totalImages === 0"
          @click="$emit('open')"
        >
          选择页码
        </UiButton>
      </div>

      <div
        v-if="isActive && !hasValidPageSelection && totalImages > 0"
        class="page-selection-error"
      >
        请至少选择一页
      </div>
    </div>
  </CollapsiblePanel>
</template>

<style scoped>
.settings-panel.collapsible-panel {
  margin: 0 0 12px;
  padding: 12px;
  border: 1px solid var(--settings-sidebar-shell-border-muted);
  border-radius: 12px;
  background: var(--settings-sidebar-shell-surface-muted);
}

.settings-form {
  display: flex;
  flex-direction: column;
}

.page-selection-form {
  gap: 8px;
}

.range-header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.page-selection-toggle-compact {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border: 1px solid var(--settings-sidebar-apply-actions-border-subtle);
  border-radius: 999px;
  background: var(--settings-sidebar-apply-actions-surface-inverse);
  color: var(--settings-sidebar-apply-actions-text-subtle);
  font-weight: 600;
  font-size: 12px;
  cursor: pointer;
}

.page-selection-toggle-compact:has(.page-selection-toggle-input:checked) {
  border-color: var(--settings-sidebar-apply-actions-border-hover);
  background: var(--settings-sidebar-apply-actions-surface-contrast);
  color: var(--settings-sidebar-apply-actions-text-supporting);
}

.total-count,
.page-selection-note {
  color: var(--settings-sidebar-apply-actions-text-disabled);
  font-size: 12px;
  font-weight: 500;
}

.page-selection-summary-block {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 4px 0 0;
}

.page-selection-summary-value {
  color: var(--settings-sidebar-apply-actions-text-inverse);
  font-size: 13px;
  line-height: 1.5;
  word-break: break-word;
}

.page-selection-open-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  align-self: stretch;
  width: 100%;
  padding: 0 14px;
}

.page-selection-error {
  margin-top: 2px;
  padding: 6px 10px;
  border: 1px solid var(--settings-sidebar-apply-actions-border-active);
  border-radius: 8px;
  background: var(--color-surface-slate-soft);
  color: var(--settings-sidebar-apply-actions-text-brand);
  font-weight: 600;
  font-size: 12px;
  text-align: center;
}

.secondary-button {
  min-height: 38px;
  border: 1px solid var(--settings-sidebar-workflow-border-strong);
  border-radius: 8px;
  background: var(--color-surface-plain);
  color: var(--settings-sidebar-workflow-text-subtle);
  font-weight: 600;
  font-size: 13px;
}
</style>
