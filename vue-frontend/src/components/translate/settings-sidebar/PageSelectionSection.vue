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
  --settings-sidebar-page-selection-toggle-border: #d4deed;
  --settings-sidebar-page-selection-toggle-border-active: #94b5e5;
  --settings-sidebar-page-selection-toggle-background: #f4f8fd;
  --settings-sidebar-page-selection-toggle-background-active: #e9f2ff;
  --settings-sidebar-page-selection-toggle-text: #5d7090;
  --settings-sidebar-page-selection-toggle-text-active: #21579c;
  --settings-sidebar-page-selection-muted-text: #6f809a;
  --settings-sidebar-page-selection-summary-text: #304464;
  --settings-sidebar-page-selection-error-border: #f3cccc;
  --settings-sidebar-page-selection-error-text: #b73535;
  --settings-sidebar-page-selection-button-border: #bfd0e5;
  --settings-sidebar-page-selection-button-text: #2f4b71;

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
  border: 1px solid var(--settings-sidebar-page-selection-toggle-border);
  border-radius: 999px;
  background: var(--settings-sidebar-page-selection-toggle-background);
  color: var(--settings-sidebar-page-selection-toggle-text);
  font-weight: 600;
  font-size: 12px;
  cursor: pointer;
}

.page-selection-toggle-compact:has(.page-selection-toggle-input:checked) {
  border-color: var(--settings-sidebar-page-selection-toggle-border-active);
  background: var(--settings-sidebar-page-selection-toggle-background-active);
  color: var(--settings-sidebar-page-selection-toggle-text-active);
}

.total-count,
.page-selection-note {
  color: var(--settings-sidebar-page-selection-muted-text);
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
  color: var(--settings-sidebar-page-selection-summary-text);
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
  border: 1px solid var(--settings-sidebar-page-selection-error-border);
  border-radius: 8px;
  background: var(--color-surface-neutral-muted);
  color: var(--settings-sidebar-page-selection-error-text);
  font-weight: 600;
  font-size: 12px;
  text-align: center;
}

.secondary-button {
  min-height: 38px;
  border: 1px solid var(--settings-sidebar-page-selection-button-border);
  border-radius: 8px;
  background: var(--color-surface-base);
  color: var(--settings-sidebar-page-selection-button-text);
  font-weight: 600;
  font-size: 13px;
}
</style>
