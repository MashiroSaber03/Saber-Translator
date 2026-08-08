<script setup lang="ts">
import { computed } from 'vue'
import ProductChipList, { type ProductChipItem } from '@/components/product/ProductChipList.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { UiSelectOption, UiSelectValue } from '@/components/ui/selectTypes'

const props = defineProps<{
  canRunWorkflow: boolean
  isDangerousWorkflow: boolean
  rememberWorkflowModeEnabled: boolean
  selectedWorkflowMode: string
  workflowContextTag: string
  workflowDescription: string
  workflowModeOptions: UiSelectOption[]
  workflowModeTag: string
  workflowStartLabel: string
}>()

defineEmits<{
  (event: 'rememberChange', value: boolean): void
  (event: 'run'): void
  (event: 'workflowModeChange', value: UiSelectValue): void
}>()

const workflowChipItems = computed<ProductChipItem[]>(() => [
  {
    id: 'workflow-context',
    label: props.workflowContextTag,
    tone: 'neutral',
  },
  {
    id: 'workflow-mode',
    label: props.workflowModeTag,
    tone: props.isDangerousWorkflow ? 'danger' : 'primary',
  },
])
</script>

<template>
  <div class="workflow-section">
    <UiField
      class="workflow-section__mode-field"
      variant="settings"
      label="操作模式"
      control-id="workflowModeSelect"
    >
      <UiSelect
        id="workflowModeSelect"
        :model-value="selectedWorkflowMode"
        :options="workflowModeOptions"
        @change="$emit('workflowModeChange', $event)"
      />
      <UiCheckbox
        input-id="rememberWorkflowMode"
        :model-value="rememberWorkflowModeEnabled"
        class="workflow-section__remember-toggle"
        label="记住操作模式"
        @change="$emit('rememberChange', $event)"
      />
    </UiField>
    <ProductChipList
      class="workflow-section__meta"
      aria-label="当前操作模式"
      :items="workflowChipItems"
    />
    <UiButton
      class="workflow-section__run-action"
      :class="{ 'workflow-section__run-action--safe': !isDangerousWorkflow }"
      :variant="isDangerousWorkflow ? 'danger' : 'primary'"
      size="lg"
      block
      :disabled="!canRunWorkflow"
      @click="$emit('run')"
    >
      {{ workflowStartLabel }}
    </UiButton>
    <div class="workflow-section__description">
      {{ workflowDescription }}
    </div>
  </div>
</template>

<style scoped>
.workflow-section {
  --settings-sidebar-workflow-panel-border: var(--color-border-muted);
  --settings-sidebar-workflow-panel-background: var(--color-surface-quiet);
  --settings-sidebar-workflow-remember-text: var(--color-text-secondary);
  --settings-sidebar-workflow-description-text: var(--color-text-secondary);
  --settings-sidebar-workflow-field-label: var(--color-text-default);
  --settings-sidebar-workflow-chip-border: var(--color-border-muted);
  --settings-sidebar-workflow-chip-background: var(--color-surface-neutral-muted);
  --settings-sidebar-workflow-chip-text: var(--color-text-supporting);

  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 14px;
  padding: 12px;
  border: 1px solid var(--settings-sidebar-workflow-panel-border);
  border-radius: 12px;
  background: var(--settings-sidebar-workflow-panel-background);
}

.workflow-section__mode-field {
  --ui-field-label-color: var(--settings-sidebar-workflow-field-label);
  --ui-field-label-font-size: 13px;
  --ui-field-label-font-weight: 600;

  margin-bottom: 0;
}

.workflow-section__remember-toggle {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
  margin-bottom: 0;
  color: var(--settings-sidebar-workflow-remember-text);
  font-weight: 600;
  font-size: 13px;
  cursor: pointer;
}

.workflow-section__meta {
  --product-chip-list-chip-font-weight: 600;
  --product-chip-list-neutral-border: var(--settings-sidebar-workflow-chip-border);
  --product-chip-list-neutral-background: var(--settings-sidebar-workflow-chip-background);
  --product-chip-list-neutral-text: var(--settings-sidebar-workflow-chip-text);
  --product-chip-list-primary-border: var(--settings-sidebar-workflow-chip-border);
  --product-chip-list-primary-background: var(--settings-sidebar-workflow-chip-background);
  --product-chip-list-primary-text: var(--settings-sidebar-workflow-chip-text);

  margin-top: 2px;
}

.workflow-section__run-action {
  min-height: 54px;
  font-weight: 700;
}

.workflow-section__run-action--safe {
  background: linear-gradient(135deg, var(--color-action-success) 0%, var(--color-status-success) 100%);
  box-shadow: 0 8px 16px color-mix(in srgb, var(--color-action-success) 24%, transparent);
}

.workflow-section__run-action--safe:hover:not(:disabled) {
  background: linear-gradient(135deg, var(--color-status-success) 0%, var(--color-action-success) 100%);
  box-shadow: 0 10px 18px color-mix(in srgb, var(--color-action-success) 28%, transparent);
  transform: translateY(-1px);
}

.workflow-section__description {
  color: var(--settings-sidebar-workflow-description-text);
  font-size: 13px;
  line-height: 1.45;
}

@media (--breakpoint-md-down) {
  .workflow-section {
    margin-top: 8px;
  }
}
</style>
