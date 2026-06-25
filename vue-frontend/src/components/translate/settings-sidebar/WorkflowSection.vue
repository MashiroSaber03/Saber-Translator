<script setup lang="ts">
import CustomSelect from '@/components/common/CustomSelect.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'

type SelectOption = { label: string; value: string | number }

defineProps<{
  canRunWorkflow: boolean
  isDangerousWorkflow: boolean
  rememberWorkflowModeEnabled: boolean
  selectedWorkflowMode: string
  workflowContextTag: string
  workflowDescription: string
  workflowModeOptions: SelectOption[]
  workflowModeTag: string
  workflowStartLabel: string
}>()

defineEmits<{
  (event: 'rememberChange', value: boolean): void
  (event: 'run'): void
  (event: 'workflowModeChange', value: string | number): void
}>()
</script>

<template>
  <div class="action-buttons workflow-controls">
    <div class="settings-sidebar__field">
      <label for="workflowModeSelect">操作模式:</label>
      <CustomSelect
        id="workflowModeSelect"
        :model-value="selectedWorkflowMode"
        :options="workflowModeOptions"
        fit
        variant="workflow"
        @change="$emit('workflowModeChange', $event)"
      />
      <UiCheckbox
        :model-value="rememberWorkflowModeEnabled"
        class="remember-workflow-mode-toggle"
        label="记住操作模式"
        @change="$emit('rememberChange', $event)"
      />
    </div>
    <div class="workflow-meta">
      <span class="workflow-chip">{{ workflowContextTag }}</span>
      <span class="workflow-chip" :class="{ 'danger-chip': isDangerousWorkflow }">
        {{ workflowModeTag }}
      </span>
    </div>
    <UiButton
      id="runWorkflowButton"
      variant="toolbar"
      class="settings-button workflow-run-button"
      :class="{ 'danger-button': isDangerousWorkflow }"
      :disabled="!canRunWorkflow"
      @click="$emit('run')"
    >
      {{ workflowStartLabel }}
    </UiButton>
    <div class="workflow-description">
      {{ workflowDescription }}
    </div>
  </div>
</template>

<style scoped>
.action-buttons {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 14px;
}

.workflow-controls {
  --settings-sidebar-workflow-panel-border: #d8e3f1;
  --settings-sidebar-workflow-panel-background: #f8fbff;
  --settings-sidebar-workflow-label-text: #2f3d56;
  --settings-sidebar-workflow-remember-text: #4b5f80;
  --settings-sidebar-workflow-chip-border: #d3e1f6;
  --settings-sidebar-workflow-chip-background: #e8f0fd;
  --settings-sidebar-workflow-chip-text: #2d4568;
  --settings-sidebar-workflow-danger-chip-border: #ffcaca;
  --settings-sidebar-workflow-danger-chip-background: #ffe7e7;
  --settings-sidebar-workflow-danger-chip-text: #9f2b2b;
  --settings-sidebar-workflow-run-start: #3ea94a;
  --settings-sidebar-workflow-run-end: #58ba54;
  --settings-sidebar-workflow-run-shadow: rgba(62, 169, 74, .24);
  --settings-sidebar-workflow-run-hover-shadow: rgba(54, 151, 64, .28);
  --settings-sidebar-workflow-danger-run-start: #d64242;
  --settings-sidebar-workflow-danger-run-end: #bf3434;
  --settings-sidebar-workflow-danger-run-shadow: rgba(214, 66, 66, .24);
  --settings-sidebar-workflow-danger-run-hover-shadow: rgba(191, 52, 52, .28);
  --settings-sidebar-workflow-disabled-background: #c1c8d1;
  --settings-sidebar-workflow-description-text: #5c6f8f;

  padding: 12px;
  border: 1px solid var(--settings-sidebar-workflow-panel-border);
  border-radius: 12px;
  background: var(--settings-sidebar-workflow-panel-background);
}

.settings-sidebar__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 0;
}

.settings-sidebar__field label {
  margin-bottom: 6px;
  color: var(--settings-sidebar-workflow-label-text);
  font-weight: 600;
  font-size: 13px;
}

.remember-workflow-mode-toggle {
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

.workflow-meta {
  display: flex;
  align-items: center;
  gap: 8px;
}

.workflow-chip {
  display: inline-flex;
  align-items: center;
  height: 24px;
  padding: 0 9px;
  border: 1px solid var(--settings-sidebar-workflow-chip-border);
  border-radius: 999px;
  background: var(--settings-sidebar-workflow-chip-background);
  color: var(--settings-sidebar-workflow-chip-text);
  font-weight: 600;
  font-size: 12px;
}

.workflow-chip.danger-chip {
  border-color: var(--settings-sidebar-workflow-danger-chip-border);
  background: var(--settings-sidebar-workflow-danger-chip-background);
  color: var(--settings-sidebar-workflow-danger-chip-text);
}

.workflow-run-button {
  min-height: 54px;
  border: none;
  border-radius: 10px;
  background: linear-gradient(135deg, var(--settings-sidebar-workflow-run-start) 0%, var(--settings-sidebar-workflow-run-end) 100%);
  box-shadow: 0 8px 16px var(--settings-sidebar-workflow-run-shadow);
  color: var(--color-text-inverse);
  font-weight: 700;
  font-size: 16px;
  cursor: pointer;
  transition:
    transform 0.2s ease,
    box-shadow 0.2s ease;
}

.workflow-run-button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 10px 18px var(--settings-sidebar-workflow-run-hover-shadow);
}

.workflow-run-button.danger-button {
  background: linear-gradient(135deg, var(--settings-sidebar-workflow-danger-run-start) 0%, var(--settings-sidebar-workflow-danger-run-end) 100%);
  box-shadow: 0 8px 16px var(--settings-sidebar-workflow-danger-run-shadow);
}

.workflow-run-button.danger-button:hover:not(:disabled) {
  box-shadow: 0 10px 18px var(--settings-sidebar-workflow-danger-run-hover-shadow);
}

.workflow-run-button:disabled {
  background: var(--settings-sidebar-workflow-disabled-background);
  box-shadow: none;
  cursor: not-allowed;
}

.workflow-description {
  color: var(--settings-sidebar-workflow-description-text);
  font-size: 13px;
  line-height: 1.45;
}

@media (--breakpoint-md-down) {
  .workflow-controls {
    margin-top: 8px;
  }
}
</style>
