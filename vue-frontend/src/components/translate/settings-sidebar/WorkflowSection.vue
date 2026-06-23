<script setup lang="ts">
import CustomSelect from '@/components/common/CustomSelect.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiInput from '@/components/ui/UiInput.vue'

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
  (event: 'rememberChange', value: Event): void
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
      <label class="remember-workflow-mode-toggle">
        <UiInput
          id="rememberWorkflowModeCheckbox"
          type="checkbox"
          :checked="rememberWorkflowModeEnabled"
          @change="$emit('rememberChange', $event)"
        />
        <span>记住操作模式</span>
      </label>
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
  padding: 12px;
  border: 1px solid var(--settings-sidebar-apply-actions-border-focus);
  border-radius: 12px;
  background: var(--settings-sidebar-apply-actions-surface-soft);
}

.settings-sidebar__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 0;
}

.settings-sidebar__field label {
  margin-bottom: 6px;
  color: var(--settings-sidebar-shell-text-danger);
  font-weight: 600;
  font-size: 13px;
}

.remember-workflow-mode-toggle {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
  margin-bottom: 0;
  color: var(--settings-sidebar-apply-actions-text-danger);
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
  border: 1px solid var(--settings-sidebar-apply-actions-border-selected);
  border-radius: 999px;
  background: var(--settings-sidebar-apply-actions-surface-stronger);
  color: var(--settings-sidebar-apply-actions-text-warning);
  font-weight: 600;
  font-size: 12px;
}

.workflow-chip.danger-chip {
  border-color: var(--settings-sidebar-apply-actions-border-danger);
  background: var(--settings-sidebar-apply-actions-surface-highlight);
  color: var(--settings-sidebar-apply-actions-text-success);
}

.workflow-run-button {
  min-height: 54px;
  border: none;
  border-radius: 10px;
  background: linear-gradient(135deg, var(--settings-sidebar-apply-actions-surface-strong) 0%, var(--settings-sidebar-apply-actions-surface-highlight-strong) 100%);
  box-shadow: 0 8px 16px var(--settings-sidebar-apply-actions-shadow-raised);
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
  box-shadow: 0 10px 18px var(--settings-sidebar-workflow-shadow-default);
}

.workflow-run-button.danger-button {
  background: linear-gradient(135deg, var(--settings-sidebar-workflow-surface-base) 0%, var(--settings-sidebar-workflow-surface-raised) 100%);
  box-shadow: 0 8px 16px var(--settings-sidebar-workflow-shadow-raised);
}

.workflow-run-button.danger-button:hover:not(:disabled) {
  box-shadow: 0 10px 18px var(--settings-sidebar-workflow-shadow-floating);
}

.workflow-run-button:disabled {
  background: var(--settings-sidebar-workflow-surface-muted);
  box-shadow: none;
  cursor: not-allowed;
}

.workflow-description {
  color: var(--settings-sidebar-workflow-text-primary);
  font-size: 13px;
  line-height: 1.45;
}

@media (--breakpoint-md-down) {
  .workflow-controls {
    margin-top: 8px;
  }
}
</style>
