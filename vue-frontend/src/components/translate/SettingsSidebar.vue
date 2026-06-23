<script setup lang="ts">
import UiFileInput from '@/components/ui/UiFileInput.vue'
import PageSelectionModal from '@/components/translate/PageSelectionModal.vue'
import { useSettingsSidebar, type ApplySettingsOptions, type SettingsSidebarEmit } from './useSettingsSidebar'
import BookConstraintSection from './settings-sidebar/BookConstraintSection.vue'
import NavigationButtons from './settings-sidebar/NavigationButtons.vue'
import PageSelectionSection from './settings-sidebar/PageSelectionSection.vue'
import TextStyleSection from './settings-sidebar/TextStyleSection.vue'
import WorkflowSection from './settings-sidebar/WorkflowSection.vue'

const emit = defineEmits<SettingsSidebarEmit>()

const {
  showApplyOptions,
  applyOptions,
  isPageSelectionEnabled,
  showPageSelectionModal,
  selectedWorkflowMode,
  rememberWorkflowModeEnabled,
  hasImages,
  totalImages,
  normalizedSelectedPages,
  hasValidPageSelection,
  canUseBookConstraints,
  canGoPrevious,
  canGoNext,
  canRunWorkflow,
  textStyle,
  supportsPageSelectionForCurrentMode,
  isPageSelectionActiveForCurrentMode,
  workflowModeOptions,
  workflowStartLabel,
  workflowContextTag,
  workflowModeTag,
  workflowDescription,
  isDangerousWorkflow,
  fontUploadInput,
  fontSelectOptions,
  layoutDirectionOptions,
  inpaintMethodOptions,
  textAlignOptions,
  createPageSelectionSummary,
  updateFontSize,
  updateAutoFontSize,
  handleFontUpload,
  handleFontSelectChange,
  handleLayoutDirectionChange,
  handleInpaintMethodChange,
  updateTextColor,
  updateLineSpacing,
  updateTextAlign,
  updateUseAutoTextColor,
  updateStrokeEnabled,
  updateStrokeColor,
  updateStrokeWidth,
  updateFillColor,
  toggleApplyOptions,
  toggleSelectAll,
  handleApplyToAll,
  openPageSelectionModal,
  handlePageSelectionConfirm,
  handleWorkflowModeChange,
  handleRememberWorkflowModeChange,
  handleRunWorkflow,
  handleOpenGlossary,
  handleOpenNonTranslate,
} = useSettingsSidebar(emit)

function updateApplyOption(key: keyof ApplySettingsOptions, value: boolean): void {
  applyOptions.value = {
    ...applyOptions.value,
    [key]: value,
  }
}
</script>

<template>
  <aside class="settings-sidebar">
    <div class="settings-card">
      <h2 class="sidebar-title">翻译设置</h2>

      <TextStyleSection
        :apply-options="applyOptions"
        :font-select-options="fontSelectOptions"
        :has-images="hasImages"
        :inpaint-method-options="inpaintMethodOptions"
        :layout-direction-options="layoutDirectionOptions"
        :show-apply-options="showApplyOptions"
        :text-align-options="textAlignOptions"
        :text-style="textStyle"
        @apply="handleApplyToAll"
        @font-select-change="handleFontSelectChange"
        @inpaint-method-change="handleInpaintMethodChange"
        @layout-direction-change="handleLayoutDirectionChange"
        @select-all="toggleSelectAll"
        @text-align-change="updateTextAlign"
        @toggle-apply-options="toggleApplyOptions"
        @update-apply-option="updateApplyOption"
        @update-auto-font-size="updateAutoFontSize"
        @update-fill-color="updateFillColor"
        @update-font-size="updateFontSize"
        @update-line-spacing="updateLineSpacing"
        @update-stroke-color="updateStrokeColor"
        @update-stroke-enabled="updateStrokeEnabled"
        @update-stroke-width="updateStrokeWidth"
        @update-text-color="updateTextColor"
        @update-use-auto-text-color="updateUseAutoTextColor"
      />

      <UiFileInput
        id="fontUpload"
        ref="fontUploadInput"
        accept=".ttf,.ttc,.otf"
        hidden
        @change="handleFontUpload"
      />

      <PageSelectionSection
        v-model:enabled="isPageSelectionEnabled"
        :has-valid-page-selection="hasValidPageSelection"
        :is-active="isPageSelectionActiveForCurrentMode"
        :normalized-selected-pages="normalizedSelectedPages"
        :summary-for="createPageSelectionSummary"
        :supports-page-selection="supportsPageSelectionForCurrentMode"
        :total-images="totalImages"
        @open="openPageSelectionModal"
      />

      <BookConstraintSection
        :can-use-book-constraints="canUseBookConstraints"
        @open-glossary="handleOpenGlossary"
        @open-non-translate="handleOpenNonTranslate"
      />

      <WorkflowSection
        :can-run-workflow="canRunWorkflow"
        :is-dangerous-workflow="isDangerousWorkflow"
        :remember-workflow-mode-enabled="rememberWorkflowModeEnabled"
        :selected-workflow-mode="selectedWorkflowMode"
        :workflow-context-tag="workflowContextTag"
        :workflow-description="workflowDescription"
        :workflow-mode-options="workflowModeOptions"
        :workflow-mode-tag="workflowModeTag"
        :workflow-start-label="workflowStartLabel"
        @remember-change="handleRememberWorkflowModeChange"
        @run="handleRunWorkflow"
        @workflow-mode-change="handleWorkflowModeChange"
      />

      <NavigationButtons
        :can-go-next="canGoNext"
        :can-go-previous="canGoPrevious"
        @next="emit('next')"
        @previous="emit('previous')"
      />
    </div>

    <PageSelectionModal
      :model-value="showPageSelectionModal"
      :selected-pages="normalizedSelectedPages"
      @update:model-value="showPageSelectionModal = $event"
      @confirm="handlePageSelectionConfirm"
    />
  </aside>
</template>

<style scoped>
.settings-sidebar {
  --settings-sidebar-apply-actions-border-default: rgba(255, 255, 255, .24);
  --settings-sidebar-apply-actions-border-strong: #d7e2f2;
  --settings-sidebar-apply-actions-border-muted: #e3ebf6;
  --settings-sidebar-apply-actions-border-subtle: #d4deed;
  --settings-sidebar-apply-actions-border-hover: #94b5e5;
  --settings-sidebar-apply-actions-border-active: #f3cccc;
  --settings-sidebar-apply-actions-border-focus: #d8e3f1;
  --settings-sidebar-apply-actions-border-selected: #d3e1f6;
  --settings-sidebar-apply-actions-border-danger: #ffcaca;
  --settings-sidebar-apply-actions-shadow-default: rgba(22, 37, 58, .16);
  --settings-sidebar-apply-actions-shadow-raised: rgba(62, 169, 74, .24);
  --settings-sidebar-apply-actions-surface-base: #4b89d0;
  --settings-sidebar-apply-actions-surface-raised: #316fb6;
  --settings-sidebar-apply-actions-surface-muted: #c2c9d4;
  --settings-sidebar-apply-actions-surface-subtle: #3f7bc4;
  --settings-sidebar-apply-actions-surface-hover: #2b64a9;
  --settings-sidebar-apply-actions-surface-active: #285d99;
  --settings-sidebar-apply-actions-surface-selected: #2a64a5;
  --settings-sidebar-apply-actions-surface-overlay: #224f82;
  --settings-sidebar-apply-actions-surface-inverse: #f4f8fd;
  --settings-sidebar-apply-actions-surface-contrast: #e9f2ff;
  --settings-sidebar-apply-actions-surface-tint: #4a82ce;
  --settings-sidebar-apply-actions-surface-soft: #f8fbff;
  --settings-sidebar-apply-actions-surface-strong: #3ea94a;
  --settings-sidebar-apply-actions-surface-stronger: #e8f0fd;
  --settings-sidebar-apply-actions-surface-highlight: #ffe7e7;
  --settings-sidebar-apply-actions-surface-highlight-strong: #58ba54;
  --settings-sidebar-apply-actions-text-primary: #6f8099;
  --settings-sidebar-apply-actions-text-secondary: #405473;
  --settings-sidebar-apply-actions-text-muted: #2b5f9d;
  --settings-sidebar-apply-actions-text-subtle: #5d7090;
  --settings-sidebar-apply-actions-text-supporting: #21579c;
  --settings-sidebar-apply-actions-text-disabled: #6f809a;
  --settings-sidebar-apply-actions-text-inverse: #304464;
  --settings-sidebar-apply-actions-text-brand: #b73535;
  --settings-sidebar-apply-actions-text-danger: #4b5f80;
  --settings-sidebar-apply-actions-text-warning: #2d4568;
  --settings-sidebar-apply-actions-text-success: #9f2b2b;
  --settings-sidebar-shell-border-default: #dbe4ef;
  --settings-sidebar-shell-border-strong: #e2e9f2;
  --settings-sidebar-shell-border-muted: #d8e3f1;
  --settings-sidebar-shell-border-subtle: #dfe8f4;
  --settings-sidebar-shell-border-hover: #d3deed;
  --settings-sidebar-shell-border-active: #94b5e5;
  --settings-sidebar-shell-border-focus: #cfdcec;
  --settings-sidebar-shell-border-selected: #d2e2fa;
  --settings-sidebar-shell-border-danger: #d7e2ef;
  --settings-sidebar-shell-shadow-default: rgba(28, 45, 72, .07);
  --settings-sidebar-shell-surface-base: #eef3f9;
  --settings-sidebar-shell-surface-raised: #c7d5e7;
  --settings-sidebar-shell-surface-muted: #f5f8fd;
  --settings-sidebar-shell-surface-subtle: #f4f8fd;
  --settings-sidebar-shell-surface-hover: #4a82ce;
  --settings-sidebar-shell-surface-active: #e9f2ff;
  --settings-sidebar-shell-surface-selected: #edf4ff;
  --settings-sidebar-shell-text-primary: #c7d5e7;
  --settings-sidebar-shell-text-secondary: #eef3f9;
  --settings-sidebar-shell-text-muted: #20314f;
  --settings-sidebar-shell-text-subtle: #d4deeb;
  --settings-sidebar-shell-text-supporting: #24a87a;
  --settings-sidebar-shell-text-disabled: #dc9a2f;
  --settings-sidebar-shell-text-inverse: #273959;
  --settings-sidebar-shell-text-brand: #7d8ba4;
  --settings-sidebar-shell-text-danger: #2f3d56;
  --settings-sidebar-shell-text-warning: #5b6f8e;
  --settings-sidebar-shell-text-success: #21579c;
  --settings-sidebar-shell-text-info: #3a6ea7;
  --settings-sidebar-workflow-border-default: #d8e3f1;
  --settings-sidebar-workflow-border-strong: #bfd0e5;
  --settings-sidebar-workflow-shadow-default: rgba(54, 151, 64, .28);
  --settings-sidebar-workflow-shadow-raised: rgba(214, 66, 66, .24);
  --settings-sidebar-workflow-shadow-floating: rgba(191, 52, 52, .28);
  --settings-sidebar-workflow-surface-base: #d64242;
  --settings-sidebar-workflow-surface-raised: #bf3434;
  --settings-sidebar-workflow-surface-muted: #c1c8d1;
  --settings-sidebar-workflow-surface-subtle: #f8fbff;
  --settings-sidebar-workflow-surface-hover: #eef2f6;
  --settings-sidebar-workflow-surface-active: #eef4fb;
  --settings-sidebar-workflow-surface-selected: #6c7784;
  --settings-sidebar-workflow-surface-overlay: #c2c9d4;
  --settings-sidebar-workflow-surface-inverse: #5a6572;
  --settings-sidebar-workflow-text-primary: #5c6f8f;
  --settings-sidebar-workflow-text-secondary: #273959;
  --settings-sidebar-workflow-text-muted: #62748f;
  --settings-sidebar-workflow-text-subtle: #2f4b71;
  --settings-sidebar-workflow-text-supporting: #8b97a7;

  display: flex;
  flex-direction: column;
  direction: rtl;
  z-index: var(--z-fixed-sidebar);
  width: 100%;
  height: 100%;
  padding: 10px 20px 20px;
  overflow-y: auto;
  scrollbar-color: var(--settings-sidebar-shell-text-primary) var(--settings-sidebar-shell-text-secondary);
  scrollbar-width: thin;
}

.settings-sidebar > * {
  direction: ltr;
}

.settings-sidebar::-webkit-scrollbar {
  width: 8px;
}

.settings-sidebar::-webkit-scrollbar-track {
  border-radius: 999px;
  background: var(--settings-sidebar-shell-surface-base);
}

.settings-sidebar::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: var(--settings-sidebar-shell-surface-raised);
}

.settings-card {
  margin-bottom: 14px;
  padding: 18px;
  border: 1px solid var(--settings-sidebar-shell-border-default);
  border-radius: 14px;
  background: var(--color-surface-base);
  box-shadow: 0 8px 20px var(--settings-sidebar-shell-shadow-default);
}

.sidebar-title {
  margin: 0 0 14px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--settings-sidebar-shell-border-strong);
  color: var(--settings-sidebar-shell-text-muted);
  font-weight: 700;
  font-size: 24px;
  text-align: center;
}

@media (--breakpoint-sidebar-height-compact) {
  .sidebar-title {
    font-size: 22px;
  }
}

@media (--breakpoint-md-down) {
  .settings-sidebar {
    order: 2;
    z-index: auto;
    width: 100%;
    height: auto;
    max-height: none;
    margin-top: 0;
    padding: 0;
    overflow: visible;
    direction: ltr;
  }

  .settings-card {
    margin-bottom: 0;
    padding: 14px 16px 30px;
  }

  .sidebar-title {
    font-size: 20px;
  }
}
</style>
