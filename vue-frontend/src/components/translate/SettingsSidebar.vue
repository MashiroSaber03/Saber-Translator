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
    <div class="settings-sidebar__card">
      <h2 class="settings-sidebar__title">翻译设置</h2>

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
        ref="fontUploadInput"
        accept=".ttf,.ttc,.otf"
        hidden
        @files-change="handleFontUpload"
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
  --settings-sidebar-card-border: var(--color-border-muted);
  --settings-sidebar-card-shadow: var(--shadow-soft);
  --settings-sidebar-scrollbar-track: var(--color-surface-muted);
  --settings-sidebar-scrollbar-thumb: var(--color-border-default);
  --settings-sidebar-title-divider: var(--color-border-muted);
  --settings-sidebar-title-text: var(--color-text-heading);

  display: flex;
  flex-direction: column;
  direction: rtl;
  z-index: var(--z-fixed-sidebar);
  width: 100%;
  height: 100%;
  padding: 10px 20px 20px;
  overflow-y: auto;
  scrollbar-color: var(--settings-sidebar-scrollbar-thumb) var(--settings-sidebar-scrollbar-track);
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
  background: var(--settings-sidebar-scrollbar-track);
}

.settings-sidebar::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: var(--settings-sidebar-scrollbar-thumb);
}

.settings-sidebar__card {
  margin-bottom: 14px;
  padding: 18px;
  border: 1px solid var(--settings-sidebar-card-border);
  border-radius: 14px;
  background: var(--color-surface-base);
  box-shadow: 0 8px 20px var(--settings-sidebar-card-shadow);
}

.settings-sidebar__title {
  margin: 0 0 14px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--settings-sidebar-title-divider);
  color: var(--settings-sidebar-title-text);
  font-weight: 700;
  font-size: 24px;
  text-align: center;
}

@media (--breakpoint-sidebar-height-compact) {
  .settings-sidebar__title {
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

  .settings-sidebar__card {
    margin-bottom: 0;
    padding: 14px 16px 30px;
  }

  .settings-sidebar__title {
    font-size: 20px;
  }
}
</style>
