<script setup lang="ts">
import './WebImportModal.global.styles.css'
import BaseModal from '@/components/common/BaseModal.vue'
import { useWebImportModal } from './useWebImportModal'
import WebImportExtractBar from './web-import/WebImportExtractBar.vue'
import WebImportFooterActions from './web-import/WebImportFooterActions.vue'
import WebImportLogsPanel from './web-import/WebImportLogsPanel.vue'
import WebImportResultsGrid from './web-import/WebImportResultsGrid.vue'
import WebImportSettingsPanel from './web-import/WebImportSettingsPanel.vue'

const {
  activeSettingsTab,
  agentProviderOptions,
  checkingSupport,
  downloadProgress,
  downloadProgressPercent,
  draftSettings,
  engineDisplayName,
  error,
  extractResult,
  galleryDLAvailable,
  galleryDLSupported,
  getPreviewUrl,
  handleClose,
  handleDiscardSettings,
  handleExtract,
  handleFetchModels,
  handleImport,
  handleResetPrompt,
  handleSaveSettings,
  handleTestAgent,
  handleTestFirecrawl,
  hasUnsavedSettings,
  isAllSelected,
  isFetchingModels,
  isProcessing,
  isSavingSettings,
  isVisible,
  logs,
  logsExpanded,
  modelList,
  modelListOptions,
  providerRequiresApiKey,
  selectedCount,
  selectedEngine,
  selectedPages,
  settingsExpanded,
  showAgentKey,
  showAgentLogs,
  showCustomUrl,
  showFirecrawlKey,
  status,
  supportsFetchModels,
  testingAgent,
  testingFirecrawl,
  toggleAll,
  togglePage,
  urlInput,
  webImportStore,
} = useWebImportModal()
</script>

<template>
  <BaseModal
    :model-value="isVisible"
    title="🌐 从网页导入漫画"
    size="large"
    custom-class="web-import-modal"
    max-width="800px"
    footer-gap="12px"
    footer-padding="16px 20px"
    footer-border="1px solid var(--color-border-muted, var(--color-border-soft))"
    :close-on-overlay="!isProcessing"
    :close-on-esc="!isProcessing"
    @close="handleClose"
  >
    <div class="web-import-modal-body">
      <WebImportExtractBar
        v-model:selected-engine="selectedEngine"
        v-model:url-input="urlInput"
        :checking-support="checkingSupport"
        :gallery-d-l-available="galleryDLAvailable"
        :gallery-d-l-supported="galleryDLSupported"
        :is-processing="isProcessing"
        :status="status"
        @extract="handleExtract"
      />

      <WebImportSettingsPanel
        v-model:active-settings-tab="activeSettingsTab"
        v-model:settings-expanded="settingsExpanded"
        v-model:show-agent-key="showAgentKey"
        v-model:show-firecrawl-key="showFirecrawlKey"
        :agent-provider-options="agentProviderOptions"
        :draft-settings="draftSettings"
        :has-unsaved-settings="hasUnsavedSettings"
        :is-fetching-models="isFetchingModels"
        :is-saving-settings="isSavingSettings"
        :model-list="modelList"
        :model-list-options="modelListOptions"
        :provider-requires-api-key="providerRequiresApiKey"
        :show-custom-url="showCustomUrl"
        :supports-fetch-models="supportsFetchModels"
        :testing-agent="testingAgent"
        :testing-firecrawl="testingFirecrawl"
        :web-import-store="webImportStore"
        @discard-settings="handleDiscardSettings"
        @fetch-models="handleFetchModels"
        @reset-prompt="handleResetPrompt"
        @save-settings="() => handleSaveSettings()"
        @test-agent="handleTestAgent"
        @test-firecrawl="handleTestFirecrawl"
      />

      <WebImportLogsPanel
        v-if="showAgentLogs"
        :expanded="logsExpanded"
        :logs="logs"
        :status="status"
        @toggle="logsExpanded = !logsExpanded"
      />

      <WebImportResultsGrid
        :download-progress="downloadProgress"
        :download-progress-percent="downloadProgressPercent"
        :engine-display-name="engineDisplayName"
        :error="error"
        :extract-result="extractResult"
        :is-all-selected="isAllSelected"
        :preview-url-for="getPreviewUrl"
        :selected-count="selectedCount"
        :selected-pages="selectedPages"
        :status="status"
        @toggle-all="toggleAll"
        @toggle-page="togglePage"
      />
    </div>

    <template #footer>
      <WebImportFooterActions
        :extract-result="extractResult"
        :is-processing="isProcessing"
        :selected-count="selectedCount"
        :status="status"
        @close="handleClose"
        @import="handleImport"
      />
    </template>
  </BaseModal>
</template>

<style scoped>
.web-import-modal-body {
  --web-import-modal-extract-border-default: #ffe0a0;
  --web-import-modal-extract-border-strong: #e6e6e6;
  --web-import-modal-extract-surface-base: #4a90d9;
  --web-import-modal-extract-surface-raised: #3a7fc8;
  --web-import-modal-extract-surface-muted: #f9f9f9;
  --web-import-modal-extract-surface-subtle: #efefef;
  --web-import-modal-extract-surface-hover: #fafafa;
  --web-import-modal-extract-text-muted: #b26a00;
  --web-import-modal-extract-text-primary: #28a745;
  --web-import-modal-extract-text-secondary: #856404;
  --web-import-modal-extract-text-subtle: #2f7d32;
  --web-import-modal-settings-border-default: #ffc0c0;
  --web-import-modal-settings-shadow-default: rgba(74, 144, 217, .2);
  --web-import-modal-settings-surface-base: #f9f9f9;
  --web-import-modal-settings-surface-muted: #eee;
  --web-import-modal-settings-surface-raised: #1e1e1e;
  --web-import-modal-settings-text-brand: #c00;
  --web-import-modal-settings-text-disabled: #dcdcaa;
  --web-import-modal-settings-text-inverse: #f1f5f9;
  --web-import-modal-settings-text-muted: #ce9178;
  --web-import-modal-settings-text-secondary: #ccc;
  --web-import-modal-settings-text-subtle: #ec4899;
  --web-import-modal-settings-text-supporting: #818cf8;
}
</style>
