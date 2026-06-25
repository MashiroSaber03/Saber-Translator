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
  display: block;
}
</style>
