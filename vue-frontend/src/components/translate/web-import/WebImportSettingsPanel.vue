<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import ProductCollapsibleSection from '@/components/product/ProductCollapsibleSection.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import WebImportPreprocessSettings from '../WebImportPreprocessSettings.vue'
import WebImportBasicSettingsPanel from './WebImportBasicSettingsPanel.vue'
import WebImportAdvancedSettingsPanel from './WebImportAdvancedSettingsPanel.vue'
import type { UiSelectOption } from '@/components/ui/selectTypes'
import type { WebImportSettings } from '@/types/webImport'
import type { WebImportSettingsActions } from './webImportSettingsActions'

type SettingsTab = 'basic' | 'preprocess' | 'advanced'

defineProps<{
  activeSettingsTab: SettingsTab
  agentProviderOptions: UiSelectOption[]
  draftSettings: WebImportSettings
  hasUnsavedSettings: boolean
  hasAgentCredential?: boolean
  hasFirecrawlCredential?: boolean
  isFetchingModels: boolean
  isSavingSettings: boolean
  modelList: string[]
  modelListOptions: UiSelectOption[]
  providerRequiresApiKey: (provider: string) => boolean
  settingsActions: WebImportSettingsActions
  settingsExpanded: boolean
  showCustomUrl: boolean
  supportsFetchModels: boolean
  testingAgent: boolean
  testingFirecrawl: boolean
}>()

const emit = defineEmits<{
  (event: 'discard-settings'): void
  (event: 'fetch-models'): void
  (event: 'reset-prompt'): void
  (event: 'save-settings'): void
  (event: 'test-agent'): void
  (event: 'test-firecrawl'): void
  (event: 'update:activeSettingsTab', value: SettingsTab): void
  (event: 'update:settingsExpanded', value: boolean): void
}>()

const settingsTabs = [
  { id: 'basic', label: '基本设置' },
  { id: 'preprocess', label: '图片预处理' },
  { id: 'advanced', label: '高级设置' },
] satisfies Array<{ id: SettingsTab; label: string }>

const settingsTabIds = settingsTabs.map(tab => tab.id)

function isSettingsTab(tabId: string): tabId is SettingsTab {
  return settingsTabIds.some(id => id === tabId)
}

function updateSettingsTab(tabId: string): void {
  if (!isSettingsTab(tabId)) return
  emit('update:activeSettingsTab', tabId)
}
</script>

<template>
  <ProductCollapsibleSection
    class="web-import-settings-section"
    title="设置"
    hint="点击展开配置"
    text-toggle
    aria-label="网页导入设置"
    :expanded="settingsExpanded"
    @update:expanded="$emit('update:settingsExpanded', $event)"
  >
    <template #icon>⚙️</template>
    <ProductSegmentedTabs
      :tabs="settingsTabs"
      :active-tab="activeSettingsTab"
      aria-label="网页导入设置分类"
      class="web-import-settings__tabs"
      @update:active-tab="updateSettingsTab"
    />

    <ProductStatusBanner
      class="web-import-settings__sync-status"
      :class="{ 'web-import-settings__sync-status--dirty': hasUnsavedSettings }"
      :tone="hasUnsavedSettings ? 'warning' : 'success'"
      role="status"
      aria-live="polite"
    >
      {{ hasUnsavedSettings ? '有未保存的修改' : '设置已同步' }}
      <template #actions>
        <UiButton
          variant="secondary"
          size="sm"
          :disabled="!hasUnsavedSettings || isSavingSettings"
          @click="$emit('discard-settings')"
        >
          取消修改
        </UiButton>
        <UiButton
          variant="primary"
          size="sm"
          :disabled="!hasUnsavedSettings || isSavingSettings"
          @click="$emit('save-settings')"
        >
          {{ isSavingSettings ? '保存中...' : '保存设置' }}
        </UiButton>
      </template>
    </ProductStatusBanner>

    <div v-show="activeSettingsTab === 'basic'" class="web-import-settings__tab-content">
      <WebImportBasicSettingsPanel
        :agent-provider-options="agentProviderOptions"
        :draft-settings="draftSettings"
        :has-agent-credential="hasAgentCredential"
        :has-firecrawl-credential="hasFirecrawlCredential"
        :is-fetching-models="isFetchingModels"
        :model-list="modelList"
        :model-list-options="modelListOptions"
        :provider-requires-api-key="providerRequiresApiKey"
        :settings-actions="settingsActions"
        :show-custom-url="showCustomUrl"
        :supports-fetch-models="supportsFetchModels"
        :testing-agent="testingAgent"
        :testing-firecrawl="testingFirecrawl"
        @fetch-models="$emit('fetch-models')"
        @reset-prompt="$emit('reset-prompt')"
        @test-agent="$emit('test-agent')"
        @test-firecrawl="$emit('test-firecrawl')"
      />
    </div>

    <div v-show="activeSettingsTab === 'preprocess'" class="web-import-settings__tab-content">
      <WebImportPreprocessSettings
        :draft-settings="draftSettings"
        :settings-actions="settingsActions"
      />
    </div>

    <div v-show="activeSettingsTab === 'advanced'" class="web-import-settings__tab-content">
      <WebImportAdvancedSettingsPanel
        :draft-settings="draftSettings"
        :settings-actions="settingsActions"
      />
    </div>
  </ProductCollapsibleSection>
</template>

<style scoped>
.web-import-settings-section {
  margin-bottom: 16px;
}

.web-import-settings__sync-status {
  --product-status-banner-align-items: center;
  --product-status-banner-icon-display: none;
  --product-status-banner-padding: 12px 14px;
  --product-status-banner-radius: 10px;
  --product-status-banner-min-height: 63px;
  --product-status-banner-background: var(--color-surface-muted);
  --product-status-banner-border: 1px solid var(--color-border-muted);
  --product-status-banner-body-color: var(--color-status-success);
  --product-status-banner-body-font-size: 13px;
  --product-status-banner-body-font-weight: 500;
  --ui-button-sm-padding: 8px 14px;
  --ui-button-primary-background: var(--color-action-primary);
  --ui-button-primary-hover-background: var(--color-action-primary-hover);
  --ui-button-primary-shadow: none;
  --ui-button-primary-disabled-background: var(--color-action-primary);
  --ui-button-primary-disabled-opacity: 0.6;

  margin-bottom: 16px;
}

.web-import-settings__sync-status--dirty {
  --product-status-banner-body-color: var(--color-status-warning-hover);
}

.web-import-settings__tabs {
  --product-segmented-tabs-padding: 0 0 8px;
  --product-segmented-tabs-border: var(--color-border-muted);
  --product-segmented-tabs-radius: 0;
  --product-segmented-tabs-background: transparent;
  --product-segmented-tabs-shadow: none;
  --product-segmented-tabs-tab-padding: 8px 16px;
  --product-segmented-tabs-tab-radius: 6px 6px 0 0;
  --product-segmented-tabs-tab-flex: 0 0 auto;
  --product-segmented-tabs-tab-min-width: 0;
  --product-segmented-tabs-tab-gap: 0;
  --product-segmented-tabs-tab-font-size: 13px;
  --product-segmented-tabs-tab-font-weight: 400;
  --product-segmented-tabs-active-background: var(--color-surface-muted);
  --product-segmented-tabs-active-shadow: none;
  --product-segmented-tabs-active-font-weight: 500;

  margin-bottom: 16px;
  border-width: 0 0 1px;
}

.web-import-settings__tab-content {
  min-height: 0;
  max-block-size: min(52dvh, 480px);
  overflow-y: auto;
  overscroll-behavior: contain;
}
</style>
