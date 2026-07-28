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
  { id: 'basic', label: '基本设置', iconName: 'settings' },
  { id: 'preprocess', label: '图片预处理', iconName: 'image' },
  { id: 'advanced', label: '高级设置', iconName: 'list' },
] satisfies Array<{ id: SettingsTab; label: string; iconName: 'settings' | 'image' | 'list' }>

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
    icon-name="settings"
    aria-label="网页导入设置"
    :expanded="settingsExpanded"
    @update:expanded="$emit('update:settingsExpanded', $event)"
  >
    <ProductSegmentedTabs
      :tabs="settingsTabs"
      :active-tab="activeSettingsTab"
      aria-label="网页导入设置分类"
      class="web-import-settings__tabs"
      @update:active-tab="updateSettingsTab"
    />

    <ProductStatusBanner
      class="web-import-settings__sync-status"
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
  margin-bottom: 16px;
}

.web-import-settings__tabs {
  margin-bottom: 16px;
}

.web-import-settings__tab-content {
  min-height: 0;
  max-block-size: min(52dvh, 480px);
  overflow-y: auto;
  overscroll-behavior: contain;
  scrollbar-gutter: stable;
}
</style>
