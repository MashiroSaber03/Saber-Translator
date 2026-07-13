<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import UiPasswordField from '@/components/ui/UiPasswordField.vue'
import AiProviderCredentialFields from '@/components/settings/AiProviderCredentialFields.vue'
import AiProviderSelectField from '@/components/settings/AiProviderSelectField.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import type { UiSelectOption } from '@/components/ui/selectTypes'
import type { WebImportSettings } from '@/types/webImport'
import type { WebImportSettingsActions } from './webImportSettingsActions'

defineProps<{
  agentProviderOptions: UiSelectOption[]
  draftSettings: WebImportSettings
  isFetchingModels: boolean
  modelList: string[]
  modelListOptions: UiSelectOption[]
  providerRequiresApiKey: (provider: string) => boolean
  settingsActions: WebImportSettingsActions
  showCustomUrl: boolean
  supportsFetchModels: boolean
  testingAgent: boolean
  testingFirecrawl: boolean
}>()

defineEmits<{
  (event: 'fetch-models'): void
  (event: 'reset-prompt'): void
  (event: 'test-agent'): void
  (event: 'test-firecrawl'): void
}>()
</script>

<template>
  <ProductFormSection>
    <template #title>Firecrawl 配置</template>
    <UiField variant="settings" label="API Key" control-id="webImportFirecrawlApiKey">
      <UiPasswordField
        input-id="webImportFirecrawlApiKey"
        :model-value="draftSettings.firecrawl.apiKey"
        placeholder="fc-xxxxxxxxxxxxxxxx"
        show-label="显示 Firecrawl API Key"
        hide-label="隐藏 Firecrawl API Key"
        @update:model-value="settingsActions.setFirecrawlApiKey"
      />
    </UiField>

    <ProductActionRow aria-label="Firecrawl 操作" justify="start">
      <UiButton
        variant="secondary"
        :disabled="testingFirecrawl || !draftSettings.firecrawl.apiKey"
        @click="$emit('test-firecrawl')"
      >
        {{ testingFirecrawl ? '测试中...' : '测试连接' }}
      </UiButton>
    </ProductActionRow>
  </ProductFormSection>

  <ProductFormSection>
    <template #title>AI Agent 配置</template>

    <AiProviderSelectField
      :model-value="draftSettings.agent.provider"
      input-id="webImportAgentProvider"
      :options="agentProviderOptions"
      @update:model-value="settingsActions.setAgentProvider"
    />

    <AiProviderCredentialFields
      :api-key="draftSettings.agent.apiKey"
      api-key-input-id="webImportAgentApiKey"
      :base-url="draftSettings.agent.customBaseUrl"
      base-url-input-id="webImportAgentBaseUrl"
      :show-api-key="providerRequiresApiKey(draftSettings.agent.provider)"
      :show-base-url="showCustomUrl"
      api-key-placeholder="sk-xxxxxxxxxxxxxxxx"
      api-key-show-label="显示 AI Agent API Key"
      api-key-hide-label="隐藏 AI Agent API Key"
      base-url-label="自定义 API 地址"
      base-url-placeholder="https://api.example.com/v1"
      @update:api-key="settingsActions.setAgentApiKey"
      @update:base-url="settingsActions.setAgentBaseUrl"
    />

    <UiField variant="settings" label="模型名称" control-id="webImportAgentModelName">
      <UiModelPicker
        input-id="webImportAgentModelName"
        :model-value="draftSettings.agent.modelName"
        placeholder="gpt-4o-mini"
        :show-fetch="supportsFetchModels"
        :fetching="isFetchingModels"
        :fetch-disabled="isFetchingModels"
        :options="modelListOptions"
        :model-count="modelList.length"
        @update:model-value="value => settingsActions.setAgentModelName(String(value))"
        @fetch="$emit('fetch-models')"
      />
    </UiField>

    <UiFormGrid>
      <UiField variant="settings" control="checkbox">
        <UiCheckbox
          :model-value="draftSettings.agent.forceJsonOutput"
          label="强制 JSON 格式"
          @change="settingsActions.setAgentForceJsonOutput"
        />
      </UiField>
      <UiField variant="settings" control="checkbox">
        <UiCheckbox
          :model-value="draftSettings.agent.useStream"
          label="流式调用"
          @change="settingsActions.setAgentUseStream"
        />
      </UiField>
    </UiFormGrid>

    <ProductActionRow aria-label="AI Agent 操作" justify="start">
      <UiButton
        variant="secondary"
        block
        :disabled="testingAgent || (providerRequiresApiKey(draftSettings.agent.provider) && !draftSettings.agent.apiKey)"
        @click="$emit('test-agent')"
      >
        {{ testingAgent ? '测试中...' : '测试 Agent 连接' }}
      </UiButton>
    </ProductActionRow>
  </ProductFormSection>

  <ProductFormSection>
    <template #title>提取设置</template>

    <UiField variant="settings" label="提取提示词" control-id="webImportExtractionPrompt">
      <UiTextarea
        id="webImportExtractionPrompt"
        :model-value="draftSettings.extraction.prompt"
        variant="panel"
        rows="6"
        placeholder="输入提取提示词..."
        @update:model-value="settingsActions.setExtractionPrompt"
      />
      <ProductActionRow aria-label="提取提示词操作" justify="start">
        <UiButton variant="secondary" size="sm" @click="$emit('reset-prompt')">重置为默认</UiButton>
      </ProductActionRow>
    </UiField>

    <UiField variant="settings" label="最大迭代次数" control-id="webImportMaxIterations">
      <UiNumberField
        input-id="webImportMaxIterations"
        :model-value="draftSettings.extraction.maxIterations"
        :min="1"
        :max="20"
        @update:model-value="settingsActions.setExtractionMaxIterations"
      />
    </UiField>
  </ProductFormSection>

  <ProductFormSection>
    <template #title>下载设置</template>

    <UiFormGrid>
      <UiField variant="settings" label="并发数" control-id="webImportDownloadConcurrency">
        <UiNumberField
          input-id="webImportDownloadConcurrency"
          :model-value="draftSettings.download.concurrency"
          :min="1"
          :max="10"
          @update:model-value="settingsActions.setDownloadConcurrency"
        />
      </UiField>

      <UiField variant="settings" label="超时 (秒)" control-id="webImportDownloadTimeout">
        <UiNumberField
          input-id="webImportDownloadTimeout"
          :model-value="draftSettings.download.timeout"
          :min="5"
          :max="120"
          @update:model-value="settingsActions.setDownloadTimeout"
        />
      </UiField>

      <UiField variant="settings" label="重试次数" control-id="webImportDownloadRetries">
        <UiNumberField
          input-id="webImportDownloadRetries"
          :model-value="draftSettings.download.retries"
          :min="0"
          :max="5"
          @update:model-value="settingsActions.setDownloadRetries"
        />
      </UiField>

      <UiField variant="settings" label="下载间隔 (ms)" control-id="webImportDownloadDelay">
        <UiNumberField
          input-id="webImportDownloadDelay"
          :model-value="draftSettings.download.delay"
          :min="0"
          :max="2000"
          :step="100"
          @update:model-value="settingsActions.setDownloadDelay"
        />
      </UiField>
    </UiFormGrid>

    <UiField variant="settings" control="checkbox">
      <UiCheckbox
        :model-value="draftSettings.download.useReferer"
        label="自动添加 Referer"
        @change="settingsActions.setDownloadUseReferer"
      />
    </UiField>
  </ProductFormSection>

  <ProductFormSection>
    <template #title>界面设置</template>
    <UiFormGrid>
      <UiField variant="settings" control="checkbox">
        <UiCheckbox
          :model-value="draftSettings.ui.showAgentLogs"
          label="显示 AI 工作日志"
          @change="settingsActions.setShowAgentLogs"
        />
      </UiField>
      <UiField variant="settings" control="checkbox">
        <UiCheckbox
          :model-value="draftSettings.ui.autoImport"
          label="提取后自动导入"
          @change="settingsActions.setAutoImport"
        />
      </UiField>
    </UiFormGrid>
  </ProductFormSection>
</template>
