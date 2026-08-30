<template>
  <div class="browser-dom-agent-settings">
    <ProductStatusBanner
      tone="info"
      title="仅在你主动选择 DOM Agent 识别时调用"
    >
      Saber 只发送裁剪后的图片节点结构、尺寸和必要属性，不发送图片内容、正文、Cookie 或配对令牌。
    </ProductStatusBanner>

    <ProductFormSection>
      <template #title>Browser DOM Agent</template>
      <UiFormGrid>
        <AiProviderSelectField
          :model-value="agent.provider"
          input-id="settingsBrowserDomAgentProvider"
          :options="providerOptions"
          custom-profile-kind="chatVision"
          :custom-profile-api-key="agent.apiKey"
          :custom-profile-base-url="agent.customBaseUrl"
          :custom-profile-model="agent.modelName"
          @change="setProvider"
          @apply-custom-profile="applyCustomProfile"
        />
        <AiProviderCredentialFields
          :api-key="agent.apiKey"
          api-key-input-id="settingsBrowserDomAgentApiKey"
          :base-url="agent.customBaseUrl"
          base-url-input-id="settingsBrowserDomAgentBaseUrl"
          :show-api-key="providerRequiresApiKey(agent.provider)"
          :show-base-url="false"
          :include-base-url="false"
          api-key-placeholder="请输入 API Key"
          @update:api-key="updateString('apiKey', $event)"
        />
      </UiFormGrid>

      <AiProviderCredentialFields
        :api-key="agent.apiKey"
        api-key-input-id="settingsBrowserDomAgentApiKeyHidden"
        :base-url="agent.customBaseUrl"
        base-url-input-id="settingsBrowserDomAgentBaseUrl"
        :show-api-key="false"
        :show-base-url="providerRequiresBaseUrl(agent.provider)"
        :include-api-key="false"
        base-url-placeholder="例如：https://api.example.com/v1"
        @update:base-url="updateString('customBaseUrl', $event)"
      />

      <UiField
        variant="settings"
        label="模型名称"
        control-id="settingsBrowserDomAgentModel"
        hint="选择能够稳定输出 JSON 的轻量聊天模型即可"
      >
        <UiInput
          id="settingsBrowserDomAgentModel"
          :model-value="agent.modelName"
          placeholder="请输入模型名称"
          @update:model-value="updateModel"
        />
      </UiField>
    </ProductFormSection>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import AiProviderCredentialFields from '@/components/settings/AiProviderCredentialFields.vue'
import AiProviderSelectField from '@/components/settings/AiProviderSelectField.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import UiInput from '@/components/ui/UiInput.vue'
import type { UiSelectValue } from '@/components/ui/selectTypes'
import type { CustomAiProfile } from '@/types/customAiProfile'
import type { PluginAgentProvider } from '@/types/settings'
import {
  getProviderOptionsForCapability,
  providerRequiresApiKey,
  providerRequiresBaseUrl,
} from '@/config/aiProviders'
import { useSettingsStore } from '@/stores/settings'

const settingsStore = useSettingsStore()
const agent = computed(() => settingsStore.settings.browserDomAgent)
const providerOptions = getProviderOptionsForCapability('pluginAgent')

function setProvider(value: UiSelectValue): void {
  if (typeof value !== 'string') return
  if (!providerOptions.some(option => option.value === value)) return
  settingsStore.setBrowserDomAgentProvider(value as PluginAgentProvider)
}

function updateString(
  field: 'apiKey' | 'customBaseUrl',
  value: string,
): void {
  settingsStore.updateBrowserDomAgent({ [field]: value })
}

function updateModel(value: string | number | boolean): void {
  if (typeof value === 'string') {
    settingsStore.updateBrowserDomAgent({ modelName: value })
  }
}

function applyCustomProfile(profile: CustomAiProfile): void {
  settingsStore.updateBrowserDomAgent({
    apiKey: profile.apiKey,
    customBaseUrl: profile.baseUrl,
    modelName: profile.model,
  })
}
</script>
