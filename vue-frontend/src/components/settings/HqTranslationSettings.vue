<template>
  <div class="hq-translation-settings">
    <ProductFormSection>
      <template #title>高质量翻译服务配置</template>
      <UiFormGrid>
        <AiProviderSelectField
          :model-value="hqSettings.provider"
          input-id="settingsHqTranslateProvider"
          :options="providerOptions"
          custom-profile-kind="chatVision"
          :custom-profile-api-key="hqSettings.apiKey"
          :custom-profile-base-url="hqSettings.customBaseUrl"
          :custom-profile-model="hqSettings.modelName"
          @change="handleProviderChange"
          @apply-custom-profile="applyCustomProfile"
        />
        <AiProviderCredentialFields
          :api-key="hqSettings.apiKey"
          api-key-input-id="settingsHqApiKey"
          :base-url="hqSettings.customBaseUrl"
          base-url-input-id="settingsHqCustomBaseUrl"
          :show-api-key="providerRequiresApiKey(hqSettings.provider)"
          :show-base-url="false"
          :include-base-url="false"
          api-key-placeholder="请输入API Key"
          api-key-show-label="显示高质量翻译 API Key"
          api-key-hide-label="隐藏高质量翻译 API Key"
          @update:api-key="updateHqString('apiKey', $event)"
        />
      </UiFormGrid>

      <AiProviderCredentialFields
        :api-key="hqSettings.apiKey"
        api-key-input-id="settingsHqApiKey"
        :base-url="hqSettings.customBaseUrl"
        base-url-input-id="settingsHqCustomBaseUrl"
        :show-api-key="false"
        :show-base-url="providerRequiresBaseUrl(hqSettings.provider)"
        :include-api-key="false"
        base-url-placeholder="例如: https://api.example.com/v1"
        @update:base-url="updateHqString('customBaseUrl', $event)"
      />

      <UiField variant="settings" label="模型名称" control-id="settingsHqModelName">
        <UiModelPicker
          input-id="settingsHqModelName"
          :model-value="hqSettings.modelName"
          placeholder="请输入模型名称"
          fetch-variant="primary"
          :fetching="isFetchingModels"
          :fetch-disabled="isFetchingModels"
          :options="modelListOptions"
          :model-count="modelList.length"
          @update:model-value="updateHqModel"
          @fetch="fetchModels"
        />
      </UiField>

      <UiField variant="settings">
        <UiButton variant="secondary" tone="info" @click="testConnection" :disabled="isTesting">
          <span v-if="isTesting">测试中...</span>
          <template v-else>
            <span aria-hidden="true">🔗</span>
            <span>测试连接</span>
          </template>
        </UiButton>
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>批处理设置</template>
      <UiFormGrid>
        <UiField
          variant="settings"
          label="批次大小"
          control-id="settingsHqBatchSize"
          hint="每批处理的图片数量 (推荐3-5张)"
        >
          <UiNumberField
            input-id="settingsHqBatchSize"
            :model-value="hqSettings.batchSize"
            :min="1"
            :step="1"
            @update:model-value="updateHqNumber('batchSize', $event)"
          />
        </UiField>
      </UiFormGrid>
      <UiFormGrid>
        <UiField
          variant="settings"
          label="RPM限制"
          control-id="settingsHqRpmLimit"
          hint="每分钟请求数，0表示无限制"
        >
          <UiNumberField
            input-id="settingsHqRpmLimit"
            :model-value="hqSettings.openaiOptions.execution.rpmLimit"
            :min="0"
            :max="100000"
            :step="1"
            @update:model-value="updateHqNumber('rpmLimit', $event)"
          />
        </UiField>
        <UiField
          variant="settings"
          label="重试次数"
          control-id="settingsHqMaxRetries"
          hint="业务重试：空结果/结构解析失败"
        >
          <UiNumberField
            input-id="settingsHqMaxRetries"
            :model-value="hqSettings.openaiOptions.execution.businessRetries"
            :min="0"
            :max="100"
            :step="1"
            @update:model-value="updateHqNumber('businessRetries', $event)"
          />
        </UiField>
        <UiField
          variant="settings"
          label="传输重试"
          control-id="settingsHqTransportRetries"
          hint="网络超时/429/5xx"
        >
          <UiNumberField
            input-id="settingsHqTransportRetries"
            :model-value="hqSettings.openaiOptions.execution.transportRetries"
            :min="0"
            :max="100"
            :step="1"
            @update:model-value="updateHqNumber('transportRetries', $event)"
          />
        </UiField>
      </UiFormGrid>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>高级选项</template>
      <UiFormGrid>
        <UiField variant="settings" control="checkbox" hint="使用 response_format: json_object">
          <UiCheckbox
            :model-value="hqSettings.openaiOptions.request.forceJsonOutput"
            label="强制JSON输出"
            @update:model-value="settingsStore.setHqForceJsonOutput"
          />
        </UiField>
        <UiField variant="settings" control="checkbox" hint="使用流式API调用">
          <UiCheckbox
            :model-value="hqSettings.openaiOptions.execution.useStream"
            label="流式调用"
            @update:model-value="settingsStore.setHqUseStream"
          />
        </UiField>
      </UiFormGrid>
      <UiField variant="settings">
        <OpenAIExtraBodyEditor
          :model-value="hqSettings.openaiOptions.request.extraBody"
          @update:model-value="updateHqExtraBody"
        />
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>高质量翻译偏好</template>
      <UiField
        variant="settings"
        label="翻译偏好"
        control-id="settingsHqPrompt"
        hint="只描述译文风格与质量要求，输出格式由后端管理"
      >
        <UiTextarea
          id="settingsHqPrompt"
          :model-value="hqSettings.prompt"
          variant="panel"
          rows="6"
          placeholder="请输入译文风格与质量要求"
          @update:model-value="updateHqString('prompt', $event)"
        />
        <SavedPromptsPicker prompt-type="hq_translate" @select="handleHqPromptSelect" />
        <ProductActionRow aria-label="高质量翻译提示词操作" justify="start">
          <UiButton variant="secondary" @click="resetHqPrompt" size="sm">重置为默认</UiButton>
        </ProductActionRow>
      </UiField>
    </ProductFormSection>
  </div>
</template>

<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import AiProviderCredentialFields from '@/components/settings/AiProviderCredentialFields.vue'
import AiProviderSelectField from '@/components/settings/AiProviderSelectField.vue'
import type { UiSelectValue } from '@/components/ui/selectTypes'
import type { CustomAiProfile } from '@/types/customAiProfile'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import { ref, computed } from 'vue'
import {
  getProviderDisplayName,
  getProviderOptionsForCapability,
  providerRequiresApiKey,
  providerRequiresApiKeyForBaseUrl,
  providerRequiresBaseUrl,
} from '@/config/aiProviders'
import { fetchModels as fetchV2Models, testAiTranslateConnection } from '@/api/v2/diagnostics'
import { useSettingsStore } from '@/stores/settings'
import { useToast } from '@/utils/toast'
import { DEFAULT_HQ_TRANSLATE_PROMPT } from '@/constants'
import type { HqTranslationProvider } from '@/types/settings'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'
import {
  useAiModelDiscovery,
  type AiModelDiscoveryMessageTone,
} from '@/composables/useAiModelDiscovery'

const providerOptions = getProviderOptionsForCapability('hqTranslation')

const settingsStore = useSettingsStore()
const toast = useToast()

const hqSettings = computed(() => settingsStore.settings.hqTranslation)

function notifyModelDiscovery(message: string, tone: AiModelDiscoveryMessageTone): void {
  toast[tone](message)
}

const modelDiscovery = useAiModelDiscovery({
  source: () => ({
    provider: hqSettings.value.provider,
    apiKey: hqSettings.value.apiKey,
    baseUrl: hqSettings.value.customBaseUrl,
  }),
  fetcher: (provider, apiKey, baseUrl) => fetchV2Models(provider, apiKey, baseUrl, 'hq'),
  notify: notifyModelDiscovery,
  emptyBaseUrl: '',
})
const { isFetchingModels } = modelDiscovery
const modelList = modelDiscovery.models

const isTesting = ref(false)

const modelListOptions = computed(() => {
  const options = [{ label: '-- 选择模型 --', value: '' }]
  modelDiscovery.models.value.forEach(model => options.push({ label: model.id, value: model.id }))
  return options
})

function handleProviderChange(newProvider: UiSelectValue) {
  if (typeof newProvider !== 'string') return
  if (!providerOptions.some(option => option.value === newProvider)) return
  modelDiscovery.invalidate()
  settingsStore.setHqProvider(newProvider as HqTranslationProvider)
}

function updateHqString(
  field: 'apiKey' | 'customBaseUrl' | 'prompt',
  value: string,
): void {
  settingsStore.updateHqTranslation({ [field]: value })
}

function applyCustomProfile(profile: CustomAiProfile): void {
  settingsStore.updateHqTranslation({
    apiKey: profile.apiKey,
    customBaseUrl: profile.baseUrl,
    modelName: profile.model,
  })
}

function updateHqModel(value: UiSelectValue): void {
  if (typeof value !== 'string') return
  settingsStore.updateHqTranslation({ modelName: value })
}

function updateHqNumber(
  field: 'batchSize' | 'rpmLimit' | 'businessRetries' | 'transportRetries',
  value: number | null,
): void {
  if (value === null) return
  settingsStore.updateHqTranslation({ [field]: value })
}

function updateHqExtraBody(value: Record<string, unknown> | undefined): void {
  settingsStore.updateHqTranslation({ extraBody: value })
}

const fetchModels = modelDiscovery.fetchModels

async function testConnection() {
  const provider = hqSettings.value.provider
  const apiKey = hqSettings.value.apiKey?.trim()
  const modelName = hqSettings.value.modelName?.trim()
  const baseUrl = hqSettings.value.customBaseUrl?.trim()

  if (
    providerRequiresApiKeyForBaseUrl(provider, baseUrl)
    && !apiKey
  ) {
    toast.warning('请先填写 API Key')
    return
  }

  if (!modelName) {
    toast.warning('请填写模型名称')
    return
  }

  if (providerRequiresBaseUrl(provider) && !baseUrl) {
    toast.warning('自定义服务需要填写 Base URL')
    return
  }

  isTesting.value = true
  toast.info('正在测试连接...')

  try {
    const result = await testAiTranslateConnection({
      provider,
      apiKey,
      modelName,
      baseUrl,
      domain: 'hq',
    })

    if (result.success) {
      toast.success(result.message || `${getProviderDisplayName(provider)} 连接成功!`)
    } else {
      toast.error(result.message || '连接失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '连接测试失败'
    toast.error(errorMessage)
  } finally {
    isTesting.value = false
  }
}

function resetHqPrompt() {
  settingsStore.updateHqTranslation({ prompt: DEFAULT_HQ_TRANSLATE_PROMPT })
  toast.success('已重置为默认提示词')
}

function handleHqPromptSelect(content: string, name: string) {
  settingsStore.updateHqTranslation({ prompt: content })
  toast.success(`已应用提示词: ${name}`)
}
</script>
