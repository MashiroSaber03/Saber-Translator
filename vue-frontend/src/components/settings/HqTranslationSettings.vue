<template>
  <div class="hq-translation-settings">
    <ProductFormSection>
      <template #title>高质量翻译服务配置</template>
      <UiFormGrid>
        <AiProviderSelectField
          :model-value="hqSettings.provider"
          input-id="settingsHqTranslateProvider"
          :options="providerOptions"
          @change="handleProviderChange"
        />
        <AiProviderCredentialFields
          :api-key="localHqSettings.apiKey"
          api-key-input-id="settingsHqApiKey"
          :base-url="localHqSettings.customBaseUrl"
          base-url-input-id="settingsHqCustomBaseUrl"
          :show-api-key="providerRequiresApiKey(hqSettings.provider)"
          :show-base-url="false"
          :include-base-url="false"
          api-key-placeholder="请输入API Key"
          :has-stored-credential="settingsStore.hasCredential('hq', hqSettings.provider)"
          api-key-show-label="显示高质量翻译 API Key"
          api-key-hide-label="隐藏高质量翻译 API Key"
          @update:api-key="localHqSettings.apiKey = $event"
        />
      </UiFormGrid>

      <AiProviderCredentialFields
        :api-key="localHqSettings.apiKey"
        api-key-input-id="settingsHqApiKey"
        :base-url="localHqSettings.customBaseUrl"
        base-url-input-id="settingsHqCustomBaseUrl"
        :show-api-key="false"
        :show-base-url="providerRequiresBaseUrl(hqSettings.provider)"
        :include-api-key="false"
        base-url-placeholder="例如: https://api.example.com/v1"
        @update:base-url="localHqSettings.customBaseUrl = $event"
      />

      <UiField variant="settings" label="模型名称" control-id="settingsHqModelName">
        <UiModelPicker
          input-id="settingsHqModelName"
          v-model="localHqSettings.modelName"
          placeholder="请输入模型名称"
          fetch-variant="primary"
          :fetching="isFetchingModels"
          :fetch-disabled="isFetchingModels"
          :options="modelListOptions"
          :model-count="modelList.length"
          @fetch="fetchModels"
        />
      </UiField>

      <UiField variant="settings">
        <UiButton variant="secondary" block @click="testConnection" :disabled="isTesting">
          <span v-if="isTesting">测试中...</span>
          <template v-else>
            <UiIcon name="link" />
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
            v-model="localHqSettings.batchSize"
            :min="1"
            :max="10"
            :step="1"
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
            v-model="localHqSettings.rpmLimit"
            :min="0"
            :step="1"
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
            v-model="localHqSettings.businessRetries"
            :min="0"
            :max="10"
            :step="1"
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
            v-model="localHqSettings.transportRetries"
            :min="0"
            :max="10"
            :step="1"
          />
        </UiField>
      </UiFormGrid>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>高级选项</template>
      <UiFormGrid>
        <UiField variant="settings" control="checkbox" hint="使用 response_format: json_object">
          <UiCheckbox v-model="localHqSettings.forceJsonOutput" label="强制JSON输出" />
        </UiField>
        <UiField variant="settings" control="checkbox" hint="使用流式API调用">
          <UiCheckbox v-model="localHqSettings.useStream" label="流式调用" />
        </UiField>
      </UiFormGrid>
      <UiField variant="settings">
        <OpenAIExtraBodyEditor v-model="localHqSettings.extraBody" />
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>高质量翻译提示词</template>
      <UiField variant="settings" label="高质量翻译提示词" control-id="settingsHqPrompt">
        <UiTextarea
          id="settingsHqPrompt"
          v-model="localHqSettings.prompt"
          variant="panel"
          rows="6"
          placeholder="高质量翻译提示词"
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
import UiIcon from '@/components/ui/UiIcon.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import AiProviderCredentialFields from '@/components/settings/AiProviderCredentialFields.vue'
import AiProviderSelectField from '@/components/settings/AiProviderSelectField.vue'
import type { UiSelectValue } from '@/components/ui/selectTypes'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import { ref, computed, watch } from 'vue'
import {
  getProviderDisplayName,
  getProviderOptionsForCapability,
  providerRequiresApiKey,
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

const localHqSettings = ref({
  apiKey: settingsStore.settings.hqTranslation.apiKey,
  modelName: settingsStore.settings.hqTranslation.modelName,
  customBaseUrl: settingsStore.settings.hqTranslation.customBaseUrl,
  batchSize: settingsStore.settings.hqTranslation.batchSize,
  rpmLimit: settingsStore.settings.hqTranslation.openaiOptions.execution.rpmLimit,
  transportRetries: settingsStore.settings.hqTranslation.openaiOptions.execution.transportRetries,
  businessRetries: settingsStore.settings.hqTranslation.openaiOptions.execution.businessRetries,
  forceJsonOutput: settingsStore.settings.hqTranslation.openaiOptions.request.forceJsonOutput,
  extraBody: settingsStore.settings.hqTranslation.openaiOptions.request.extraBody,
  useStream: settingsStore.settings.hqTranslation.openaiOptions.execution.useStream,
  prompt: settingsStore.settings.hqTranslation.prompt,
})

watch(
  () => localHqSettings.value.apiKey,
  val => {
    settingsStore.updateHqTranslation({ apiKey: val })
  }
)
watch(
  () => localHqSettings.value.modelName,
  val => {
    settingsStore.updateHqTranslation({ modelName: val })
  }
)
watch(
  () => localHqSettings.value.customBaseUrl,
  val => {
    settingsStore.updateHqTranslation({ customBaseUrl: val })
  }
)
watch(
  () => localHqSettings.value.batchSize,
  val => {
    settingsStore.updateHqTranslation({ batchSize: val })
  }
)
watch(
  () => localHqSettings.value.rpmLimit,
  val => {
    settingsStore.updateHqTranslation({ rpmLimit: val })
  }
)
watch(
  () => localHqSettings.value.transportRetries,
  val => {
    settingsStore.updateHqTranslation({ transportRetries: val })
  }
)
watch(
  () => localHqSettings.value.businessRetries,
  val => {
    settingsStore.updateHqTranslation({ businessRetries: val })
  }
)
watch(
  () => localHqSettings.value.forceJsonOutput,
  val => {
    settingsStore.updateHqTranslation({ forceJsonOutput: val })
  }
)
watch(
  () => localHqSettings.value.extraBody,
  val => {
    settingsStore.updateHqTranslation({ extraBody: val })
  }
)
watch(
  () => localHqSettings.value.useStream,
  val => {
    settingsStore.updateHqTranslation({ useStream: val })
  }
)
watch(
  () => localHqSettings.value.prompt,
  val => {
    settingsStore.updateHqTranslation({ prompt: val })
  }
)

function notifyModelDiscovery(message: string, tone: AiModelDiscoveryMessageTone): void {
  toast[tone](message)
}

const modelDiscovery = useAiModelDiscovery({
  source: () => ({
    provider: hqSettings.value.provider,
    apiKey: localHqSettings.value.apiKey,
    baseUrl: localHqSettings.value.customBaseUrl,
    hasStoredCredential: settingsStore.hasCredential('hq', hqSettings.value.provider),
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
  modelDiscovery.invalidate()
  settingsStore.setHqProvider(String(newProvider) as HqTranslationProvider)
  syncLocalHqSettings()
}

function syncLocalHqSettings() {
  const hq = settingsStore.settings.hqTranslation
  localHqSettings.value.apiKey = hq.apiKey
  localHqSettings.value.modelName = hq.modelName
  localHqSettings.value.customBaseUrl = hq.customBaseUrl
  localHqSettings.value.batchSize = hq.batchSize
  localHqSettings.value.rpmLimit = hq.openaiOptions.execution.rpmLimit
  localHqSettings.value.transportRetries = hq.openaiOptions.execution.transportRetries
  localHqSettings.value.businessRetries = hq.openaiOptions.execution.businessRetries
  localHqSettings.value.forceJsonOutput = hq.openaiOptions.request.forceJsonOutput
  localHqSettings.value.extraBody = hq.openaiOptions.request.extraBody
  localHqSettings.value.useStream = hq.openaiOptions.execution.useStream
  localHqSettings.value.prompt = hq.prompt
}

const fetchModels = modelDiscovery.fetchModels

async function testConnection() {
  const provider = hqSettings.value.provider
  const apiKey = localHqSettings.value.apiKey?.trim()
  const modelName = localHqSettings.value.modelName?.trim()
  const baseUrl = localHqSettings.value.customBaseUrl?.trim()

  if (providerRequiresApiKey(provider) && !apiKey && !settingsStore.hasCredential('hq', provider)) {
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
  localHqSettings.value.prompt = DEFAULT_HQ_TRANSLATE_PROMPT
  toast.success('已重置为默认提示词')
}

function handleHqPromptSelect(content: string, name: string) {
  settingsStore.updateHqTranslation({ prompt: content })
  localHqSettings.value.prompt = content
  toast.success(`已应用提示词: ${name}`)
}
</script>
