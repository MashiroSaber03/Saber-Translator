<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import {
  getTranslationApiKeyLabel,
  getTranslationModelNameLabel,
} from '../../../vue-frontend/src/components/settings/translationSettingsLabels'
import SelectControl from './SelectControl.vue'
import NumberControl from './NumberControl.vue'
import providers from '../../../src/shared/ai_provider_manifest.json'
import type { OpenAICompatibleOptions } from '../../../vue-frontend/src/types/openaiSettings'
import type { PluginSettingsApi } from '../../../vue-frontend/src/types/browserExtensionSettings'
import type { components } from '../../../vue-frontend/src/api/generated/v2'
export interface ServiceConfig {
  provider: string
  modelName: string
  customBaseUrl: string
  openaiOptions: OpenAICompatibleOptions
  prompt?: string
}
const props = defineProps<{
  config: ServiceConfig
  domain: string
  capability: string
  api: PluginSettingsApi
  secret: string
}>()
const emit = defineEmits<{
  change: [patch: Partial<ServiceConfig>]
  provider: [value: string]
  secret: [value: string]
}>()
const reveal = ref(false),
  diagnosing = ref(false),
  message = ref(''),
  failed = ref(false)
const models = ref<{ value: string; label: string }[]>([])
const options = computed(() =>
  providers
    .filter(p => p.capabilities.includes(props.capability))
    .map(p => ({ value: p.id, label: p.label }))
)
const metadata = computed(() => providers.find(p => p.id === props.config.provider))
const keyLabel = computed(() =>
  props.domain === 'translation' ? getTranslationApiKeyLabel(props.config.provider) : 'API Key'
)
const modelLabel = computed(() =>
  props.domain === 'translation' ? getTranslationModelNameLabel(props.config.provider) : '模型名称'
)
const extra = ref('')
watch(
  () => props.config.openaiOptions.request.extraBody,
  value => {
    extra.value = value ? JSON.stringify(value, null, 2) : ''
  },
  { immediate: true }
)
watch(
  () => props.config.provider,
  () => {
    models.value = []
    message.value = ''
    reveal.value = false
  }
)
function execution(patch: Partial<OpenAICompatibleOptions['execution']>) {
  emit('change', {
    openaiOptions: {
      ...props.config.openaiOptions,
      execution: { ...props.config.openaiOptions.execution, ...patch },
    },
  })
}
function extraBody(event: Event) {
  const node = event.target as HTMLTextAreaElement
  try {
    const body = node.value.trim() ? JSON.parse(node.value) : undefined
    if (node.value.trim() && (!body || typeof body !== 'object' || Array.isArray(body)))
      throw new Error('请输入 JSON 对象')
    if (
      body &&
      Object.keys(body).some(key =>
        ['model', 'messages', 'temperature', 'response_format', 'stream'].includes(key)
      )
    )
      throw new Error('不能覆盖 model、messages、temperature、response_format 或 stream 字段')
    node.setCustomValidity('')
    emit('change', {
      openaiOptions: {
        ...props.config.openaiOptions,
        request: { ...props.config.openaiOptions.request, extraBody: body },
      },
    })
  } catch (e) {
    node.setCustomValidity(
      e instanceof SyntaxError ? '请输入有效的 JSON 对象' : (e as Error).message
    )
    node.reportValidity()
  }
}
async function diagnose(kind: 'models' | 'connection') {
  diagnosing.value = true
  failed.value = false
  message.value = ''
  const provider = props.config.provider
  const body = {
    provider,
    domain: props.domain,
    baseUrl: props.config.customBaseUrl,
    ...(metadata.value?.requiresApiKey || props.secret.trim()
      ? {
          secret: {
            [props.domain === 'ai_vision_ocr' ? 'ai_vision_api_key' : 'api_key']:
              props.secret.trim(),
          },
        }
      : {}),
  }
  try {
    if (kind === 'models') {
      const result = await props.api<components['schemas']['ModelCatalogResponse']>(
        '/model-catalog',
        'POST',
        body
      )
      if (props.config.provider !== provider) return
      models.value = result.models.map(m => ({ value: m.id, label: m.name || m.id }))
      message.value = `已获取 ${models.value.length} 个模型`
    } else {
      const traditional = ['baidu_translate', 'youdao_translate'].includes(provider)
      const kind = traditional
        ? provider
        : props.domain === 'ai_vision_ocr'
          ? 'ai_vision_ocr'
          : 'ai_translate'
      const key = props.secret.trim()
      const request = traditional
        ? {
            domain: props.domain,
            secret:
              provider === 'baidu_translate'
                ? { app_id: key, app_key: props.config.modelName }
                : { app_key: key, app_secret: props.config.modelName },
          }
        : {
            ...body,
            model: props.config.modelName,
            ...(kind === 'ai_vision_ocr' ? { prompt: props.config.prompt } : {}),
          }
      const result = await props.api<components['schemas']['ConnectionTestResponse']>(
        `/connection-tests/${kind}`,
        'POST',
        request
      )
      if (props.config.provider !== provider) return
      message.value = result.message ?? (result.success ? '连接成功' : '连接失败')
      failed.value = !result.success
    }
  } catch (e) {
    if (props.config.provider !== provider) return
    message.value = (e as Error).message
    failed.value = true
  } finally {
    diagnosing.value = false
  }
}
</script>
<template>
  <label class="field"
    >服务商<SelectControl
      :model-value="config.provider"
      :options="options"
      label="服务商"
      @update:model-value="value => emit('provider', String(value))"
  /></label>
  <label v-if="metadata?.requiresApiKey" class="field"
    >{{ keyLabel }}<span class="password-field"
      ><input
        :type="reveal ? 'text' : 'password'"
        :aria-label="keyLabel"
        :value="secret"
        autocomplete="new-password"
        @input="emit('secret', ($event.target as HTMLInputElement).value)"
      /><button
        type="button"
        class="text-button"
        :aria-label="reveal ? '隐藏 API Key' : '显示 API Key'"
        @click="reveal = !reveal"
      >
        {{ reveal ? '隐藏' : '显示' }}
      </button></span
    ></label
  >
  <label v-if="metadata?.requiresBaseUrl" class="field"
    >API 地址<input
      :value="config.customBaseUrl"
      :placeholder="metadata?.defaultBaseUrl"
      @input="emit('change', { customBaseUrl: ($event.target as HTMLInputElement).value })"
  /></label>
  <label v-if="metadata?.requiresModel" class="field"
    >{{ modelLabel
    }}<input
      :value="config.modelName"
      @input="emit('change', { modelName: ($event.target as HTMLInputElement).value })"
  /></label>
  <SelectControl
    v-if="models.length"
    :model-value="config.modelName"
    :options="models"
    label="选择已获取的模型"
    searchable
    @update:model-value="value => emit('change', { modelName: String(value) })"
  />
  <div class="actions">
    <button
      v-if="metadata?.capabilities.includes('modelFetch')"
      type="button"
      class="button"
      :disabled="diagnosing"
      @click="diagnose('models')"
    >
      获取模型列表</button
    ><button type="button" class="button" :disabled="diagnosing" @click="diagnose('connection')">
      测试连接
    </button>
  </div>
  <div v-if="message" class="notice" :class="{ error: failed }" role="status">{{ message }}</div>
  <details class="disclosure">
    <summary>请求参数<span class="chevron" /></summary>
    <NumberControl
      v-if="metadata?.kind === 'openai_compatible'"
      :model-value="config.openaiOptions.execution.rpmLimit"
      label="RPM 限制"
      :min="0"
      :max="100000"
      @update:model-value="rpmLimit => execution({ rpmLimit })"
    />
    <div class="field-grid">
      <NumberControl
        :model-value="config.openaiOptions.execution.businessRetries"
        label="业务重试次数"
        :min="0"
        :max="100"
        @update:model-value="businessRetries => execution({ businessRetries })"
      /><NumberControl
        :model-value="config.openaiOptions.execution.transportRetries"
        label="传输重试次数"
        :min="0"
        :max="100"
        @update:model-value="transportRetries => execution({ transportRetries })"
      />
    </div>
    <label v-if="metadata?.supportsStream" class="switch-field"
      >流式调用<input
        class="switch"
        type="checkbox"
        :checked="config.openaiOptions.execution.useStream"
        @change="execution({ useStream: ($event.target as HTMLInputElement).checked })"
    /></label>
    <label v-if="metadata?.kind === 'openai_compatible'" class="field"
      >附加请求参数（JSON）<textarea
        v-model="extra"
        rows="4"
        placeholder='例如：{"thinking":{"type":"disabled"}}'
        @change="extraBody"
      />
    </label>
  </details>
</template>
