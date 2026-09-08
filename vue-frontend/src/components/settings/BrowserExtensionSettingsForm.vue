<template>
  <div class="browser-extension-settings">
    <ProductStatusBanner
      v-if="saveError || notice"
      :tone="saveError || hasError ? 'danger' : 'info'"
      role="status"
    >
      {{ saveError || notice }}
    </ProductStatusBanner>
    <ProductActionRow v-if="saveError" justify="start">
      <UiButton variant="secondary" :disabled="saving" @click="save()">重试保存</UiButton>
      <UiButton variant="secondary" :disabled="saving" @click="load">放弃修改并重新读取</UiButton>
    </ProductActionRow>
    <form v-if="document && style" ref="form" @submit.prevent="save()">
      <fieldset :disabled="busy">
        <p class="browser-extension-settings__intro">
          插件文本样式独立保存，翻译服务沿用翻译器配置。
        </p>
        <TextStyleForm
          :model-value="style"
          :id-prefix="idPrefix"
          :font-select-options="fontOptions"
          @change="changeStyle"
          @font-change="updateFont"
        />
        <details class="browser-extension-settings__agent">
          <summary>网页识别助手（可选）</summary>
          <ProductFormSection>
            <p class="browser-extension-settings__intro">用于识别网页中的漫画图片，不参与翻译。</p>
            <UiField variant="settings" label="识别助手服务商" :control-id="idPrefix + 'Provider'">
              <UiSelect
                :id="idPrefix + 'Provider'"
                :model-value="provider"
                :options="providerOptions"
                @change="changeProvider"
              />
            </UiField>
            <AiProviderCredentialFields
              :api-key="secretDrafts[provider] ?? String(credential?.secret?.api_key ?? '')"
              :api-key-input-id="idPrefix + 'Key'"
              api-key-label="API Key"
              :base-url="draft.customBaseUrl"
              :base-url-input-id="idPrefix + 'BaseUrl'"
              :include-api-key="Boolean(providerMetadata?.requiresApiKey)"
              :show-base-url="Boolean(providerMetadata?.requiresBaseUrl)"
              @update:api-key="updateKey"
              @update:base-url="updateBaseUrl"
            />
            <UiField variant="settings" label="模型名称" :control-id="idPrefix + 'Model'">
              <UiInput
                :id="idPrefix + 'Model'"
                v-model="draft.modelName"
                :list="idPrefix + 'Models'"
                @update:model-value="markProvider"
              />
              <datalist :id="idPrefix + 'Models'">
                <option v-for="model in models" :key="model.id" :value="model.id">
                  {{ model.name }}
                </option>
              </datalist>
            </UiField>
            <ProductActionRow justify="start">
              <UiButton
                v-if="providerMetadata?.capabilities.includes('modelFetch')"
                type="button"
                variant="secondary"
                :disabled="diagnosing"
                @click="fetchModels"
              >
                获取模型列表
              </UiButton>
              <UiButton
                type="button"
                variant="secondary"
                :disabled="diagnosing"
                @click="testConnection"
              >
                测试连接
              </UiButton>
            </ProductActionRow>
          </ProductFormSection>
        </details>
      </fieldset>
    </form>
    <UiButton v-else-if="hasError" variant="secondary" @click="load">重新读取</UiButton>
  </div>
</template>

<script setup lang="ts">
import { useId, watch } from 'vue'
import type { PluginSettingsApi } from '@/types/browserExtensionSettings'
import { useBrowserExtensionSettings } from '@/composables/useBrowserExtensionSettings'
import TextStyleForm from './TextStyleForm.vue'
import AiProviderCredentialFields from './AiProviderCredentialFields.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiField from '@/components/ui/UiField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'

const props = defineProps<{ api: PluginSettingsApi }>()
const emit = defineEmits<{ saving: [value: boolean] }>()
const idPrefix = `browserSettings-${useId()}-`
const {
  document,
  style,
  fontOptions,
  form,
  notice,
  hasError,
  busy,
  saving,
  saveError,
  diagnosing,
  draft,
  provider,
  providerOptions,
  providerMetadata,
  credential,
  secretDrafts,
  models,
  changeStyle,
  updateFont,
  changeProvider,
  markProvider,
  updateBaseUrl,
  updateKey,
  load,
  save,
  fetchModels,
  testConnection,
} = useBrowserExtensionSettings(props.api)
watch([busy, saving], ([loading, submitting]) => emit('saving', loading || submitting), {
  flush: 'sync',
})
defineExpose({ save })
</script>

<style scoped>
.browser-extension-settings fieldset {
  border: 0;
  padding: 0;
  margin: 0;
  min-width: 0;
}
.browser-extension-settings__intro {
  color: var(--color-text-supporting);
  font-size: 13px;
  line-height: 1.6;
  margin: 0 0 14px;
}
.browser-extension-settings__agent {
  margin-top: 25px;
}
.browser-extension-settings__agent > summary {
  cursor: pointer;
  color: var(--color-action-primary);
  font-weight: 600;
  margin-bottom: 15px;
}
</style>
