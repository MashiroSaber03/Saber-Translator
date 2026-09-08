<script setup lang="ts">
import { ref } from 'vue'
import { useBrowserExtensionSettings } from '../../../vue-frontend/src/composables/useBrowserExtensionSettings'
import SelectControl from './SelectControl.vue'
import ColorControl from './ColorControl.vue'
import type { PluginSettingsApi } from '../../../vue-frontend/src/types/browserExtensionSettings'
const props = defineProps<{ api: PluginSettingsApi }>()
const {
  style,
  fontOptions,
  form: settingsForm,
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
function bindForm(element: unknown) {
  settingsForm.value = element as HTMLFormElement | undefined
}
const reveal = ref(false)
const align = [
  { value: 'start', label: '起始' },
  { value: 'center', label: '居中' },
  { value: 'end', label: '末尾' },
]
const layouts = [
  { value: 'auto', label: '自动识别' },
  { value: 'vertical', label: '竖排' },
  { value: 'horizontal', label: '横排' },
]
const fills = [
  { value: 'solid', label: '纯色填充' },
  { value: 'lama_mpe', label: 'LAMA 修复' },
  { value: 'litelama', label: 'LiteLAMA 修复' },
]
function number(event: Event, field: 'fontSize' | 'strokeWidth' | 'lineSpacing') {
  const input = event.target as HTMLInputElement
  if (input.validity.valid && Number.isFinite(input.valueAsNumber))
    changeStyle({ [field]: input.valueAsNumber })
}
</script>
<template>
  <div class="view-heading">
    <div>
      <h2>阅读样式</h2>
      <p>仅应用于网页漫画，服务沿用翻译器配置</p>
    </div>
    <span class="save-state" :class="{ error: saveError }">{{
      saveError ? '未保存' : saving ? '保存中…' : '自动保存'
    }}</span>
  </div>
  <div
    v-if="saveError || notice"
    class="notice"
    :class="{ error: saveError || hasError }"
    role="status"
  >
    {{ saveError || notice }}
  </div>
  <div v-if="saveError" class="actions">
    <button class="button" :disabled="saving" @click="save()">重试保存</button
    ><button class="button subtle" :disabled="saving" @click="load">放弃修改并重新读取</button>
  </div>
  <form v-if="style" :ref="bindForm" @submit.prevent="save()">
    <fieldset :disabled="busy">
      <section class="setting-group">
        <h3><span>01</span>字体与排版</h3>
        <label class="field"
          >文本字体<SelectControl
            :model-value="style.fontFamily"
            :options="fontOptions"
            label="文本字体"
            searchable
            @update:model-value="updateFont"
        /></label>
        <div class="field-grid">
          <label class="field"
            >字号<input
              type="number"
              aria-label="字号"
              :value="style.fontSize"
              :disabled="style.autoFontSize"
              min="1"
              @input="number($event, 'fontSize')" /></label
          ><label class="switch-field"
            >自动计算初始字号<input
              type="checkbox"
              class="switch"
              :checked="style.autoFontSize"
              @change="
                changeStyle({
                  autoFontSize: ($event.target as HTMLInputElement).checked,
                })
              "
          /></label>
        </div>
        <label class="field"
          >排版方向<SelectControl
            :model-value="style.layoutDirection"
            :options="layouts"
            label="排版方向"
            @update:model-value="
              value =>
                changeStyle({
                  layoutDirection: value as 'auto' | 'vertical' | 'horizontal',
                })
            "
        /></label>
        <div class="field-grid">
          <label class="field"
            >行内对齐<SelectControl
              :model-value="style.inlineAlign"
              :options="align"
              label="行内对齐"
              @update:model-value="
                value =>
                  changeStyle({
                    inlineAlign: value as 'start' | 'center' | 'end',
                  })
              " /></label
          ><label class="field"
            >文本块对齐<SelectControl
              :model-value="style.blockAlign"
              :options="align"
              label="文本块对齐"
              @update:model-value="
                value =>
                  changeStyle({
                    blockAlign: value as 'start' | 'center' | 'end',
                  })
              "
          /></label>
        </div>
        <label class="field"
          >行间距<input
            type="number"
            aria-label="行间距"
            :value="style.lineSpacing"
            min="0.1"
            step="0.1"
            @input="number($event, 'lineSpacing')"
          /><small>行间距倍数</small></label
        >
      </section>
      <section class="setting-group">
        <h3><span>02</span>颜色与修复</h3>
        <label class="switch-field"
          >自动识别文字颜色<input
            type="checkbox"
            class="switch"
            :checked="style.useAutoTextColor"
            @change="
              changeStyle({
                useAutoTextColor: ($event.target as HTMLInputElement).checked,
              })
            "
        /></label>
        <div class="field-grid">
          <label class="field"
            >文字颜色<ColorControl
              :model-value="style.textColor"
              label="文字颜色"
              :disabled="style.useAutoTextColor"
              @update:model-value="textColor => changeStyle({ textColor })" /></label
          ><label class="field"
            >气泡填充方式<SelectControl
              :model-value="style.inpaintMethod"
              :options="fills"
              label="气泡填充方式"
              @update:model-value="
                value =>
                  changeStyle({
                    inpaintMethod: value as 'solid' | 'lama_mpe' | 'litelama',
                  })
              "
          /></label>
        </div>
        <label v-if="style.inpaintMethod === 'solid'" class="field"
          >填充颜色<ColorControl
            :model-value="style.fillColor"
            label="填充颜色"
            @update:model-value="fillColor => changeStyle({ fillColor })"
        /></label>
      </section>
      <section class="setting-group">
        <h3><span>03</span>文字描边</h3>
        <label class="switch-field"
          >启用描边<input
            type="checkbox"
            class="switch"
            :checked="style.strokeEnabled"
            @change="
              changeStyle({
                strokeEnabled: ($event.target as HTMLInputElement).checked,
              })
            "
        /></label>
        <div v-if="style.strokeEnabled" class="field-grid">
          <label class="field"
            >描边颜色<ColorControl
              :model-value="style.strokeColor"
              label="描边颜色"
              @update:model-value="strokeColor => changeStyle({ strokeColor })" /></label
          ><label class="field"
            >描边宽度 (px)<input
              type="number"
              aria-label="描边宽度 (px)"
              :value="style.strokeWidth"
              min="0"
              step="0.1"
              @input="number($event, 'strokeWidth')"
          /></label>
        </div>
      </section>
      <details class="disclosure">
        <summary>网页识别助手（可选）<span class="chevron" /></summary>
        <p class="muted">帮助识别网页中的漫画图片，不参与翻译。</p>
        <label class="field"
          >识别助手服务商<SelectControl
            :model-value="provider"
            :options="providerOptions"
            label="识别助手服务商"
            @update:model-value="changeProvider"
        /></label>
        <label v-if="providerMetadata?.requiresApiKey" class="field"
          >{{ credential ? 'API Key · 已配置，留空保持' : 'API Key'
          }}<span class="password-field"
            ><input
              :type="reveal ? 'text' : 'password'"
              :aria-label="credential ? 'API Key · 已配置，留空保持' : 'API Key'"
              :value="secretDrafts[provider] ?? ''"
              autocomplete="new-password"
              @input="updateKey(($event.target as HTMLInputElement).value)"
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
        <label class="field"
          >API 地址<input
            :value="draft.customBaseUrl"
            @input="updateBaseUrl(($event.target as HTMLInputElement).value)"
        /></label>
        <label class="field"
          >模型名称<input
            v-model="draft.modelName"
            list="agent-models"
            @input="markProvider" /><datalist id="agent-models">
            <option v-for="model in models" :key="model.id" :value="model.id" /></datalist
        ></label>
        <div class="actions">
          <button
            v-if="providerMetadata?.capabilities.includes('modelFetch')"
            type="button"
            class="button"
            :disabled="diagnosing"
            @click="fetchModels"
          >
            获取模型列表</button
          ><button type="button" class="button" :disabled="diagnosing" @click="testConnection">
            测试连接
          </button>
        </div>
      </details>
    </fieldset>
  </form>
  <button v-else-if="hasError" class="button" @click="load">重新读取</button>
</template>
