<script setup lang="ts">
import { ref, watch } from 'vue'
import SelectControl from './SelectControl.vue'
import StyleSettingsView from './StyleSettingsView.vue'
import TranslatorSettingsView from './TranslatorSettingsView.vue'
import ConnectionView from './ConnectionView.vue'
import type { StudioAction, StudioState } from './protocol'
import type { PluginSettingsApi } from '../../../vue-frontend/src/types/browserExtensionSettings'
defineProps<{ api: PluginSettingsApi; active?: boolean; state: StudioState | null; request: (action: StudioAction) => Promise<unknown> }>()
const section = ref('connection')
const styleVisited = ref(false)
const sharedVisited = ref(false)
const sharedSection = ref('ocr')
const styleEditor = ref<InstanceType<typeof StyleSettingsView>>()
const sharedEditor = ref<InstanceType<typeof TranslatorSettingsView>>()
async function save(): Promise<boolean> {
  if (styleEditor.value && !(await styleEditor.value.save(false))) {
    section.value = 'style'
    return false
  }
  if (sharedEditor.value && !(await sharedEditor.value.save())) {
    section.value = sharedSection.value
    return false
  }
  return true
}
defineExpose({ save })
watch(section, value => {
  if (value === 'style') styleVisited.value = true
  else if (value !== 'connection') {
    sharedVisited.value = true
    sharedSection.value = value
  }
})
const options = [
  { value: 'connection', label: '连接与站点' },
  { value: 'style', label: '插件文本样式' },
  { value: 'ocr', label: 'OCR 识别' },
  { value: 'translation', label: '翻译服务' },
  { value: 'detection', label: '检测设置' },
  { value: 'hq', label: '高质量翻译' },
]
</script>
<template>
  <label class="field"
    >配置分类<SelectControl v-model="section" :options="options" label="配置分类"
  /></label>
  <div v-show="section === 'connection'">
    <ConnectionView :active="(active ?? true) && section === 'connection'" :state="state" :request="request" />
  </div>
  <div v-if="styleVisited" v-show="section === 'style'"><StyleSettingsView ref="styleEditor" :api="api" :active="(active ?? true) && section === 'style'" /></div>
  <div v-if="sharedVisited" v-show="section !== 'style' && section !== 'connection'" class="translator-settings">
    <TranslatorSettingsView
      ref="sharedEditor"
      :api="api"
      :section="sharedSection"
      :active="(active ?? true) && section !== 'style' && section !== 'connection'"
    />
  </div>
</template>
