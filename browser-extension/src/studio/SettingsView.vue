<script setup lang="ts">
import { ref, watch } from 'vue'
import SelectControl from './SelectControl.vue'
import StyleSettingsView from './StyleSettingsView.vue'
import TranslatorSettingsView from './TranslatorSettingsView.vue'
import type { PluginSettingsApi } from '../../../vue-frontend/src/types/browserExtensionSettings'
defineProps<{ api: PluginSettingsApi; active?: boolean }>()
const section = ref('style')
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
  if (value !== 'style') {
    sharedVisited.value = true
    sharedSection.value = value
  }
})
const options = [
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
  <div v-show="section === 'style'"><StyleSettingsView ref="styleEditor" :api="api" :active="(active ?? true) && section === 'style'" /></div>
  <div v-if="sharedVisited" v-show="section !== 'style'" class="translator-settings">
    <TranslatorSettingsView
      ref="sharedEditor"
      :api="api"
      :section="sharedSection"
      :active="(active ?? true) && section !== 'style'"
    />
  </div>
</template>
