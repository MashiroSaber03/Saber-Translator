<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import SelectControl from './SelectControl.vue'
import StyleSettingsView from './StyleSettingsView.vue'
import TranslatorSettingsView from './TranslatorSettingsView.vue'
import ConnectionView from './ConnectionView.vue'
import type { StudioAction, StudioState } from './protocol'
import type { PluginSettingsApi } from '../../../vue-frontend/src/types/browserExtensionSettings'
const props = defineProps<{ api: PluginSettingsApi; active?: boolean; state: StudioState | null; request: (action: StudioAction) => Promise<unknown> }>()
defineEmits<{ connectionSaved: [] }>()
const section = ref('connection')
const installedModels = ref<string[]>([])
const modelSettingsActive = computed(() => (props.active ?? true) && section.value !== 'connection')
watch(modelSettingsActive, (active, _previous, cleanup) => {
  if (!active) return
  let stopped = false
  let timer: ReturnType<typeof setTimeout> | undefined
  cleanup(() => { stopped = true; clearTimeout(timer) })
  async function refreshModels() {
    try {
      const result = await props.api<{ installedModels: string[] }>('/system/models/installed')
      if (!stopped) installedModels.value = result.installedModels
    } catch { /* Keep the last inventory while the translator is disconnected. */ }
    finally { if (!stopped) timer = setTimeout(refreshModels, 5000) }
  }
  void refreshModels()
}, { immediate: true })
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
defineExpose({ save, showConnection: () => { section.value = 'connection' } })
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
    <ConnectionView :active="(active ?? true) && section === 'connection'" :state="state" :request="request" @saved="$emit('connectionSaved')" />
  </div>
  <div v-if="styleVisited" v-show="section === 'style'"><StyleSettingsView ref="styleEditor" :api="api" :installed-models="installedModels" :active="(active ?? true) && section === 'style'" /></div>
  <div v-if="sharedVisited" v-show="section !== 'style' && section !== 'connection'" class="translator-settings">
    <TranslatorSettingsView
      ref="sharedEditor"
      :api="api"
      :installed-models="installedModels"
      :section="sharedSection"
      :active="(active ?? true) && section !== 'style' && section !== 'connection'"
    />
  </div>
</template>
