<script setup lang="ts">

import UiInput from '@/components/ui/UiInput.vue'

import UiButton from '@/components/ui/UiButton.vue'
/**
 * 批量分析设置选项卡组件
 */
import { ref, computed } from 'vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { useInsightStore } from '@/stores/insightStore'
import { ARCHITECTURE_OPTIONS, ARCHITECTURE_PRESETS, type CustomLayer } from './types'

const insightStore = useInsightStore()

const pagesPerBatch = ref(insightStore.config.batch.pagesPerBatch)
const contextBatchCount = ref(insightStore.config.batch.contextBatchCount)
const architecturePreset = ref(insightStore.config.batch.architecturePreset)

const customLayers = ref<CustomLayer[]>(
  insightStore.config.batch.customLayers?.length > 0
    ? insightStore.config.batch.customLayers.map((l: any) => ({
        name: l.name,
        units: l.units_per_group ?? l.units ?? 5,
        align: l.align_to_chapter ?? l.align ?? false
      }))
    : [
        { name: "批量分析", units: 5, align: false },
        { name: "段落总结", units: 5, align: false },
        { name: "全书总结", units: 0, align: false }
      ]
)

const batchEstimate = computed(() => `每批次分析 ${pagesPerBatch.value || 5} 页`)
const showCustomLayersEditor = computed(() => architecturePreset.value === 'custom')

const architectureDescription = computed(() => {
  if (architecturePreset.value === 'custom') return '完全自定义层级架构，灵活配置分析流程'
  return ARCHITECTURE_PRESETS[architecturePreset.value]?.description || '根据漫画类型选择合适的分析架构'
})

const previewLayers = computed(() => {
  if (architecturePreset.value === 'custom') return customLayers.value
  return ARCHITECTURE_PRESETS[architecturePreset.value]?.layers || ARCHITECTURE_PRESETS['standard']!.layers
})

function addCustomLayer(): void {
  const insertIdx = customLayers.value.length - 1
  customLayers.value.splice(insertIdx, 0, { name: `汇总层${insertIdx}`, units: 5, align: false })
}

function removeCustomLayer(idx: number): void {
  if (idx > 0 && idx < customLayers.value.length - 1) customLayers.value.splice(idx, 1)
}

function updateCustomLayer(idx: number, field: keyof CustomLayer, value: string | number | boolean): void {
  if (customLayers.value[idx]) {
    (customLayers.value[idx] as any)[field] = value
    if (idx === 0 && field === 'units') pagesPerBatch.value = value as number
  }
}

function onPagesPerBatchChange(): void {
  if (customLayers.value.length > 0 && customLayers.value[0]) {
    customLayers.value[0].units = pagesPerBatch.value
  }
}

function canDeleteLayer(idx: number): boolean {
  return idx > 0 && idx < customLayers.value.length - 1 && customLayers.value.length > 2
}

function canEditLayerName(idx: number): boolean {
  return idx > 0 && idx < customLayers.value.length - 1
}

function canEditLayerUnits(idx: number): boolean {
  return idx < customLayers.value.length - 1
}

function getLayerUnitsTitle(idx: number): string {
  return idx === 0 ? '每批分析的页数' : '每组包含单元数（0=全部汇总）'
}

function getConfig() {
  return {
    pagesPerBatch: pagesPerBatch.value,
    contextBatchCount: contextBatchCount.value,
    architecturePreset: architecturePreset.value,
    // 返回前端格式（units/align），getConfigForApi 会转换为后端格式
    customLayers: customLayers.value.map(l => ({
      name: l.name,
      units: l.units,
      align: l.align
    }))
  }
}

function syncFromStore(): void {
  pagesPerBatch.value = insightStore.config.batch.pagesPerBatch
  contextBatchCount.value = insightStore.config.batch.contextBatchCount
  architecturePreset.value = insightStore.config.batch.architecturePreset
  
  // 同步 customLayers
  if (insightStore.config.batch.customLayers?.length > 0) {
    customLayers.value = insightStore.config.batch.customLayers.map((l: any) => ({
      name: l.name,
      units: l.units_per_group ?? l.units ?? 5,
      align: l.align_to_chapter ?? l.align ?? false
    }))
  }
}

defineExpose({ getConfig, syncFromStore })
</script>

<template>
  <div class="insight-settings-content">
    <p class="settings-hint">配置批量分析的参数，影响分析速度和质量。</p>
    
    <div class="insight-settings-field">
      <label>每批次分析页数</label>
      <UiInput v-model.number="pagesPerBatch" type="number" min="1" max="10" @change="onPagesPerBatchChange" />
      <p class="form-hint">每次发送给 VLM 的图片数量，建议 3-5 张。{{ batchEstimate }}</p>
    </div>
    
    <div class="insight-settings-field">
      <label>上文参考批次数</label>
      <UiInput v-model.number="contextBatchCount" type="number" min="0" max="5" />
      <p class="form-hint">每批分析时参考前几批的结果作为上下文，0 表示不参考</p>
    </div>
    
    <div class="insight-settings-field">
      <label>分析架构</label>
      <CustomSelect v-model="architecturePreset" :options="ARCHITECTURE_OPTIONS" />
      <p class="form-hint">{{ architectureDescription }}</p>
    </div>
    
    <!-- 自定义层级编辑器 -->
    <div v-if="showCustomLayersEditor" class="custom-layers-section">
      <label class="custom-layers-label">自定义层级</label>
      <div class="custom-layers-list">
        <div 
          v-for="(layer, idx) in customLayers" 
          :key="idx"
          class="custom-layer-row"
        >
          <span class="layer-index">第{{ idx + 1 }}层</span>
          <UiInput 
            type="text" 
            :value="layer.name"
            :disabled="!canEditLayerName(idx)"
            placeholder="层级名称"
            class="layer-name-input"
            @change="updateCustomLayer(idx, 'name', ($event.target as HTMLInputElement).value)"
          />
          <UiInput 
            type="number" 
            :value="layer.units"
            :disabled="!canEditLayerUnits(idx)"
            :title="getLayerUnitsTitle(idx)"
            min="0" max="20"
            class="layer-units-input"
            @change="updateCustomLayer(idx, 'units', parseInt(($event.target as HTMLInputElement).value) || 0)"
          />
          <label class="layer-align-label">
            <UiInput type="checkbox" :checked="layer.align" class="layer-align-checkbox" @change="updateCustomLayer(idx, 'align', ($event.target as HTMLInputElement).checked)" />
            <span class="layer-align-text">章节<br>对齐</span>
          </label>
          <UiButton variant="toolbar" v-if="canDeleteLayer(idx)" type="button" class="layer-delete-btn" @click="removeCustomLayer(idx)">删除</UiButton>
        </div>
      </div>
      <UiButton variant="secondary" type="button" class="layer-add-btn" @click="addCustomLayer" size="sm">+ 添加层级</UiButton>
      <p class="form-hint">第一层固定为批量分析，最后一层固定为全书总结。中间可添加任意汇总层级。</p>
    </div>
    
    <!-- 当前架构预览 -->
    <div class="batch-info-box">
      <h4>当前架构预览</h4>
      <ul class="layers-preview-list">
        <li v-for="(layer, idx) in previewLayers" :key="idx">
          <strong>第{{ idx + 1 }}层 - {{ layer.name }}</strong>
          {{ layer.units > 0 ? ` - 每${layer.units}个单元汇总` : ' - 汇总全部' }}
          <span v-if="layer.align" class="align-badge">(按章节对齐)</span>
        </li>
      </ul>
    </div>
    
    <div class="batch-estimate-box">
      <p>当前配置：每 <strong>{{ pagesPerBatch }}</strong> 页一批</p>
    </div>
  </div>
</template>

<style scoped>.insight-settings-content {
  padding: 16px 0;
  min-height: 300px;
}

.insight-settings-content .settings-hint {
  color: var(--color-text-supporting, var(--color-text-secondary));
  font-size: 13px;
  margin-bottom: 16px;
  padding: 8px 12px;
  background: var(--color-surface-muted);
  border-radius: 4px;
}

.insight-settings-content .insight-settings-field {
  margin-bottom: 16px;
}

.insight-settings-content .insight-settings-field label {
  display: block;
  margin-bottom: 6px;
  font-weight: 500;
  font-size: 14px;
  color: var(--color-text-default, var(--color-text-default));
}

.insight-settings-content .insight-settings-field input[type="text"],
.insight-settings-content .insight-settings-field input[type="password"],
.insight-settings-content .insight-settings-field input[type="number"],
.insight-settings-content .insight-settings-field select,
.insight-settings-content .insight-settings-field textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 6px;
  font-size: 14px;
  background: var(--color-surface-input, var(--color-surface-base));
  color: var(--color-text-default, var(--color-text-default));
  transition: border-color 0.2s, box-shadow 0.2s;
}

.insight-settings-content .insight-settings-field input:focus,
.insight-settings-content .insight-settings-field select:focus,
.insight-settings-content .insight-settings-field textarea:focus {
  outline: none;
  border-color: var(--color-border-brand);
  box-shadow: 0 0 0 3px var(--color-focus-brand-soft);
}

.insight-settings-content .form-hint {
  margin-top: 4px;
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.insight-settings-content .ui-checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-weight: normal;
}

.insight-settings-content .ui-checkbox-label input[type="checkbox"] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}

.insight-settings-content .ui-button {
  padding: 10px 16px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.insight-settings-content .ui-button--primary {
  background: var(--color-surface-brand);
  color: white;
}

.insight-settings-content .ui-button--primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.insight-settings-content .ui-button--primary:hover:not(:disabled) {
  background: var(--color-surface-brand-strong);
}

.insight-settings-content .ui-button--secondary {
  background: var(--color-surface-muted);
  color: var(--color-text-default, var(--color-text-default));
  border: 1px solid var(--color-border-muted, var(--color-border-default));
}

.insight-settings-content .ui-button--secondary:hover:not(:disabled) {
  background: var(--color-surface-hover);
}

.insight-settings-content .form-row {
  display: flex;
  gap: 16px;
}

.insight-settings-content .form-row .insight-settings-field {
  flex: 1;
}

.insight-settings-content .placeholder-text {
  color: var(--color-text-supporting, var(--color-text-secondary));
  text-align: center;
  padding: 40px;
}

.insight-settings-content .prompts-settings {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.insight-settings-content .prompt-editor {
  width: 100%;
  min-height: 200px;
  font-family: Consolas, Monaco, monospace;
  font-size: 13px;
  line-height: 1.5;
  padding: 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 4px;
  background: var(--color-surface-muted);
  color: var(--color-text-default, var(--color-text-default));
  resize: vertical;
}

.insight-settings-content .prompt-editor:focus {
  outline: none;
  border-color: var(--color-border-brand);
}

.insight-settings-content .prompt-actions-bar {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.insight-settings-content .ui-button--sm {
  padding: 6px 12px;
  font-size: 13px;
}

.insight-settings-content .section-divider {
  border: none;
  border-top: 1px solid var(--color-border-muted, var(--color-border-default));
  margin: 16px 0;
}

.insight-settings-content .prompts-library-section {
  margin-top: 8px;
}

.insight-settings-content .library-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.insight-settings-content .library-header h4 {
  margin: 0;
  font-size: 14px;
  font-weight: 500;
}

.insight-settings-content .library-actions {
  display: flex;
  gap: 8px;
}

.insight-settings-content .saved-prompts-list {
  max-height: 200px;
  overflow-y: auto;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 4px;
  background: var(--color-surface-muted);
}

.insight-settings-content .saved-prompt-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  cursor: pointer;
  border-bottom: 1px solid var(--color-border-muted, var(--color-border-default));
  transition: background 0.2s;
}

.insight-settings-content .saved-prompt-item:last-child {
  border-bottom: none;
}

.insight-settings-content .saved-prompt-item:hover {
  background: var(--color-surface-hover);
}

.insight-settings-content .prompt-name {
  flex: 1;
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.insight-settings-content .prompt-type-badge {
  font-size: 11px;
  padding: 2px 6px;
  background: var(--color-focus-brand-soft);
  color: var(--color-text-brand);
  border-radius: 4px;
  white-space: nowrap;
}

.insight-settings-content .button-icon-sm {
  padding: 2px 6px;
  background: none;
  border: none;
  cursor: pointer;
  opacity: 0.6;
  transition: opacity 0.2s;
}

.insight-settings-content .button-icon-sm:hover {
  opacity: 1;
}

.insight-settings-content .loading-text {
  text-align: center;
  padding: 20px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.insight-settings-content .batch-info-box {
  margin-top: 16px;
  padding: 12px;
  background: var(--color-surface-subtle);
  border-radius: 8px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
}

.insight-settings-content .batch-info-box h4 {
  margin: 0 0 8px;
  font-size: 14px;
  font-weight: 500;
  color: var(--color-text-default, var(--color-text-default));
}

.insight-settings-content .layers-preview-list {
  margin: 0;
  padding-left: 20px;
  font-size: 13px;
  line-height: 1.6;
}

.insight-settings-content .layers-preview-list li {
  margin-bottom: 4px;
}

.insight-settings-content .align-badge {
  color: var(--color-text-brand);
  font-size: 12px;
}

.insight-settings-content .batch-estimate-box {
  margin-top: 12px;
  padding: 10px 12px;
  background: linear-gradient(135deg, var(--color-focus-brand-soft), var(--batch-settings-tab-surface-base));
  border-radius: 6px;
  border: 1px solid var(--batch-settings-tab-border-default);
}

.insight-settings-content .batch-estimate-box p {
  margin: 0;
  font-size: 13px;
  color: var(--color-text-default, var(--color-text-default));
}

.insight-settings-content .batch-estimate-box strong {
  color: var(--color-text-brand);
}

.insight-settings-content .model-input-row {
  display: flex;
  gap: 8px;
  align-items: center;
}

.insight-settings-content .model-input-row input {
  flex: 1;
}

.insight-settings-content .fetch-btn {
  white-space: nowrap;
  flex-shrink: 0;
}

.insight-settings-content .model-select-container {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
  padding: 8px 12px;
  background: var(--color-surface-subtle);
  border-radius: 6px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
}

.insight-settings-content .model-select {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 4px;
  font-size: 13px;
  background: var(--color-surface-input, var(--color-surface-base));
  color: var(--color-text-default, var(--color-text-default));
  cursor: pointer;
}

.insight-settings-content .model-select:focus {
  outline: none;
  border-color: var(--color-border-brand);
}

.insight-settings-content .model-count {
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-secondary));
  white-space: nowrap;
}

.custom-layers-section {
  margin-top: 16px;
}

.insight-settings-content .custom-layers-label {
  display: block;
  margin-bottom: 8px;
  font-weight: 500;
  font-size: 14px;
}

.insight-settings-content .custom-layers-list {
  margin-bottom: 8px;
}

.insight-settings-content .custom-layer-row {
  display: flex;
  flex-direction: row;
  gap: 8px;
  align-items: center;
  margin-bottom: 8px;
  padding: 12px;
  background: var(--color-surface-subtle);
  border-radius: 8px;
  border: 1px solid var(--color-border-default);
}

.insight-settings-content .layer-index {
  min-width: 50px;
  color: var(--color-text-secondary);
  font-size: 13px;
}

.insight-settings-content .layer-name-input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid var(--color-border-default);
  border-radius: 6px;
  font-size: 14px;
}

.insight-settings-content .layer-units-input {
  width: 70px;
  padding: 8px 12px;
  border: 1px solid var(--color-border-default);
  border-radius: 6px;
  font-size: 14px;
}

.insight-settings-content .layer-align-label {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  font-size: 11px;
  cursor: pointer;
  min-width: 40px;
  text-align: center;
}

.insight-settings-content .layer-align-checkbox {
  width: 16px;
  height: 16px;
}

.insight-settings-content .layer-align-text {
  line-height: 1.2;
}

.insight-settings-content .layer-delete-btn {
  padding: 6px 12px;
  background: var(--batch-settings-tab-surface-raised);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
}

.insight-settings-content .layer-delete-btn:hover {
  background: var(--batch-settings-tab-surface-muted);
}

.insight-settings-content .layer-add-btn {
  margin-top: 4px;
  border: 1px solid var(--color-border-default);
}
</style>
