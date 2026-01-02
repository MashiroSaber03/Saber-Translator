<script setup lang="ts">
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
    architecturePreset: architecturePreset.value
  }
}

function syncFromStore(): void {
  pagesPerBatch.value = insightStore.config.batch.pagesPerBatch
  contextBatchCount.value = insightStore.config.batch.contextBatchCount
  architecturePreset.value = insightStore.config.batch.architecturePreset
}

defineExpose({ getConfig, syncFromStore })
</script>

<template>
  <div class="insight-settings-content">
    <p class="settings-hint">配置批量分析的参数，影响分析速度和质量。</p>
    
    <div class="form-group">
      <label>每批次分析页数</label>
      <input v-model.number="pagesPerBatch" type="number" min="1" max="10" @change="onPagesPerBatchChange">
      <p class="form-hint">每次发送给 VLM 的图片数量，建议 3-5 张。{{ batchEstimate }}</p>
    </div>
    
    <div class="form-group">
      <label>上文参考批次数</label>
      <input v-model.number="contextBatchCount" type="number" min="0" max="5">
      <p class="form-hint">每批分析时参考前几批的结果作为上下文，0 表示不参考</p>
    </div>
    
    <div class="form-group">
      <label>分析架构</label>
      <CustomSelect v-model="architecturePreset" :options="ARCHITECTURE_OPTIONS" />
      <p class="form-hint">{{ architectureDescription }}</p>
    </div>
    
    <!-- 自定义层级编辑器 -->
    <div v-if="showCustomLayersEditor" style="margin-top: 16px;">
      <label style="display: block; margin-bottom: 8px; font-weight: 500; font-size: 14px;">自定义层级</label>
      <div style="margin-bottom: 8px;">
        <div 
          v-for="(layer, idx) in customLayers" 
          :key="idx"
          style="display: flex; flex-direction: row; gap: 8px; align-items: center; margin-bottom: 8px; padding: 12px; background: #f5f5f5; border-radius: 8px; border: 1px solid #e0e0e0;"
        >
          <span style="min-width: 50px; color: #666; font-size: 13px;">第{{ idx + 1 }}层</span>
          <input 
            type="text" 
            :value="layer.name"
            :disabled="!canEditLayerName(idx)"
            placeholder="层级名称"
            style="flex: 1; padding: 8px 12px; border: 1px solid #e0e0e0; border-radius: 6px; font-size: 14px;"
            @change="updateCustomLayer(idx, 'name', ($event.target as HTMLInputElement).value)"
          >
          <input 
            type="number" 
            :value="layer.units"
            :disabled="!canEditLayerUnits(idx)"
            :title="getLayerUnitsTitle(idx)"
            min="0" max="20"
            style="width: 70px; padding: 8px 12px; border: 1px solid #e0e0e0; border-radius: 6px; font-size: 14px;"
            @change="updateCustomLayer(idx, 'units', parseInt(($event.target as HTMLInputElement).value) || 0)"
          >
          <label style="display: flex; flex-direction: column; align-items: center; gap: 2px; font-size: 11px; cursor: pointer; min-width: 40px; text-align: center;">
            <input type="checkbox" :checked="layer.align" style="width: 16px; height: 16px;" @change="updateCustomLayer(idx, 'align', ($event.target as HTMLInputElement).checked)">
            <span style="line-height: 1.2;">章节<br>对齐</span>
          </label>
          <button v-if="canDeleteLayer(idx)" type="button" style="padding: 6px 12px; background: #ef4444; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500;" @click="removeCustomLayer(idx)">删除</button>
        </div>
      </div>
      <button type="button" class="btn btn-sm" style="margin-top: 4px; border: 1px solid #e0e0e0;" @click="addCustomLayer">+ 添加层级</button>
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
