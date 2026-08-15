<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import { ref, computed } from 'vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { useInsightStore } from '@/stores/insightStore'
import type { BatchConfig } from '@/types/insight'
import InsightSettingsPanel from './InsightSettingsPanel.vue'
import { useInsightSettingsDraft } from './useInsightSettingsDraft'
import { ARCHITECTURE_OPTIONS, ARCHITECTURE_PRESETS, type CustomLayer } from './types'

const emit = defineEmits<{
  (e: 'update:config', config: BatchConfig): void
}>()

const props = defineProps<{
  syncRequestId?: number
}>()

const insightStore = useInsightStore()

const pagesPerBatch = ref(insightStore.config.batch.pagesPerBatch)
const contextBatchCount = ref(insightStore.config.batch.contextBatchCount)
const architecturePreset = ref(insightStore.config.batch.architecturePreset)

function createDefaultCustomLayers(): CustomLayer[] {
  return [
    { name: '批量分析', units: 5, align: false },
    { name: '段落总结', units: 5, align: false },
    { name: '全书总结', units: 0, align: false },
  ]
}

function cloneCustomLayers(layers: CustomLayer[]): CustomLayer[] {
  return layers.map(layer => ({
    name: layer.name,
    units: layer.units,
    align: layer.align,
  }))
}

const customLayers = ref<CustomLayer[]>(
  insightStore.config.batch.customLayers.length > 0
    ? cloneCustomLayers(insightStore.config.batch.customLayers)
    : createDefaultCustomLayers()
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

function updateCustomLayer(idx: number, field: keyof CustomLayer, value: string | number | boolean | null): void {
  if (value === null) return
  const layer = customLayers.value[idx]
  if (!layer) return

  if (field === 'name' && typeof value === 'string') {
    layer.name = value
  } else if (field === 'units' && typeof value === 'number') {
    layer.units = value
    if (idx === 0) pagesPerBatch.value = value
  } else if (field === 'align' && typeof value === 'boolean') {
    layer.align = value
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

function buildDraftConfig(): BatchConfig {
  return {
    pagesPerBatch: pagesPerBatch.value,
    contextBatchCount: contextBatchCount.value,
    architecturePreset: architecturePreset.value,
    customLayers: customLayers.value.map(l => ({
      name: l.name,
      units: l.units,
      align: l.align,
    })),
  }
}

function applyDraftConfig(config: BatchConfig): void {
  pagesPerBatch.value = config.pagesPerBatch
  contextBatchCount.value = config.contextBatchCount
  architecturePreset.value = config.architecturePreset

  if (config.customLayers.length > 0) {
    customLayers.value = cloneCustomLayers(config.customLayers)
  } else {
    customLayers.value = createDefaultCustomLayers()
  }
}

useInsightSettingsDraft<BatchConfig>({
  sources: [pagesPerBatch, contextBatchCount, architecturePreset, customLayers],
  buildDraft: buildDraftConfig,
  applyDraft: applyDraftConfig,
  loadDraft: () => insightStore.config.batch,
  emitDraft: config => emit('update:config', config),
  syncRequestId: () => props.syncRequestId,
  deep: true,
})
</script>

<template>
  <InsightSettingsPanel class="batch-settings-tab" description="配置批量分析的参数，影响分析速度和质量。">
    <UiFormGrid class="batch-settings-tab__primary-fields">
      <UiField
        variant="settings"
        label="每批次分析页数"
        :hint="`每次发送给 VLM 的图片数量，建议 3-5 张。${batchEstimate}`"
        control-id="insight-batch-pages-per-batch"
      >
        <UiNumberField
          v-model="pagesPerBatch"
          input-id="insight-batch-pages-per-batch"
          :min="1"
          @change="onPagesPerBatchChange"
        />
      </UiField>

      <UiField
        variant="settings"
        label="上文参考批次数"
        hint="每批分析时参考前几批的结果作为上下文，0 表示不参考"
        control-id="insight-batch-context-batch-count"
      >
        <UiNumberField
          v-model="contextBatchCount"
          input-id="insight-batch-context-batch-count"
          :min="0"
        />
      </UiField>
    </UiFormGrid>

    <UiField variant="settings" label="分析架构" :hint="architectureDescription">
      <UiSelect v-model="architecturePreset" :options="ARCHITECTURE_OPTIONS" />
    </UiField>

    <div v-if="showCustomLayersEditor" class="batch-settings-tab__custom-layers">
      <UiField variant="settings" label="自定义层级" hint="第一层固定为批量分析，最后一层固定为全书总结。中间可添加任意汇总层级。">
        <div class="batch-settings-tab__custom-layer-list">
          <div
            v-for="(layer, idx) in customLayers"
            :key="idx"
            class="batch-settings-tab__layer-row"
          >
            <span class="batch-settings-tab__layer-index">第{{ idx + 1 }}层</span>
            <UiInput
              type="text"
              :model-value="layer.name"
              :disabled="!canEditLayerName(idx)"
              placeholder="层级名称"
              class="batch-settings-tab__layer-name-input"
              @update:model-value="updateCustomLayer(idx, 'name', $event)"
            />
            <UiNumberField
              :model-value="layer.units"
              :disabled="!canEditLayerUnits(idx)"
              :input-id="`insight-batch-layer-units-${idx}`"
              :title="getLayerUnitsTitle(idx)"
              :min="0"
              size="xs"
              @change="updateCustomLayer(idx, 'units', $event)"
            />
            <div class="batch-settings-tab__layer-align-label">
              <UiCheckbox
                :model-value="layer.align"
                class="batch-settings-tab__layer-align-checkbox"
                label="章节对齐"
                @update:model-value="updateCustomLayer(idx, 'align', $event)"
              />
            </div>
            <UiButton v-if="canDeleteLayer(idx)" variant="danger" type="button" size="sm" @click="removeCustomLayer(idx)">
              <UiIcon name="trash" />
              <span>删除</span>
            </UiButton>
          </div>
        </div>
        <ProductActionRow aria-label="自定义层级操作" justify="start">
          <UiButton variant="secondary" type="button" @click="addCustomLayer" size="sm">
            <UiIcon name="plus" />
            <span>添加层级</span>
          </UiButton>
        </ProductActionRow>
      </UiField>
    </div>

    <ProductStatusBanner
      class="batch-settings-tab__summary-banner"
      title="当前架构预览"
      tone="neutral"
      icon-name="list"
      role="note"
    >
      <ol class="batch-settings-tab__layers-preview">
        <li
          v-for="(layer, idx) in previewLayers"
          :key="idx"
          class="batch-settings-tab__preview-layer"
        >
          <strong class="batch-settings-tab__preview-layer-title">第{{ idx + 1 }}层 - {{ layer.name }}</strong>
          {{ layer.units > 0 ? ` - 每${layer.units}个单元汇总` : ' - 汇总全部' }}
          <span v-if="layer.align" class="batch-settings-tab__align-badge">(按章节对齐)</span>
        </li>
      </ol>
    </ProductStatusBanner>

    <ProductStatusBanner
      class="batch-settings-tab__config-banner"
      title="当前配置"
      tone="info"
      icon-name="settings"
      role="note"
    >
      ：每 <strong class="batch-settings-tab__config-value">{{ pagesPerBatch }}</strong> 页一批
    </ProductStatusBanner>
  </InsightSettingsPanel>
</template>

<style scoped>
.batch-settings-tab {
  --ui-number-field-width: 100%;
  --ui-number-field-input-width: 100%;
  --ui-number-field-text-align: left;
}

.batch-settings-tab__primary-fields {
  grid-template-columns: 1fr;
  gap: 20px;
}

.batch-settings-tab__layers-preview {
  margin: 0;
  padding-left: 20px;
  font-size: 13px;
  line-height: 1.6;
}

.batch-settings-tab__preview-layer {
  margin-bottom: 4px;
}

.batch-settings-tab__align-badge {
  color: var(--color-text-brand);
  font-size: 12px;
}

.batch-settings-tab__config-value {
  color: var(--color-text-brand);
}

.batch-settings-tab__summary-banner {
  margin-top: 16px;

  --product-status-banner-icon-display: none;
  --product-status-banner-border: 0;
  --product-status-banner-background: var(--color-surface-muted);
  --product-status-banner-padding: 14px;
}

.batch-settings-tab__config-banner {
  margin-top: 12px;

  --product-status-banner-icon-display: none;
  --product-status-banner-content-display: flex;
  --product-status-banner-content-align-items: center;
  --product-status-banner-content-gap: 0;
  --product-status-banner-title-display: inline;
  --product-status-banner-title-margin-bottom: 0;
  --product-status-banner-body-display: inline;
  --product-status-banner-border: 1px solid color-mix(in srgb, var(--color-border-brand) 18%, transparent);
  --product-status-banner-radius: 4px;
  --product-status-banner-background: color-mix(in srgb, var(--color-surface-brand) 7%, var(--color-surface-card));
  --product-status-banner-padding: 10px 12px;
}

.batch-settings-tab__custom-layers {
  margin-top: 16px;
}

.batch-settings-tab__custom-layer-list {
  margin-bottom: 8px;
}

.batch-settings-tab__layer-row {
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

.batch-settings-tab__layer-index {
  min-width: 50px;
  color: var(--color-text-secondary);
  font-size: 13px;
}

.batch-settings-tab__layer-name-input {
  flex: 1;
}

.batch-settings-tab__layer-align-label {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  font-size: 11px;
  cursor: pointer;
  min-width: 40px;
  text-align: center;
}

.batch-settings-tab__layer-align-checkbox {
  align-items: center;
}
</style>
