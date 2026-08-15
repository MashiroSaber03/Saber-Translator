<template>
  <div class="detection-settings">
    <ProductFormSection>
      <template #title>文字检测器</template>
      <UiField variant="settings" label="检测器类型" control-id="settingsTextDetector">
        <UiSelect
          id="settingsTextDetector"
          :model-value="settings.textDetector"
          :options="detectorOptions"
          @change="updateTextDetector"
        />
      </UiField>
      <UiField
        variant="settings"
        label="最小文本框面积占比 (%)"
        control-id="settingsMinTextBlockAreaPercent"
        hint="检测完成后自动删除面积低于原图该百分比的极小文本框，0 表示不过滤"
      >
        <UiNumberField
          input-id="settingsMinTextBlockAreaPercent"
          :model-value="settings.minTextBlockAreaPercent"
          :min="0"
          :max="100"
          :step="0.01"
          @change="updateNumber(settingsStore.setMinTextBlockAreaPercent, $event)"
        />
      </UiField>
      <UiField
        variant="settings"
        control="checkbox"
        hint="使用 YSGYolo 在一阶段检测后补框/替框，提升主检测器结果质量"
      >
        <UiCheckbox
          :model-value="settings.enableAuxYoloDetection"
          label="启用辅助 YSGYolo 检测"
          @change="settingsStore.setEnableAuxYoloDetection"
        />
      </UiField>
      <UiFormGrid>
        <UiField variant="settings" label="辅助 YSGYolo 置信度" control-id="settingsAuxYoloConfThreshold">
          <UiNumberField
            input-id="settingsAuxYoloConfThreshold"
            :model-value="settings.auxYoloConfThreshold"
            :min="0"
            :max="1"
            :step="0.05"
            @change="updateNumber(settingsStore.setAuxYoloConfThreshold, $event)"
          />
        </UiField>
        <UiField variant="settings" label="辅助 YSGYolo 重叠阈值" control-id="settingsAuxYoloOverlapThreshold">
          <UiNumberField
            input-id="settingsAuxYoloOverlapThreshold"
            :model-value="settings.auxYoloOverlapThreshold"
            :min="0"
            :max="1"
            :step="0.05"
            @change="updateNumber(settingsStore.setAuxYoloOverlapThreshold, $event)"
          />
        </UiField>
      </UiFormGrid>
      <UiField
        variant="settings"
        control="checkbox"
        hint="使用 SaberYOLO 对误合并的大文本块进行二次拆分修正"
      >
        <UiCheckbox
          :model-value="settings.enableSaberYoloRefine"
          label="启用 SaberYOLO 二阶段纠错"
          @change="settingsStore.setEnableSaberYoloRefine"
        />
      </UiField>
      <UiField
        variant="settings"
        label="SaberYOLO 拆分阈值 (%)"
        control-id="settingsSaberYoloRefineOverlapThreshold"
        hint="参考块与当前 block 的重叠面积占参考块面积的最小百分比，默认 50%"
      >
        <UiNumberField
          input-id="settingsSaberYoloRefineOverlapThreshold"
          :model-value="settings.saberYoloRefineOverlapThreshold"
          :min="0"
          :max="100"
          :step="1"
          @change="updateNumber(settingsStore.setSaberYoloRefineOverlapThreshold, $event)"
        />
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>文本框扩展参数</template>
      <UiField
        variant="settings"
        label="整体扩展 (%)"
        control-id="settingsBoxExpandRatio"
        hint="向四周均匀扩展的百分比 (0-50%)"
      >
        <UiNumberField
          input-id="settingsBoxExpandRatio"
          :model-value="settings.boxExpand.ratio"
          :min="0"
          :max="50"
          :step="1"
          @change="updateBoxExpand('ratio', $event)"
        />
      </UiField>
      <UiFormGrid>
        <UiField variant="settings" label="上方扩展 (%)" control-id="settingsBoxExpandTop">
          <UiNumberField input-id="settingsBoxExpandTop" :model-value="settings.boxExpand.top" :min="0" :max="50" :step="1" @change="updateBoxExpand('top', $event)" />
        </UiField>
        <UiField variant="settings" label="下方扩展 (%)" control-id="settingsBoxExpandBottom">
          <UiNumberField input-id="settingsBoxExpandBottom" :model-value="settings.boxExpand.bottom" :min="0" :max="50" :step="1" @change="updateBoxExpand('bottom', $event)" />
        </UiField>
      </UiFormGrid>
      <UiFormGrid>
        <UiField variant="settings" label="左侧扩展 (%)" control-id="settingsBoxExpandLeft">
          <UiNumberField input-id="settingsBoxExpandLeft" :model-value="settings.boxExpand.left" :min="0" :max="50" :step="1" @change="updateBoxExpand('left', $event)" />
        </UiField>
        <UiField variant="settings" label="右侧扩展 (%)" control-id="settingsBoxExpandRight">
          <UiNumberField input-id="settingsBoxExpandRight" :model-value="settings.boxExpand.right" :min="0" :max="50" :step="1" @change="updateBoxExpand('right', $event)" />
        </UiField>
      </UiFormGrid>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>精确文字掩膜</template>
      <UiFormGrid>
        <UiField
          variant="settings"
          label="掩膜膨胀大小"
          control-id="settingsMaskDilateSize"
          hint="掩膜膨胀像素数"
        >
          <UiNumberField input-id="settingsMaskDilateSize" :model-value="settings.preciseMask.dilateSize" :min="0" :step="1" @change="updatePreciseMask('dilateSize', $event)" />
        </UiField>
        <UiField
          variant="settings"
          label="标注框扩大比例 (%)"
          control-id="settingsMaskBoxExpandRatio"
          hint="标注框区域扩大百分比"
        >
          <UiNumberField
            input-id="settingsMaskBoxExpandRatio"
            :model-value="settings.preciseMask.boxExpandRatio"
            :min="0"
            :max="100"
            :step="1"
            @change="updatePreciseMask('boxExpandRatio', $event)"
          />
        </UiField>
      </UiFormGrid>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>调试选项</template>
      <UiField
        variant="settings"
        control="checkbox"
        hint="在翻译结果中显示气泡检测框，用于调试"
      >
        <UiCheckbox
          :model-value="settings.showDetectionDebug"
          label="显示检测框调试信息"
          @change="settingsStore.setShowDetectionDebug"
        />
      </UiField>
    </ProductFormSection>
  </div>
</template>

<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { computed } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import type { BoxExpandSettings, PreciseMaskSettings } from '@/types/settings'
import type { UiSelectValue } from '@/components/ui/selectTypes'

const detectorOptions = [
  { label: 'CTD (Comic Text Detector)', value: 'ctd' },
  { label: 'YOLO', value: 'yolo' },
  { label: 'Default (DBNet)', value: 'default' }
]

const settingsStore = useSettingsStore()

const settings = computed(() => settingsStore.settings)

function updateTextDetector(value: UiSelectValue): void {
  if (value === 'ctd' || value === 'yolo' || value === 'default') {
    settingsStore.setTextDetector(value)
  }
}

function updateNumber(commit: (value: number) => void, value: number | null): void {
  if (value !== null) commit(value)
}

function updateBoxExpand(key: keyof BoxExpandSettings, value: number | null): void {
  if (value !== null) settingsStore.updateBoxExpand({ [key]: value })
}

function updatePreciseMask(key: keyof PreciseMaskSettings, value: number | null): void {
  if (value !== null) settingsStore.updatePreciseMask({ [key]: value })
}
</script>
