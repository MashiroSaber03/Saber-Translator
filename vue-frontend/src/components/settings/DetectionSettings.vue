<template>
  <div class="detection-settings">
    <ProductFormSection>
      <template #title>文字检测器</template>
      <UiField variant="settings" label="检测器类型" control-id="settingsTextDetector">
        <UiSelect
          id="settingsTextDetector"
          v-model="settings.textDetector"
          :options="detectorOptions"
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
          v-model="settings.minTextBlockAreaPercent"
          :min="0"
          :max="100"
          :step="0.01"
        />
      </UiField>
      <UiField
        variant="settings"
        control="checkbox"
        hint="使用 YSGYolo 在一阶段检测后补框/替框，提升主检测器结果质量"
      >
        <UiCheckbox v-model="settings.enableAuxYoloDetection" label="启用辅助 YSGYolo 检测" />
      </UiField>
      <UiFormGrid>
        <UiField variant="settings" label="辅助 YSGYolo 置信度" control-id="settingsAuxYoloConfThreshold">
          <UiNumberField
            input-id="settingsAuxYoloConfThreshold"
            v-model="settings.auxYoloConfThreshold"
            :min="0"
            :max="1"
            :step="0.05"
          />
        </UiField>
        <UiField variant="settings" label="辅助 YSGYolo 重叠阈值" control-id="settingsAuxYoloOverlapThreshold">
          <UiNumberField
            input-id="settingsAuxYoloOverlapThreshold"
            v-model="settings.auxYoloOverlapThreshold"
            :min="0"
            :max="1"
            :step="0.05"
          />
        </UiField>
      </UiFormGrid>
      <UiField
        variant="settings"
        control="checkbox"
        hint="使用 SaberYOLO 对误合并的大文本块进行二次拆分修正"
      >
        <UiCheckbox v-model="settings.enableSaberYoloRefine" label="启用 SaberYOLO 二阶段纠错" />
      </UiField>
      <UiField
        variant="settings"
        label="SaberYOLO 拆分阈值 (%)"
        control-id="settingsSaberYoloRefineOverlapThreshold"
        hint="参考块与当前 block 的重叠面积占参考块面积的最小百分比，默认 50%"
      >
        <UiNumberField
          input-id="settingsSaberYoloRefineOverlapThreshold"
          v-model="settings.saberYoloRefineOverlapThreshold"
          :min="0"
          :max="100"
          :step="1"
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
        <UiNumberField input-id="settingsBoxExpandRatio" v-model="settings.boxExpandRatio" :min="0" :max="50" :step="1" />
      </UiField>
      <UiFormGrid>
        <UiField variant="settings" label="上方扩展 (%)" control-id="settingsBoxExpandTop">
          <UiNumberField input-id="settingsBoxExpandTop" v-model="settings.boxExpandTop" :min="0" :max="50" :step="1" />
        </UiField>
        <UiField variant="settings" label="下方扩展 (%)" control-id="settingsBoxExpandBottom">
          <UiNumberField input-id="settingsBoxExpandBottom" v-model="settings.boxExpandBottom" :min="0" :max="50" :step="1" />
        </UiField>
      </UiFormGrid>
      <UiFormGrid>
        <UiField variant="settings" label="左侧扩展 (%)" control-id="settingsBoxExpandLeft">
          <UiNumberField input-id="settingsBoxExpandLeft" v-model="settings.boxExpandLeft" :min="0" :max="50" :step="1" />
        </UiField>
        <UiField variant="settings" label="右侧扩展 (%)" control-id="settingsBoxExpandRight">
          <UiNumberField input-id="settingsBoxExpandRight" v-model="settings.boxExpandRight" :min="0" :max="50" :step="1" />
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
          <UiNumberField input-id="settingsMaskDilateSize" v-model="settings.maskDilateSize" :min="0" :step="1" />
        </UiField>
        <UiField
          variant="settings"
          label="标注框扩大比例 (%)"
          control-id="settingsMaskBoxExpandRatio"
          hint="标注框区域扩大百分比"
        >
          <UiNumberField
            input-id="settingsMaskBoxExpandRatio"
            v-model="settings.maskBoxExpandRatio"
            :min="0"
            :max="100"
            :step="1"
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
        <UiCheckbox v-model="settings.showDetectionDebug" label="显示检测框调试信息" />
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
import { reactive, watch } from 'vue'
import { useSettingsStore } from '@/stores/settings'

const detectorOptions = [
  { label: 'CTD (Comic Text Detector)', value: 'ctd' },
  { label: 'YOLO', value: 'yolo' },
  { label: 'Default (DBNet)', value: 'default' }
]

const settingsStore = useSettingsStore()

const settings = reactive({
  textDetector: settingsStore.settings.textDetector,
  minTextBlockAreaPercent: settingsStore.settings.minTextBlockAreaPercent,
  enableAuxYoloDetection: settingsStore.settings.enableAuxYoloDetection,
  auxYoloConfThreshold: settingsStore.settings.auxYoloConfThreshold,
  auxYoloOverlapThreshold: settingsStore.settings.auxYoloOverlapThreshold,
  enableSaberYoloRefine: settingsStore.settings.enableSaberYoloRefine,
  saberYoloRefineOverlapThreshold: settingsStore.settings.saberYoloRefineOverlapThreshold,
  boxExpandRatio: settingsStore.settings.boxExpand.ratio,
  boxExpandTop: settingsStore.settings.boxExpand.top,
  boxExpandBottom: settingsStore.settings.boxExpand.bottom,
  boxExpandLeft: settingsStore.settings.boxExpand.left,
  boxExpandRight: settingsStore.settings.boxExpand.right,
  maskDilateSize: settingsStore.settings.preciseMask.dilateSize,
  maskBoxExpandRatio: settingsStore.settings.preciseMask.boxExpandRatio,
  showDetectionDebug: settingsStore.settings.showDetectionDebug
})

watch(() => settings.textDetector, (value) => {
  settingsStore.setTextDetector(value as 'ctd' | 'yolo' | 'default')
})

watch(() => settings.minTextBlockAreaPercent, (value) => {
  settingsStore.setMinTextBlockAreaPercent(value)
})

watch(() => settings.enableAuxYoloDetection, (value) => {
  settingsStore.setEnableAuxYoloDetection(value)
})

watch(() => settings.auxYoloConfThreshold, (value) => {
  settingsStore.setAuxYoloConfThreshold(value)
})

watch(() => settings.auxYoloOverlapThreshold, (value) => {
  settingsStore.setAuxYoloOverlapThreshold(value)
})

watch(() => settings.enableSaberYoloRefine, (value) => {
  settingsStore.setEnableSaberYoloRefine(value)
})

watch(() => settings.saberYoloRefineOverlapThreshold, (value) => {
  settingsStore.setSaberYoloRefineOverlapThreshold(value)
})

watch(() => settings.boxExpandRatio, (value) => {
  settingsStore.updateBoxExpand({ ratio: value })
})

watch(() => settings.boxExpandTop, (value) => {
  settingsStore.updateBoxExpand({ top: value })
})

watch(() => settings.boxExpandBottom, (value) => {
  settingsStore.updateBoxExpand({ bottom: value })
})

watch(() => settings.boxExpandLeft, (value) => {
  settingsStore.updateBoxExpand({ left: value })
})

watch(() => settings.boxExpandRight, (value) => {
  settingsStore.updateBoxExpand({ right: value })
})

watch(() => settings.maskDilateSize, (value) => {
  settingsStore.updatePreciseMask({ dilateSize: value })
})

watch(() => settings.maskBoxExpandRatio, (value) => {
  settingsStore.updatePreciseMask({ boxExpandRatio: value })
})

watch(() => settings.showDetectionDebug, (value) => {
  settingsStore.setShowDetectionDebug(value)
})
</script>
