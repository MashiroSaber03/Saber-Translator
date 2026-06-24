<template>
  <div class="detection-settings">
    <UiPanel variant="settings">
      <template #title>文字检测器</template>
      <UiField class="ui-settings-field">
        <label for="settingsTextDetector">检测器类型:</label>
        <CustomSelect
          v-model="settings.textDetector"
          :options="detectorOptions"
        />
      </UiField>
      <UiField class="ui-settings-field">
        <label for="settingsMinTextBlockAreaPercent">最小文本框面积占比 (%):</label>
        <UiInput
          type="number"
          id="settingsMinTextBlockAreaPercent"
          v-model.number="settings.minTextBlockAreaPercent"
          min="0"
          max="100"
          step="0.01"
        />
        <div class="ui-form-hint">检测完成后自动删除面积低于原图该百分比的极小文本框，0 表示不过滤</div>
      </UiField>
      <UiField class="ui-settings-field">
        <label class="ui-checkbox-label">
          <UiInput type="checkbox" class="detection-settings__checkbox-input" v-model="settings.enableAuxYoloDetection" />
          启用辅助 YSGYolo 检测
        </label>
        <div class="ui-form-hint">使用 YSGYolo 在一阶段检测后补框/替框，提升主检测器结果质量</div>
      </UiField>
      <UiFormGrid>
        <UiField class="ui-settings-field">
          <label for="settingsAuxYoloConfThreshold">辅助 YSGYolo 置信度:</label>
          <UiInput
            type="number"
            id="settingsAuxYoloConfThreshold"
            v-model.number="settings.auxYoloConfThreshold"
            min="0"
            max="1"
            step="0.05"
          />
        </UiField>
        <UiField class="ui-settings-field">
          <label for="settingsAuxYoloOverlapThreshold">辅助 YSGYolo 重叠阈值:</label>
          <UiInput
            type="number"
            id="settingsAuxYoloOverlapThreshold"
            v-model.number="settings.auxYoloOverlapThreshold"
            min="0"
            max="1"
            step="0.05"
          />
        </UiField>
      </UiFormGrid>
      <UiField class="ui-settings-field">
        <label class="ui-checkbox-label">
          <UiInput type="checkbox" class="detection-settings__checkbox-input" v-model="settings.enableSaberYoloRefine" />
          启用 SaberYOLO 二阶段纠错
        </label>
        <div class="ui-form-hint">使用 SaberYOLO 对误合并的大文本块进行二次拆分修正</div>
      </UiField>
      <UiField class="ui-settings-field">
        <label for="settingsSaberYoloRefineOverlapThreshold">SaberYOLO 拆分阈值 (%):</label>
        <UiInput
          type="number"
          id="settingsSaberYoloRefineOverlapThreshold"
          v-model.number="settings.saberYoloRefineOverlapThreshold"
          min="0"
          max="100"
          step="1"
        />
        <div class="ui-form-hint">参考块与当前 block 的重叠面积占参考块面积的最小百分比，默认 50%</div>
      </UiField>
    </UiPanel>

    <UiPanel variant="settings">
      <template #title>文本框扩展参数</template>
      <UiField class="ui-settings-field">
        <label for="settingsBoxExpandRatio">整体扩展 (%):</label>
        <UiInput type="number" id="settingsBoxExpandRatio" v-model.number="settings.boxExpandRatio" min="0" max="50" step="1" />
        <div class="ui-form-hint">向四周均匀扩展的百分比 (0-50%)</div>
      </UiField>
      <UiFormGrid>
        <UiField class="ui-settings-field">
          <label for="settingsBoxExpandTop">上方扩展 (%):</label>
          <UiInput type="number" id="settingsBoxExpandTop" v-model.number="settings.boxExpandTop" min="0" max="50" step="1" />
        </UiField>
        <UiField class="ui-settings-field">
          <label for="settingsBoxExpandBottom">下方扩展 (%):</label>
          <UiInput type="number" id="settingsBoxExpandBottom" v-model.number="settings.boxExpandBottom" min="0" max="50" step="1" />
        </UiField>
      </UiFormGrid>
      <UiFormGrid>
        <UiField class="ui-settings-field">
          <label for="settingsBoxExpandLeft">左侧扩展 (%):</label>
          <UiInput type="number" id="settingsBoxExpandLeft" v-model.number="settings.boxExpandLeft" min="0" max="50" step="1" />
        </UiField>
        <UiField class="ui-settings-field">
          <label for="settingsBoxExpandRight">右侧扩展 (%):</label>
          <UiInput type="number" id="settingsBoxExpandRight" v-model.number="settings.boxExpandRight" min="0" max="50" step="1" />
        </UiField>
      </UiFormGrid>
    </UiPanel>


    <UiPanel variant="settings">
      <template #title>精确文字掩膜</template>
      <UiFormGrid>
        <UiField class="ui-settings-field">
          <label for="settingsMaskDilateSize">膨胀大小:</label>
          <UiInput type="number" id="settingsMaskDilateSize" v-model.number="settings.maskDilateSize" min="0" step="1" />
          <div class="ui-form-hint">掩膜膨胀像素数</div>
        </UiField>
        <UiField class="ui-settings-field">
          <label for="settingsMaskBoxExpandRatio">标注框扩大比例 (%):</label>
          <UiInput
            type="number"
            id="settingsMaskBoxExpandRatio"
            v-model.number="settings.maskBoxExpandRatio"
            min="0"
            max="100"
            step="1"
          />
          <div class="ui-form-hint">标注框区域扩大百分比</div>
        </UiField>
      </UiFormGrid>
    </UiPanel>

    <UiPanel variant="settings">
      <template #title>调试选项</template>
      <UiField class="ui-settings-field">
        <label class="ui-checkbox-label">
          <UiInput type="checkbox" class="detection-settings__checkbox-input" v-model="settings.showDetectionDebug" />
          显示检测框调试信息
        </label>
        <div class="ui-form-hint">在翻译结果中显示气泡检测框，用于调试</div>
      </UiField>
    </UiPanel>
  </div>
</template>

<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import UiPanel from '@/components/ui/UiPanel.vue'
import UiInput from '@/components/ui/UiInput.vue'
import { reactive, watch } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import CustomSelect from '@/components/common/CustomSelect.vue'

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

<style scoped>
.ui-checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.detection-settings__checkbox-input {
  width: auto;
}
</style>
