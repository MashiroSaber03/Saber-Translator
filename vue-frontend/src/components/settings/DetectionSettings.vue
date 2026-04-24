<template>
  <div class="detection-settings">
    <!-- 文字检测器设置 -->
    <div class="settings-group">
      <div class="settings-group-title">文字检测器</div>
      <div class="settings-item">
        <label for="settingsTextDetector">检测器类型:</label>
        <CustomSelect
          v-model="settings.textDetector"
          :options="detectorOptions"
        />
      </div>
      <div class="settings-item">
        <label class="checkbox-label">
          <input type="checkbox" v-model="settings.enableAuxYoloDetection" />
          启用辅助 YSGYolo 检测
        </label>
        <div class="input-hint">使用 YSGYolo 在一阶段检测后补框/替框，提升主检测器结果质量</div>
      </div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="settingsAuxYoloConfThreshold">辅助 YSGYolo 置信度:</label>
          <input
            type="number"
            id="settingsAuxYoloConfThreshold"
            v-model.number="settings.auxYoloConfThreshold"
            min="0"
            max="1"
            step="0.05"
          />
        </div>
        <div class="settings-item">
          <label for="settingsAuxYoloOverlapThreshold">辅助 YSGYolo 重叠阈值:</label>
          <input
            type="number"
            id="settingsAuxYoloOverlapThreshold"
            v-model.number="settings.auxYoloOverlapThreshold"
            min="0"
            max="1"
            step="0.05"
          />
        </div>
      </div>
      <div class="settings-item">
        <label class="checkbox-label">
          <input type="checkbox" v-model="settings.enableSaberYoloRefine" />
          启用 SaberYOLO 二阶段纠错
        </label>
        <div class="input-hint">使用 SaberYOLO 对误合并的大文本块进行二次拆分修正</div>
      </div>
      <div class="settings-item">
        <label for="settingsSaberYoloRefineOverlapThreshold">SaberYOLO 拆分阈值 (%):</label>
        <input
          type="number"
          id="settingsSaberYoloRefineOverlapThreshold"
          v-model.number="settings.saberYoloRefineOverlapThreshold"
          min="0"
          max="100"
          step="1"
        />
        <div class="input-hint">参考块与当前 block 的重叠面积占参考块面积的最小百分比，默认 50%</div>
      </div>
    </div>

    <!-- 文本框扩展参数 -->
    <div class="settings-group">
      <div class="settings-group-title">文本框扩展参数</div>
      <div class="settings-item">
        <label for="settingsBoxExpandRatio">整体扩展 (%):</label>
        <input type="number" id="settingsBoxExpandRatio" v-model.number="settings.boxExpandRatio" min="0" max="50" step="1" />
        <div class="input-hint">向四周均匀扩展的百分比 (0-50%)</div>
      </div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="settingsBoxExpandTop">上方扩展 (%):</label>
          <input type="number" id="settingsBoxExpandTop" v-model.number="settings.boxExpandTop" min="0" max="50" step="1" />
        </div>
        <div class="settings-item">
          <label for="settingsBoxExpandBottom">下方扩展 (%):</label>
          <input type="number" id="settingsBoxExpandBottom" v-model.number="settings.boxExpandBottom" min="0" max="50" step="1" />
        </div>
      </div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="settingsBoxExpandLeft">左侧扩展 (%):</label>
          <input type="number" id="settingsBoxExpandLeft" v-model.number="settings.boxExpandLeft" min="0" max="50" step="1" />
        </div>
        <div class="settings-item">
          <label for="settingsBoxExpandRight">右侧扩展 (%):</label>
          <input type="number" id="settingsBoxExpandRight" v-model.number="settings.boxExpandRight" min="0" max="50" step="1" />
        </div>
      </div>
    </div>


    <!-- 精确文字掩膜设置 (常驻功能) -->
    <div class="settings-group">
      <div class="settings-group-title">精确文字掩膜</div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="settingsMaskDilateSize">膨胀大小:</label>
          <input type="number" id="settingsMaskDilateSize" v-model.number="settings.maskDilateSize" min="0" step="1" />
          <div class="input-hint">掩膜膨胀像素数</div>
        </div>
        <div class="settings-item">
          <label for="settingsMaskBoxExpandRatio">标注框扩大比例 (%):</label>
          <input
            type="number"
            id="settingsMaskBoxExpandRatio"
            v-model.number="settings.maskBoxExpandRatio"
            min="0"
            max="100"
            step="1"
          />
          <div class="input-hint">标注框区域扩大百分比</div>
        </div>
      </div>
    </div>

    <!-- 调试选项 -->
    <div class="settings-group">
      <div class="settings-group-title">调试选项</div>
      <div class="settings-item">
        <label class="checkbox-label">
          <input type="checkbox" v-model="settings.showDetectionDebug" />
          显示检测框调试信息
        </label>
        <div class="input-hint">在翻译结果中显示气泡检测框，用于调试</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 检测设置组件
 * 管理文字检测器和相关参数配置
 */
import { reactive, watch } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'
import CustomSelect from '@/components/common/CustomSelect.vue'

/** 检测器类型选项 */
const detectorOptions = [
  { label: 'CTD (Comic Text Detector)', value: 'ctd' },
  { label: 'YOLO', value: 'yolo' },
  { label: 'Default (DBNet)', value: 'default' }
]

// Store
const settingsStore = useSettingsStore()

// 本地设置状态（用于双向绑定）
const settings = reactive({
  textDetector: settingsStore.settings.textDetector,
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

// 监听本地设置变化，同步到 store
watch(() => settings.textDetector, (value) => {
  settingsStore.setTextDetector(value as 'ctd' | 'yolo' | 'default')
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
.checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.checkbox-label input[type='checkbox'] {
  width: auto;
}
</style>
