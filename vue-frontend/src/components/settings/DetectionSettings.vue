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
  { label: 'YOLOv5', value: 'yolov5' },
  { label: 'Default (DBNet)', value: 'default' }
]

// Store
const settingsStore = useSettingsStore()

// 本地设置状态（用于双向绑定）
const settings = reactive({
  textDetector: settingsStore.settings.textDetector,
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
  settingsStore.setTextDetector(value as 'ctd' | 'yolo' | 'yolov5' | 'default')
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
