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
          @change="handleDetectorChange"
        />
      </div>
    </div>

    <!-- 文本框扩展参数 -->
    <div class="settings-group">
      <div class="settings-group-title">文本框扩展参数</div>
      <div class="settings-item">
        <label for="settingsBoxExpandRatio">整体扩展 (像素):</label>
        <input type="number" id="settingsBoxExpandRatio" v-model.number="settings.boxExpandRatio" min="0" step="1" />
        <div class="input-hint">向四周均匀扩展的像素数</div>
      </div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="settingsBoxExpandTop">上方扩展:</label>
          <input type="number" id="settingsBoxExpandTop" v-model.number="settings.boxExpandTop" min="0" step="1" />
        </div>
        <div class="settings-item">
          <label for="settingsBoxExpandBottom">下方扩展:</label>
          <input type="number" id="settingsBoxExpandBottom" v-model.number="settings.boxExpandBottom" min="0" step="1" />
        </div>
      </div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="settingsBoxExpandLeft">左侧扩展:</label>
          <input type="number" id="settingsBoxExpandLeft" v-model.number="settings.boxExpandLeft" min="0" step="1" />
        </div>
        <div class="settings-item">
          <label for="settingsBoxExpandRight">右侧扩展:</label>
          <input type="number" id="settingsBoxExpandRight" v-model.number="settings.boxExpandRight" min="0" step="1" />
        </div>
      </div>
    </div>

    <!-- 精确文字掩膜设置 (仅CTD和Default支持) -->
    <div v-show="supportsPreciseMask" class="settings-group">
      <div class="settings-group-title">精确文字掩膜</div>
      <div class="settings-item">
        <label class="checkbox-label">
          <input type="checkbox" v-model="settings.usePreciseMask" />
          启用精确文字掩膜
        </label>
        <div class="input-hint">使用更精确的文字区域掩膜进行修复</div>
      </div>
      <div v-show="settings.usePreciseMask" class="settings-row">
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

    <!-- LAMA修复测试 -->
    <div class="settings-group">
      <div class="settings-group-title">修复功能测试</div>
      <button class="settings-test-btn" @click="testLamaRepair" :disabled="isTesting">
        {{ isTesting ? '测试中...' : '🔗 测试LAMA修复' }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 检测设置组件
 * 管理文字检测器和相关参数配置
 */
import { ref, reactive, computed, watch } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'
import { configApi } from '@/api/config'
import { useToast } from '@/utils/toast'
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
const toast = useToast()

// 本地设置状态（用于双向绑定）
const settings = reactive({
  textDetector: settingsStore.settings.textDetector,
  boxExpandRatio: settingsStore.settings.boxExpand.ratio,
  boxExpandTop: settingsStore.settings.boxExpand.top,
  boxExpandBottom: settingsStore.settings.boxExpand.bottom,
  boxExpandLeft: settingsStore.settings.boxExpand.left,
  boxExpandRight: settingsStore.settings.boxExpand.right,
  usePreciseMask: settingsStore.settings.preciseMask.enabled,
  maskDilateSize: settingsStore.settings.preciseMask.dilateSize,
  maskBoxExpandRatio: settingsStore.settings.preciseMask.boxExpandRatio,
  showDetectionDebug: settingsStore.settings.showDetectionDebug
})

// 测试状态
const isTesting = ref(false)

// 计算属性：是否支持精确掩膜
const supportsPreciseMask = computed(() => {
  return ['ctd', 'default'].includes(settings.textDetector)
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

watch(() => settings.usePreciseMask, (value) => {
  settingsStore.updatePreciseMask({ enabled: value })
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

// 处理检测器切换
function handleDetectorChange() {
  // 如果切换到不支持精确掩膜的检测器，自动关闭该选项
  if (!supportsPreciseMask.value) {
    settings.usePreciseMask = false
  }
}

// 测试LAMA修复
async function testLamaRepair() {
  isTesting.value = true
  try {
    const result = await configApi.testLamaRepair()
    if (result.success) {
      toast.success('LAMA修复功能正常')
    } else {
      toast.error(`LAMA修复测试失败: ${result.error || '未知错误'}`)
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '测试失败'
    toast.error(errorMessage)
  } finally {
    isTesting.value = false
  }
}
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
