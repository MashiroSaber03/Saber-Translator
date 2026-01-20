<template>
  <div class="more-settings">
    <!-- 并行翻译设置 -->
    <ParallelSettings />

    <!-- 书架模式自动保存 -->
    <div class="settings-group">
      <div class="settings-group-title">自动保存设置</div>
      <div class="settings-item checkbox-item">
        <label class="checkbox-label">
          <input 
            type="checkbox" 
            v-model="localSettings.autoSaveInBookshelfMode"
          />
          <span class="checkbox-text">书架模式自动保存</span>
        </label>
        <div class="input-hint">
          开启后，在书架模式下翻译时会自动保存进度（翻译一张保存一张），防止意外关闭导致数据丢失。
          <br />
          <span class="hint-note">注意：此功能仅在书架模式下生效，快速翻译模式不支持。</span>
        </div>
      </div>
    </div>

    <!-- 消除文字模式设置 -->
    <div class="settings-group">
      <div class="settings-group-title">消除文字模式</div>
      <div class="settings-item checkbox-item">
        <label class="checkbox-label">
          <input 
            type="checkbox" 
            v-model="localSettings.removeTextWithOcr"
          />
          <span class="checkbox-text">同时执行OCR识别</span>
        </label>
        <div class="input-hint">
          开启后，消除文字模式会同时执行OCR识别，获取带有原文的干净背景图。
          <br />
          <span class="hint-note">适用于需要保留原文信息以便后续翻译或参考的场景。</span>
        </div>
      </div>
    </div>

    <!-- 详细日志设置 -->
    <div class="settings-group">
      <div class="settings-group-title">调试选项</div>
      <div class="settings-item checkbox-item">
        <label class="checkbox-label">
          <input 
            type="checkbox" 
            v-model="localSettings.enableVerboseLogs"
          />
          <span class="checkbox-text">详细日志</span>
        </label>
        <div class="input-hint">
          开启后，后端终端会打印详细的诊断日志（包括完整的消息结构、模型响应等），便于调试问题。
          <br />
          <span class="hint-note">影响所有翻译模式，默认关闭以保持日志简洁。</span>
        </div>
      </div>
    </div>

    <!-- PDF处理方式 -->
    <div class="settings-group">
      <div class="settings-group-title">PDF处理设置</div>
      <div class="settings-item">
        <label for="settingsPdfProcessingMethod">PDF处理方式:</label>
        <CustomSelect
          v-model="localSettings.pdfProcessingMethod"
          :options="pdfMethodOptions"
        />
        <div class="input-hint">前端处理速度更快，后端处理兼容性更好</div>
      </div>
    </div>

    <!-- 字体设置 -->
    <div class="settings-group">
      <div class="settings-group-title">字体设置</div>
      <div class="settings-item">
        <label>系统字体列表:</label>
        <button class="btn btn-secondary" @click="refreshFontList" :disabled="isLoadingFonts">
          {{ isLoadingFonts ? '加载中...' : '🔄 刷新字体列表' }}
        </button>
        <div v-if="fontList.length > 0" class="font-count">共 {{ fontList.length }} 个字体</div>
      </div>
      <div class="settings-item">
        <label>上传自定义字体:</label>
        <input type="file" accept=".ttf,.ttc,.otf" @change="handleFontUpload" ref="fontInput" />
        <div class="input-hint">支持 .ttf, .ttc, .otf 格式</div>
      </div>
    </div>

    <!-- 缓存清理 -->
    <div class="settings-group">
      <div class="settings-group-title">缓存清理</div>
      <div class="settings-row">
        <div class="settings-item">
          <button class="btn btn-secondary" @click="cleanDebugFiles" :disabled="isCleaning">
            {{ isCleaning ? '清理中...' : '🗑️ 清理调试文件' }}
          </button>
          <div class="input-hint">清理调试过程中生成的临时文件</div>
        </div>
        <div class="settings-item">
          <button class="btn btn-secondary" @click="cleanTempFiles" :disabled="isCleaning">
            {{ isCleaning ? '清理中...' : '🗑️ 清理临时文件' }}
          </button>
          <div class="input-hint">清理下载和处理过程中的临时文件</div>
        </div>
      </div>
    </div>

    <!-- 关于 -->
    <div class="settings-group">
      <div class="settings-group-title">关于</div>
      <div class="about-info">
        <p><strong>Saber-Translator</strong></p>
        <p>AI驱动的漫画翻译工具</p>
        <p class="links">
          <a href="http://www.mashirosaber.top" target="_blank">📖 使用教程</a>
          <a href="https://github.com/MashiroSaber/saber-translator" target="_blank">🐙 GitHub</a>
        </p>
        <p class="disclaimer">本项目完全开源免费，请勿上当受骗</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 更多设置组件
 * 管理PDF处理、字体、缓存清理等杂项设置
 */
import { ref, watch } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'
import { configApi } from '@/api/config'
import * as systemApi from '@/api/system'
import { useToast } from '@/utils/toast'
import CustomSelect from '@/components/common/CustomSelect.vue'
import ParallelSettings from './ParallelSettings.vue'

/** PDF处理方式选项 */
const pdfMethodOptions = [
  { label: '前端 pdf.js (推荐)', value: 'frontend' },
  { label: '后端 PyMuPDF', value: 'backend' }
]

// Store
const settingsStore = useSettingsStore()
const toast = useToast()

// 状态
const isLoadingFonts = ref(false)
const fontList = ref<(string | import('@/types').FontInfo)[]>([])
const isCleaning = ref(false)
const fontInput = ref<HTMLInputElement | null>(null)

// 本地设置状态（用于双向绑定，修改后自动同步到 store）
const localSettings = ref({
  pdfProcessingMethod: settingsStore.settings.pdfProcessingMethod || 'frontend',
  autoSaveInBookshelfMode: settingsStore.settings.autoSaveInBookshelfMode || false,
  removeTextWithOcr: settingsStore.settings.removeTextWithOcr || false,
  enableVerboseLogs: settingsStore.settings.enableVerboseLogs || false
})

// ============================================================
// Watch 同步：本地状态变化时自动保存到 store
// ============================================================
watch(() => localSettings.value.pdfProcessingMethod, (val) => {
  settingsStore.setPdfProcessingMethod(val as 'frontend' | 'backend')
})

watch(() => localSettings.value.autoSaveInBookshelfMode, (val) => {
  settingsStore.setAutoSaveInBookshelfMode(val)
})

watch(() => localSettings.value.removeTextWithOcr, (val) => {
  settingsStore.setRemoveTextWithOcr(val)
})

watch(() => localSettings.value.enableVerboseLogs, (val) => {
  settingsStore.setEnableVerboseLogs(val)
})

// 刷新字体列表
async function refreshFontList() {
  isLoadingFonts.value = true
  try {
    const result = await configApi.getFontList()
    fontList.value = result.fonts || []
    toast.success(`获取到 ${fontList.value.length} 个字体`)
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '获取字体列表失败'
    toast.error(errorMessage)
  } finally {
    isLoadingFonts.value = false
  }
}

// 上传自定义字体
async function handleFontUpload(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return

  // 验证文件类型
  const validExtensions = ['.ttf', '.ttc', '.otf']
  const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'))
  if (!validExtensions.includes(ext)) {
    toast.error('不支持的字体格式，请上传 .ttf, .ttc 或 .otf 文件')
    return
  }

  try {
    const result = await configApi.uploadFont(file)
    if (result.success) {
      toast.success(`字体 "${result.fontPath || file.name}" 上传成功`)
      // 刷新字体列表
      await refreshFontList()
    } else {
      toast.error(result.error || '字体上传失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '字体上传失败'
    toast.error(errorMessage)
  } finally {
    // 清空文件输入
    if (fontInput.value) {
      fontInput.value.value = ''
    }
  }
}

// 清理调试文件
async function cleanDebugFiles() {
  isCleaning.value = true
  try {
    const result = await systemApi.cleanDebugFiles() as { success: boolean; deleted_count?: number; error?: string }
    if (result.success) {
      toast.success(`已清理 ${result.deleted_count || 0} 个调试文件`)
    } else {
      toast.error(result.error || '清理失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '清理失败'
    toast.error(errorMessage)
  } finally {
    isCleaning.value = false
  }
}

// 清理临时文件
async function cleanTempFiles() {
  isCleaning.value = true
  try {
    const result = await systemApi.cleanTempFiles() as { success: boolean; deleted_count?: number; error?: string }
    if (result.success) {
      toast.success(`已清理 ${result.deleted_count || 0} 个临时文件`)
    } else {
      toast.error(result.error || '清理失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '清理失败'
    toast.error(errorMessage)
  } finally {
    isCleaning.value = false
  }
}
</script>

<style scoped>
.font-count {
  margin-top: 8px;
  font-size: 13px;
  color: var(--text-secondary);
}

.about-info {
  padding: 15px;
  background: var(--bg-secondary);
  border-radius: 8px;
}

.about-info p {
  margin: 8px 0;
}

.about-info .links {
  display: flex;
  gap: 20px;
}

.about-info .links a {
  color: var(--primary-color);
  text-decoration: none;
}

.about-info .links a:hover {
  text-decoration: underline;
}

.about-info .disclaimer {
  color: var(--warning-color, #f0ad4e);
  font-weight: 500;
}

/* 复选框设置项样式 */
.checkbox-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  font-weight: 500;
}

.checkbox-label input[type="checkbox"] {
  width: 18px;
  height: 18px;
  cursor: pointer;
  accent-color: var(--primary-color);
}

.checkbox-text {
  color: var(--text-primary);
}

.hint-note {
  color: var(--warning-color, #f0ad4e);
  font-size: 12px;
}
</style>
