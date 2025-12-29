<template>
  <div class="more-settings">
    <!-- PDF处理方式 -->
    <div class="settings-group">
      <div class="settings-group-title">PDF处理设置</div>
      <div class="settings-item">
        <label for="settingsPdfProcessingMethod">PDF处理方式:</label>
        <CustomSelect
          v-model="settingsStore.settings.pdfProcessingMethod"
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

    <!-- 调试选项 -->
    <div class="settings-group">
      <div class="settings-group-title">调试选项</div>
      <div class="settings-item">
        <label class="checkbox-label">
          <input type="checkbox" v-model="localSettings.showDetectionDebug" @change="saveSettings" />
          显示检测框调试信息
        </label>
        <div class="input-hint">开启后会在翻译结果中显示气泡检测框</div>
      </div>
    </div>

    <!-- 重试次数配置 -->
    <div class="settings-group">
      <div class="settings-group-title">重试次数配置</div>
      <div class="settings-item">
        <label for="settingsTranslationMaxRetries">普通翻译重试次数:</label>
        <input 
          type="number" 
          id="settingsTranslationMaxRetries" 
          v-model.number="localSettings.translationMaxRetries" 
          min="0" 
          max="10"
          @change="saveSettings"
        />
        <div class="input-hint">翻译失败时的最大重试次数（0-10）</div>
      </div>
      <div class="settings-item">
        <label for="settingsHqTranslationMaxRetries">高质量翻译重试次数:</label>
        <input 
          type="number" 
          id="settingsHqTranslationMaxRetries" 
          v-model.number="localSettings.hqTranslationMaxRetries" 
          min="0" 
          max="10"
          @change="saveSettings"
        />
        <div class="input-hint">高质量翻译失败时的最大重试次数（0-10）</div>
      </div>
      <div class="settings-item">
        <label for="settingsProofreadingMaxRetries">AI校对重试次数:</label>
        <input 
          type="number" 
          id="settingsProofreadingMaxRetries" 
          v-model.number="localSettings.proofreadingMaxRetries" 
          min="0" 
          max="10"
          @change="saveSettings"
        />
        <div class="input-hint">AI校对失败时的最大重试次数（0-10）</div>
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
import { ref } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'
import { configApi } from '@/api/config'
import * as systemApi from '@/api/system'
import CustomSelect from '@/components/common/CustomSelect.vue'

/** PDF处理方式选项 */
const pdfMethodOptions = [
  { label: '前端 pdf.js (推荐)', value: 'frontend' },
  { label: '后端 PyMuPDF', value: 'backend' }
]
import { useToast } from '@/utils/toast'

// Store
const settingsStore = useSettingsStore()
// settings 计算属性已移除，直接使用 settingsStore
const toast = useToast()

// 状态
const isLoadingFonts = ref(false)
const fontList = ref<(string | import('@/types').FontInfo)[]>([])
const isCleaning = ref(false)
const fontInput = ref<HTMLInputElement | null>(null)

// 本地设置状态
const localSettings = ref({
  showDetectionDebug: settingsStore.settings.showDetectionDebug || false,
  translationMaxRetries: settingsStore.settings.translation?.maxRetries || 2,
  hqTranslationMaxRetries: settingsStore.settings.hqTranslation?.maxRetries || 2,
  proofreadingMaxRetries: settingsStore.settings.proofreading?.maxRetries || 2
})

// 保存设置
function saveSettings() {
  // 更新调试设置
  settingsStore.setShowDetectionDebug(localSettings.value.showDetectionDebug)
  
  // 更新翻译重试次数
  if (settingsStore.settings.translation) {
    settingsStore.settings.translation.maxRetries = localSettings.value.translationMaxRetries
  }
  
  // 更新高质量翻译重试次数
  if (settingsStore.settings.hqTranslation) {
    settingsStore.settings.hqTranslation.maxRetries = localSettings.value.hqTranslationMaxRetries
  }
  
  // 更新校对重试次数
  settingsStore.setProofreadingMaxRetries(localSettings.value.proofreadingMaxRetries)
}

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
</style>
