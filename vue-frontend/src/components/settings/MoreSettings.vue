<template>
  <div class="more-settings">
    <ParallelSettings />

    <ProductFormSection>
      <template #title>自动保存设置</template>
      <UiField
        variant="settings"
        control="checkbox"
        hint="开启后，在书架模式下翻译时会自动保存进度（翻译一张保存一张），防止意外关闭导致数据丢失。注意：此功能仅在书架模式下生效，快速翻译模式不支持。"
      >
        <UiCheckbox v-model="localSettings.autoSaveInBookshelfMode" label="书架模式自动保存" />
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>消除文字模式</template>
      <UiField
        variant="settings"
        control="checkbox"
        hint="开启后，消除文字模式会同时执行OCR识别，获取带有原文的干净背景图。适用于需要保留原文信息以便后续翻译或参考的场景。"
      >
        <UiCheckbox v-model="localSettings.removeTextWithOcr" label="同时执行OCR识别" />
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>LAMA 修复设置</template>
      <UiField
        variant="settings"
        control="checkbox"
        hint="开启后，LAMA 修复将使用原图尺寸进行处理（不缩放到1024px），可获得更高画质。需要更强的 GPU 和更多显存，处理速度会变慢。推荐 RTX 4060 或更高配置使用。适用于两种LAMA修复方法（速度优化和通用）。"
      >
        <UiCheckbox v-model="localSettings.lamaDisableResize" label="禁用自动缩放" />
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>调试选项</template>
      <UiField
        variant="settings"
        control="checkbox"
        hint="开启后，后端终端会打印详细的诊断日志（包括完整的消息结构、模型响应等），便于调试问题。影响所有翻译模式，默认关闭以保持日志简洁。"
      >
        <UiCheckbox v-model="localSettings.enableVerboseLogs" label="详细日志" />
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>PDF处理设置</template>
      <UiField
        variant="settings"
        label="PDF处理方式"
        control-id="settingsPdfProcessingMethod"
        hint="前端处理速度更快，后端处理适配性更好"
      >
        <UiSelect
          id="settingsPdfProcessingMethod"
          v-model="localSettings.pdfProcessingMethod"
          :options="pdfMethodOptions"
        />
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>字体设置</template>
      <UiField variant="settings" label="系统字体列表">
        <UiButton variant="secondary" @click="refreshFontList" :disabled="isLoadingFonts">
          {{ isLoadingFonts ? '加载中...' : '刷新字体列表' }}
        </UiButton>
        <div v-if="fontList.length > 0" class="more-settings__font-count">共 {{ fontList.length }} 个字体</div>
      </UiField>
      <UiField
        variant="settings"
        label="上传自定义字体"
        hint="支持 .ttf, .ttc, .otf 格式"
      >
        <div class="more-settings__font-upload-row">
          <UiFileInput
            ref="fontInput"
            data-testid="font-upload-input"
            class="more-settings__hidden-file-input"
            accept=".ttf,.ttc,.otf"
            @files-change="handleFontUpload"
          />
          <UiButton
            variant="secondary"
            type="button"
            data-testid="font-upload-trigger"
            @click="triggerFontUpload"
          >
            选择字体文件
          </UiButton>
          <span class="more-settings__font-upload-filename" data-testid="font-upload-filename">
            {{ selectedFontFileName || '未选择文件' }}
          </span>
        </div>
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>缓存清理</template>
      <UiFormGrid>
        <UiField
          variant="settings"
          label="清理调试文件"
          hint="清理调试过程中生成的临时文件"
        >
          <UiButton variant="secondary" @click="cleanDebugFiles" :disabled="isCleaning">
            {{ isCleaning ? '清理中...' : '清理调试文件' }}
          </UiButton>
        </UiField>
        <UiField
          variant="settings"
          label="清理临时文件"
          hint="清理下载和处理过程中的临时文件"
        >
          <UiButton variant="secondary" @click="cleanTempFiles" :disabled="isCleaning">
            {{ isCleaning ? '清理中...' : '清理临时文件' }}
          </UiButton>
        </UiField>
      </UiFormGrid>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>关于</template>
      <div class="more-settings__about">
        <p class="more-settings__about-title"><strong>Saber-Translator</strong></p>
        <p class="more-settings__about-description">AI驱动的漫画翻译工具</p>
        <p class="more-settings__about-links">
          <a class="more-settings__about-link" href="http://www.mashirosaber.top" target="_blank" rel="noopener noreferrer">使用教程</a>
          <a class="more-settings__about-link" href="https://github.com/MashiroSaber/saber-translator" target="_blank" rel="noopener noreferrer">GitHub</a>
        </p>
        <p class="more-settings__about-disclaimer">本项目完全开源免费，请勿上当受骗</p>
      </div>
    </ProductFormSection>
  </div>
</template>

<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { ref, watch } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import { configApi } from '@/api/config'
import * as systemApi from '@/api/system'
import { useToast } from '@/utils/toast'
import ParallelSettings from './ParallelSettings.vue'

const pdfMethodOptions = [
  { label: '前端 pdf.js (推荐)', value: 'frontend' },
  { label: '后端 PyMuPDF', value: 'backend' }
]

const settingsStore = useSettingsStore()
const toast = useToast()

const isLoadingFonts = ref(false)
const fontList = ref<import('@/types').FontInfo[]>([])
const isCleaning = ref(false)
const fontInput = ref<InstanceType<typeof UiFileInput> | null>(null)
const selectedFontFileName = ref('')

const localSettings = ref({
  pdfProcessingMethod: settingsStore.settings.pdfProcessingMethod || 'frontend',
  autoSaveInBookshelfMode: settingsStore.settings.autoSaveInBookshelfMode || false,
  removeTextWithOcr: settingsStore.settings.removeTextWithOcr || false,
  enableVerboseLogs: settingsStore.settings.enableVerboseLogs || false,
  lamaDisableResize: settingsStore.settings.lamaDisableResize || false
})

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

watch(() => localSettings.value.lamaDisableResize, (val) => {
  settingsStore.setLamaDisableResize(val)
})

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

function triggerFontUpload() {
  fontInput.value?.click()
}

async function handleFontUpload(files: File[]) {
  const file = files[0]
  if (!file) return
  selectedFontFileName.value = file.name

  const validExtensions = ['.ttf', '.ttc', '.otf']
  const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'))
  if (!validExtensions.includes(ext)) {
    toast.error('不支持的字体格式，请上传 .ttf, .ttc 或 .otf 文件')
    fontInput.value?.clear()
    return
  }

  try {
    const result = await configApi.uploadFont(file)
    if (result.success) {
      toast.success(`字体 "${result.fontPath || file.name}" 上传成功`)
      await refreshFontList()
    } else {
      toast.error(result.error || '字体上传失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '字体上传失败'
    toast.error(errorMessage)
  } finally {
    fontInput.value?.clear()
  }
}

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
.more-settings__font-upload-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}

.more-settings__hidden-file-input {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.more-settings__font-upload-filename {
  color: var(--color-text-supporting);
  font-size: 0.95em;
}

.more-settings__font-count {
  margin-top: 8px;
  font-size: 13px;
  color: var(--color-text-supporting);
}

.more-settings__about {
  padding: 15px;
  background: var(--color-surface-subtle);
  border-radius: 8px;
}

.more-settings__about-title,
.more-settings__about-description,
.more-settings__about-links,
.more-settings__about-disclaimer {
  margin: 8px 0;
}

.more-settings__about-links {
  display: flex;
  gap: 20px;
}

.more-settings__about-link {
  color: var(--color-action-primary);
  text-decoration: none;
}

.more-settings__about-link:hover {
  text-decoration: underline;
}

.more-settings__about-disclaimer {
  color: var(--color-status-warning);
  font-weight: 500;
}
</style>
