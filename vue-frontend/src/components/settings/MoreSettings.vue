<template>
  <div class="more-settings">
    <ParallelSettings />

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

    <div class="settings-group">
      <div class="settings-group-title">消除文字模式</div>
      <div class="settings-item checkbox-item">
        <label class="checkbox-label">
          <input
            type="checkbox"
            v-model="localSettings.removeTextWithOcr"
          />
          <span class="checkbox-text">同时执行 OCR 识别</span>
        </label>
        <div class="input-hint">
          开启后，消除文字模式会同时执行 OCR 识别，获取带有原文的干净背景图。
        </div>
      </div>
    </div>

    <div class="settings-group">
      <div class="settings-group-title">LAMA 修复设置</div>
      <div class="settings-item checkbox-item">
        <label class="checkbox-label">
          <input
            type="checkbox"
            v-model="localSettings.lamaDisableResize"
          />
          <span class="checkbox-text">禁用自动缩放</span>
        </label>
        <div class="input-hint">
          开启后，LAMA 修复将直接使用原图尺寸处理，可获得更高画质，但更吃性能与显存。
        </div>
      </div>
    </div>

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
          开启后，后端终端会输出更详细的调试信息，便于排查问题。
        </div>
      </div>
    </div>

    <div class="settings-group">
      <div class="settings-group-title">PDF 处理设置</div>
      <div class="settings-item">
        <label for="settingsPdfProcessingMethod">PDF 处理方式</label>
        <CustomSelect
          :model-value="localSettings.pdfProcessingMethod"
          :options="pdfMethodOptions"
          @change="handlePdfMethodChange"
        />
        <div class="input-hint">前端处理速度更快，后端处理兼容性更好。</div>
      </div>
    </div>

    <div class="settings-group">
      <div class="settings-group-title">默认文字设置</div>
      <div class="input-hint">
        这里配置的是「翻译设置 → 文字设置」默认值。每次新书/新翻译开始前都会优先加载这些内容，之后仍可在翻译页手动调整。
      </div>

      <div class="settings-grid">
        <div class="settings-item">
          <label for="defaultFontSize">默认字号</label>
          <input
            id="defaultFontSize"
            type="number"
            min="10"
            max="200"
            :value="defaultTextStyle.fontSize"
            @input="updateDefaultNumber('fontSize', $event)"
          />
        </div>

        <div class="settings-item checkbox-item compact-checkbox-item">
          <label class="checkbox-label">
            <input
              type="checkbox"
              :checked="defaultTextStyle.autoFontSize"
              @change="updateDefaultBoolean('autoFontSize', $event)"
            />
            <span class="checkbox-text">默认启用自动字号</span>
          </label>
        </div>

        <div class="settings-item settings-item-span-2">
          <label for="defaultFontFamily">默认字体</label>
          <CustomSelect
            :model-value="defaultTextStyle.fontFamily"
            :groups="fontSelectGroups"
            :searchable="true"
            search-placeholder="????"
            no-results-text="???????"
            @change="handleDefaultFontChange"
          />
        </div>

        <div class="settings-item">
          <label for="defaultLayoutDirection">排版方向</label>
          <CustomSelect
            :model-value="defaultTextStyle.layoutDirection"
            :options="layoutDirectionOptions"
            @change="handleDefaultLayoutDirectionChange"
          />
        </div>

        <div class="settings-item">
          <label for="defaultTextAlign">文字对齐</label>
          <CustomSelect
            :model-value="defaultTextStyle.textAlign"
            :options="textAlignOptions"
            @change="handleDefaultTextAlignChange"
          />
        </div>

        <div class="settings-item">
          <label for="defaultTextColor">文字颜色</label>
          <input
            id="defaultTextColor"
            type="color"
            class="color-input"
            :value="defaultTextStyle.textColor"
            :disabled="defaultTextStyle.useAutoTextColor"
            @input="updateDefaultColor('textColor', $event)"
          />
        </div>

        <div class="settings-item checkbox-item compact-checkbox-item">
          <label class="checkbox-label">
            <input
              type="checkbox"
              :checked="defaultTextStyle.useAutoTextColor"
              @change="updateDefaultBoolean('useAutoTextColor', $event)"
            />
            <span class="checkbox-text">默认启用自动文字颜色</span>
          </label>
        </div>

        <div class="settings-item">
          <label for="defaultInpaintMethod">修复方式</label>
          <CustomSelect
            :model-value="defaultTextStyle.inpaintMethod"
            :options="inpaintMethodOptions"
            @change="handleDefaultInpaintMethodChange"
          />
        </div>

        <div class="settings-item">
          <label for="defaultFillColor">纯色填充颜色</label>
          <input
            id="defaultFillColor"
            type="color"
            class="color-input"
            :value="defaultTextStyle.fillColor"
            @input="updateDefaultColor('fillColor', $event)"
          />
          <div class="input-hint compact-hint">仅在“纯色填充”修复方式下生效。</div>
        </div>

        <div class="settings-item checkbox-item compact-checkbox-item settings-item-span-2">
          <label class="checkbox-label">
            <input
              type="checkbox"
              :checked="defaultTextStyle.strokeEnabled"
              @change="updateDefaultBoolean('strokeEnabled', $event)"
            />
            <span class="checkbox-text">默认启用描边</span>
          </label>
        </div>

        <template v-if="defaultTextStyle.strokeEnabled">
          <div class="settings-item">
            <label for="defaultStrokeColor">描边颜色</label>
            <input
              id="defaultStrokeColor"
              type="color"
              class="color-input"
              :value="defaultTextStyle.strokeColor"
              @input="updateDefaultColor('strokeColor', $event)"
            />
          </div>

          <div class="settings-item">
            <label for="defaultStrokeWidth">描边宽度</label>
            <input
              id="defaultStrokeWidth"
              type="number"
              min="0"
              max="10"
              :value="defaultTextStyle.strokeWidth"
              @input="updateDefaultNumber('strokeWidth', $event)"
            />
          </div>
        </template>
      </div>

      <div class="default-text-actions">
        <button class="btn btn-secondary" type="button" @click="applyDefaultToCurrent">
          将默认文字设置应用到当前翻译设置
        </button>
      </div>
    </div>

    <div class="settings-group">
      <div class="settings-group-title">字体管理</div>
      <div class="settings-row">
        <div class="settings-item">
          <button class="btn btn-secondary" @click="refreshFontList" :disabled="isLoadingFonts">
            {{ isLoadingFonts ? '加载中...' : '刷新字体列表' }}
          </button>
          <div class="font-count">共 {{ fontList.length }} 个字体（含内置 / 系统 / 自定义）</div>
        </div>
        <div class="settings-item">
          <button class="btn btn-secondary" type="button" @click="triggerFontUpload">
            上传自定义字体
          </button>
          <div class="input-hint">支持 .ttf / .ttc / .otf，上传后可在翻译页与默认文字设置中直接使用。</div>
        </div>
      </div>
      <input
        ref="fontInput"
        type="file"
        accept=".ttf,.ttc,.otf"
        class="hidden-file-input"
        @change="handleFontUpload"
      />
    </div>

    <div class="settings-group">
      <div class="settings-group-title">缓存清理</div>
      <div class="settings-row">
        <div class="settings-item">
          <button class="btn btn-secondary" @click="cleanDebugFiles" :disabled="isCleaning">
            {{ isCleaning ? '清理中...' : '清理调试文件' }}
          </button>
          <div class="input-hint">清理调试过程中生成的临时文件。</div>
        </div>
        <div class="settings-item">
          <button class="btn btn-secondary" @click="cleanTempFiles" :disabled="isCleaning">
            {{ isCleaning ? '清理中...' : '清理临时文件' }}
          </button>
          <div class="input-hint">清理下载与处理过程中产生的临时缓存。</div>
        </div>
      </div>
    </div>

    <div class="settings-group">
      <div class="settings-group-title">关于</div>
      <div class="about-info">
        <p><strong>Saber-Translator</strong></p>
        <p>AI 驱动的漫画翻译工具</p>
        <p class="links">
          <a href="http://www.mashirosaber.top" target="_blank" rel="noreferrer">使用教程</a>
          <a href="https://github.com/MashiroSaber/saber-translator" target="_blank" rel="noreferrer">GitHub</a>
        </p>
        <p class="disclaimer">本项目完全开源免费，请勿上当受骗。</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'
import { configApi } from '@/api/config'
import * as systemApi from '@/api/system'
import { useToast } from '@/utils/toast'
import CustomSelect from '@/components/common/CustomSelect.vue'
import ParallelSettings from './ParallelSettings.vue'
import type { FontInfo } from '@/types/api'
import type { TextDirection, TextAlign, InpaintMethod } from '@/types/bubble'
import type { TextStyleSettings } from '@/types/settings'
import { normalizeFontList, ensureFontPresent, createFontSelectGroups } from '@/utils/fontUtils'

const pdfMethodOptions = [
  { label: '前端 pdf.js（推荐）', value: 'frontend' },
  { label: '后端 PyMuPDF', value: 'backend' }
]

const layoutDirectionOptions = [
  { label: '自动（根据检测）', value: 'auto' },
  { label: '竖向排版', value: 'vertical' },
  { label: '横向排版', value: 'horizontal' }
]

const textAlignOptions = [
  { label: '左对齐', value: 'left' },
  { label: '居中对齐', value: 'center' },
  { label: '右对齐', value: 'right' },
  { label: '两侧对齐', value: 'justify' }
]

const inpaintMethodOptions = [
  { label: '纯色填充', value: 'solid' },
  { label: 'LAMA 修复（漫画）', value: 'lama_mpe' },
  { label: 'LAMA 修复（通用）', value: 'litelama' }
]

const CUSTOM_FONT_UPLOAD_VALUE = '__upload_custom_font__'

const settingsStore = useSettingsStore()
const toast = useToast()

const isLoadingFonts = ref(false)
const fontList = ref<FontInfo[]>([])
const isCleaning = ref(false)
const fontInput = ref<HTMLInputElement | null>(null)

const localSettings = ref({
  pdfProcessingMethod: settingsStore.settings.pdfProcessingMethod || 'frontend',
  autoSaveInBookshelfMode: settingsStore.settings.autoSaveInBookshelfMode || false,
  removeTextWithOcr: settingsStore.settings.removeTextWithOcr || false,
  enableVerboseLogs: settingsStore.settings.enableVerboseLogs || false,
  lamaDisableResize: settingsStore.settings.lamaDisableResize || false
})

const defaultTextStyle = computed(() => settingsStore.defaultTextStyle)
const fontSelectGroups = computed(() => {
  const fonts = ensureFontPresent(fontList.value, defaultTextStyle.value.fontFamily)
  return [
    ...createFontSelectGroups(fonts),
    {
      label: '操作',
      options: [{ label: '上传自定义字体...', value: CUSTOM_FONT_UPLOAD_VALUE }]
    }
  ]
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

onMounted(async () => {
  await refreshFontList()
})

function handlePdfMethodChange(value: string | number) {
  localSettings.value.pdfProcessingMethod = String(value) as 'frontend' | 'backend'
}

function updateDefaultStyle(updates: Partial<TextStyleSettings>) {
  settingsStore.updateDefaultTextStyle(updates)
}

function updateDefaultNumber(key: 'fontSize' | 'strokeWidth', event: Event) {
  const value = Number((event.target as HTMLInputElement).value)
  if (!Number.isNaN(value)) {
    updateDefaultStyle({ [key]: value } as Partial<TextStyleSettings>)
  }
}

function updateDefaultBoolean(
  key: 'autoFontSize' | 'useAutoTextColor' | 'strokeEnabled',
  event: Event
) {
  updateDefaultStyle({ [key]: (event.target as HTMLInputElement).checked } as Partial<TextStyleSettings>)
}

function updateDefaultColor(key: 'textColor' | 'fillColor' | 'strokeColor', event: Event) {
  updateDefaultStyle({ [key]: (event.target as HTMLInputElement).value } as Partial<TextStyleSettings>)
}

function handleDefaultFontChange(value: string | number) {
  const fontValue = String(value)
  if (fontValue === CUSTOM_FONT_UPLOAD_VALUE) {
    triggerFontUpload()
    return
  }
  updateDefaultStyle({ fontFamily: fontValue })
}

function handleDefaultLayoutDirectionChange(value: string | number) {
  updateDefaultStyle({ layoutDirection: String(value) as TextDirection })
}

function handleDefaultTextAlignChange(value: string | number) {
  updateDefaultStyle({ textAlign: String(value) as TextAlign })
}

function handleDefaultInpaintMethodChange(value: string | number) {
  updateDefaultStyle({ inpaintMethod: String(value) as InpaintMethod })
}

function applyDefaultToCurrent() {
  settingsStore.applyDefaultTextStyleToCurrent()
  toast.success('已将默认文字设置应用到当前翻译设置')
}

function triggerFontUpload() {
  fontInput.value?.click()
}

async function refreshFontList() {
  isLoadingFonts.value = true
  try {
    const result = await configApi.getFontList()
    fontList.value = ensureFontPresent(normalizeFontList(result.fonts), defaultTextStyle.value.fontFamily)
    toast.success(`已获取 ${fontList.value.length} 个字体`)
  } catch (error: unknown) {
    fontList.value = ensureFontPresent([], defaultTextStyle.value.fontFamily)
    const errorMessage = error instanceof Error ? error.message : '获取字体列表失败'
    toast.error(errorMessage)
  } finally {
    isLoadingFonts.value = false
  }
}

async function handleFontUpload(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return

  const validExtensions = ['.ttf', '.ttc', '.otf']
  const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'))
  if (!validExtensions.includes(ext)) {
    toast.error('不支持的字体格式，请上传 .ttf、.ttc 或 .otf 文件')
    target.value = ''
    return
  }

  try {
    const result = await configApi.uploadFont(file)
    if (result.success && result.fontPath) {
      updateDefaultStyle({ fontFamily: result.fontPath })
      toast.success(`字体“${result.fontPath}”上传成功`)
      await refreshFontList()
    } else {
      toast.error(result.error || '字体上传失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '字体上传失败'
    toast.error(errorMessage)
  } finally {
    if (fontInput.value) {
      fontInput.value.value = ''
    }
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
.more-settings {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.settings-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-top: 14px;
}

.settings-item-span-2 {
  grid-column: span 2;
}

.settings-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.default-text-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}

.hidden-file-input {
  display: none;
}

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
  flex-wrap: wrap;
}

.about-info .links a {
  color: var(--color-primary);
  text-decoration: none;
}

.about-info .links a:hover {
  text-decoration: underline;
}

.about-info .disclaimer {
  color: var(--warning-color, #f0ad4e);
  font-weight: 500;
}

.checkbox-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.compact-checkbox-item {
  justify-content: flex-end;
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
  accent-color: var(--color-primary);
}

.checkbox-text {
  color: var(--text-primary);
}

.hint-note,
.compact-hint {
  color: var(--warning-color, #f0ad4e);
  font-size: 12px;
}

.color-input {
  width: 72px;
  height: 38px;
  padding: 2px;
  border: 1px solid var(--border-color, #d0d7de);
  border-radius: 8px;
  background: var(--card-bg-color, #fff);
}

@media (max-width: 768px) {
  .settings-grid,
  .settings-row {
    grid-template-columns: 1fr;
  }

  .settings-item-span-2 {
    grid-column: span 1;
  }

  .default-text-actions {
    justify-content: stretch;
  }

  .default-text-actions .btn {
    width: 100%;
  }
}
</style>
