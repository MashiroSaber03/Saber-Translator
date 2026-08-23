<template>
  <div class="more-settings">
    <ParallelSettings />

    <ProductFormSection>
      <template #title>下载设置</template>
      <UiField
        variant="settings"
        control="checkbox"
        hint="开启后，单张与批量图片导出不再添加 translated、clean 或 original 前缀；文件扩展名仍与实际导出格式一致。"
      >
        <UiCheckbox
          :model-value="settingsStore.exportPreferences.preserveOriginalFilenames"
          label="保持原文件名"
          @change="settingsStore.exportPreferences.preserveOriginalFilenames = $event"
        />
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>自定义 OpenAI 服务</template>
      <UiField
        variant="settings"
        label="配置管理"
        hint="统一管理可在翻译、OCR、插件和漫画分析中复用的 Base URL、API Key 与模型名。"
      >
        <UiButton variant="secondary" type="button" @click="customAiProfileManagerOpen = true">
          管理自定义服务
        </UiButton>
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>消除文字模式</template>
      <UiField
        variant="settings"
        control="checkbox"
        hint="开启后，消除文字模式会同时执行OCR识别，获取带有原文的干净背景图。适用于需要保留原文信息以便后续翻译或参考的场景。"
      >
        <UiCheckbox
          :model-value="settingsStore.settings.removeTextWithOcr"
          label="同时执行OCR识别"
          @change="settingsStore.setRemoveTextWithOcr"
        />
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>视觉模型图片</template>
      <UiField
        variant="settings"
        control="checkbox"
        hint="开启后，高质量翻译、AI 校对和 AI 视觉 OCR 会使用高质量 JPEG 传图，显著减少传输体积；关闭后使用无损 PNG。此设置不会改变后端保存的原图或导出文件。"
      >
        <UiCheckbox
          :model-value="settingsStore.settings.compressVisionImages"
          label="压缩发送给视觉模型的图片"
          @change="settingsStore.setCompressVisionImages"
        />
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>LAMA 修复设置</template>
      <UiField
        variant="settings"
        control="checkbox"
        hint="开启后，LAMA 修复将使用原图尺寸进行处理（不缩放到1024px），可获得更高画质。需要更强的 GPU 和更多显存，处理速度会变慢。推荐 RTX 4060 或更高配置使用。适用于两种LAMA修复方法（速度优化和通用）。"
      >
        <UiCheckbox
          :model-value="lamaDisableResizeValue"
          :disabled="!lamaDisableResizeEditable"
          label="禁用自动缩放"
          @change="settingsStore.setLamaDisableResize"
        />
      </UiField>
      <p v-if="!lamaDisableResizeEditable" class="more-settings__policy-note">
        该选项由管理员统一设置，普通用户不能修改。
      </p>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>调试选项</template>
      <UiField
        variant="settings"
        control="checkbox"
        hint="开启后，后端终端会打印详细的诊断日志（包括完整的消息结构、模型响应等），便于调试问题。影响所有翻译模式，默认关闭以保持日志简洁。"
      >
        <UiCheckbox
          :model-value="settingsStore.settings.enableVerboseLogs"
          label="详细日志"
          @change="settingsStore.setEnableVerboseLogs"
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
        :hint="`支持 ${FONT_FILE_FORMATS_LABEL} 格式`"
      >
        <div class="more-settings__font-upload-row">
          <UiFileInput
            ref="fontInput"
            data-testid="font-upload-input"
            class="more-settings__hidden-file-input"
            :accept="FONT_FILE_ACCEPT"
            :disabled="isLoadingFonts || isUploadingFont"
            @files-change="handleFontUpload"
          />
          <UiButton
            variant="secondary"
            type="button"
            data-testid="font-upload-trigger"
            :disabled="isLoadingFonts || isUploadingFont"
            @click="triggerFontUpload"
          >
            {{ isUploadingFont ? '上传中...' : isLoadingFonts ? '字体列表加载中...' : '选择字体文件' }}
          </UiButton>
          <span class="more-settings__font-upload-filename" data-testid="font-upload-filename">
            {{ selectedFontFileName || '未选择文件' }}
          </span>
        </div>
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>存储维护</template>
      <UiField
        variant="settings"
        label="修复临时文件记录"
        hint="恢复中断的文件写入，并清理过期的未完成临时文件"
      >
        <UiButton variant="secondary" @click="recoverAssetJournal" :disabled="isRecoveringAssets">
          {{ isRecoveringAssets ? '修复中...' : '检查并修复' }}
        </UiButton>
      </UiField>
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

    <CustomAiProfileManager
      v-if="customAiProfileManagerOpen"
      v-model="customAiProfileManagerOpen"
    />
  </div>
</template>

<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import { computed, ref } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import {
  cleanV2TempFiles,
  listV2Fonts,
  uploadV2Font,
  type V2Font,
} from '@/api/v2/settings'
import { useToast } from '@/utils/toast'
import {
  FONT_FILE_ACCEPT,
  FONT_FILE_FORMATS_LABEL,
  isSupportedFontFileName,
} from '@/utils/fontFiles'
import ParallelSettings from './ParallelSettings.vue'
import CustomAiProfileManager from './CustomAiProfileManager.vue'
import { usePublicUserAccess } from '@/composables/usePublicUserAccess'

const settingsStore = useSettingsStore()
const toast = useToast()
const publicAccess = usePublicUserAccess()
const lamaDisableResizeEditable = computed(() => publicAccess.lamaDisableResizeEditable())
const lamaDisableResizeValue = computed(() => (
  lamaDisableResizeEditable.value
    ? settingsStore.settings.lamaDisableResize
    : publicAccess.lamaDisableResizeValue()
))

const isLoadingFonts = ref(false)
const fontList = computed<V2Font[]>(() => settingsStore.fontCatalog)
const isUploadingFont = ref(false)
const isRecoveringAssets = ref(false)
const customAiProfileManagerOpen = ref(false)
const fontInput = ref<InstanceType<typeof UiFileInput> | null>(null)
const selectedFontFileName = ref('')

async function refreshFontList() {
  if (isLoadingFonts.value) return
  isLoadingFonts.value = true
  try {
    const fonts = await listV2Fonts()
    settingsStore.hydrateResourceCatalogs(
      fonts,
      settingsStore.promptCatalog,
    )
    toast.success(`获取到 ${fonts.length} 个字体`)
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '获取字体列表失败'
    toast.error(errorMessage)
  } finally {
    isLoadingFonts.value = false
  }
}

function triggerFontUpload() {
  if (isLoadingFonts.value || isUploadingFont.value) return
  fontInput.value?.click()
}

async function handleFontUpload(files: File[]) {
  if (isLoadingFonts.value || isUploadingFont.value) return
  const file = files[0]
  if (!file) return
  selectedFontFileName.value = file.name

  if (!isSupportedFontFileName(file.name)) {
    toast.error(`不支持的字体格式，请上传 ${FONT_FILE_FORMATS_LABEL} 文件`)
    fontInput.value?.clear()
    selectedFontFileName.value = ''
    return
  }

  isUploadingFont.value = true
  try {
    const uploadedFont = await uploadV2Font(file)
    settingsStore.upsertFont(uploadedFont)
    toast.success(`字体 "${file.name}" 上传成功`)
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '字体上传失败'
    toast.error(errorMessage)
  } finally {
    isUploadingFont.value = false
    fontInput.value?.clear()
    selectedFontFileName.value = ''
  }
}

async function recoverAssetJournal() {
  if (isRecoveringAssets.value) return
  isRecoveringAssets.value = true
  try {
    const result = await cleanV2TempFiles()
    toast.success(`已处理 ${result.recovered} 个临时文件记录`)
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '存储修复失败'
    toast.error(errorMessage)
  } finally {
    isRecoveringAssets.value = false
  }
}
</script>

<style scoped>
.more-settings__policy-note {
  margin: 8px 0 0;
  color: var(--color-text-supporting);
  font-size: 12px;
}

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
