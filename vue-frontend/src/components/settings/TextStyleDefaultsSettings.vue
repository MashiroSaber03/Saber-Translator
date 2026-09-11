<template>
  <div class="text-style-defaults-settings">
    <ProductFormSection>
      <template #title>文本默认值</template>
      <p class="text-style-defaults-settings__intro">
        这里修改的是后端数据库中的全局默认文字设置。
        <br />
        保存成功后，新导入页面和后续任务会使用这些默认值。
      </p>
      <ProductActionRow aria-label="文本默认值操作" justify="start">
        <UiButton
          variant="secondary"
          type="button"
          data-testid="reset-text-style-defaults"
          :disabled="isLoading"
          @click="resetDraftToFactory"
        >
          恢复出厂默认
        </UiButton>
      </ProductActionRow>
      <ProductStatusBanner v-if="errorMessage" tone="danger" role="alert">
        {{ errorMessage }}
      </ProductStatusBanner>
    </ProductFormSection>

    <TextStyleForm
      :model-value="draftDefaults"
      :font-select-options="fontSelectOptions"
      :font-disabled="isLoading || isUploadingFont"
      :inpaint-method-options="inpaintMethodOptions"
      @change="updateDraft"
      @font-change="handleFontSelectChange"
    >
      <template #font-extra>
        <UiFileInput
          ref="fontUploadInput" :accept="FONT_FILE_ACCEPT"
          :disabled="isLoading || isUploadingFont" hidden @files-change="handleFontUpload"
        />
      </template>
    </TextStyleForm>
  </div>
</template>

<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import TextStyleForm from './TextStyleForm.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { computed, ref } from 'vue'
import type { TextStyleSettings } from '@/types/settings'
import {
  getTextStyleDefaults,
  normalizeTextStyleSettings,
} from '@/defaults/textStyleDefaults'
import { listV2Fonts, uploadV2Font, type V2Font } from '@/api/v2/settings'
import { useSettingsStore } from '@/stores/settings'
import { useToast } from '@/utils/toast'
import {
  FONT_FILE_ACCEPT,
  FONT_FILE_FORMATS_LABEL,
  isSupportedFontFileName,
} from '@/utils/fontFiles'
import {
  inpaintMethodOptions as rawInpaintMethodOptions,
} from '@/utils/textStyleForm'
import { usePublicUserAccess } from '@/composables/usePublicUserAccess'

const toast = useToast()
const settingsStore = useSettingsStore()
const publicAccess = usePublicUserAccess()
const inpaintMethodOptions = computed(() => publicAccess.modelOptions(
  rawInpaintMethodOptions,
  {
    lama_mpe: 'lama_mpe',
    litelama: 'litelama',
    lama_manga: 'lama_manga',
  },
))
const draftDefaults = computed(() => settingsStore.textStyleDefaults)
const isLoading = ref(false)
const isUploadingFont = ref(false)
const errorMessage = ref('')
const fontList = computed<V2Font[]>(() => settingsStore.fontCatalog)
const fontUploadInput = ref<InstanceType<typeof UiFileInput> | null>(null)

const fontSelectOptions = computed(() => {
  const options = fontList.value.map(font => ({
    label: font.displayName,
    value: font.id,
  }))
  options.push({ label: '自定义字体...', value: 'custom-font' })
  return options
})

async function loadFontList(): Promise<void> {
  isLoading.value = true
  errorMessage.value = ''
  try {
    const fonts = await listV2Fonts()
    settingsStore.hydrateResourceCatalogs(fonts, settingsStore.promptCatalog)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '获取字体列表失败'
  } finally {
    isLoading.value = false
  }
}

void loadFontList()

function updateDraft(updates: Partial<TextStyleSettings>): void {
  const normalized = normalizeTextStyleSettings({
    ...draftDefaults.value,
    ...updates,
  })
  settingsStore.textStyleDefaults = normalized
}

function resetDraftToFactory(): void {
  const defaultFont = fontList.value.find(
    font => font.kind === 'builtin' && font.builtinKey === 'default',
  )
  if (!defaultFont) {
    errorMessage.value = '未找到内置默认字体，请刷新字体列表后重试'
    return
  }
  settingsStore.textStyleDefaults = {
    ...getTextStyleDefaults(),
    fontFamily: defaultFont.id,
  }
  errorMessage.value = ''
}

async function handleFontUpload(files: File[]): Promise<void> {
  if (isUploadingFont.value) return
  const file = files[0]
  if (!file) return

  if (!isSupportedFontFileName(file.name)) {
    toast.error(`请选择 ${FONT_FILE_FORMATS_LABEL} 格式的字体文件`)
    fontUploadInput.value?.clear()
    return
  }

  isUploadingFont.value = true
  try {
    const uploadedFont = await uploadV2Font(file)
    settingsStore.upsertFont(uploadedFont)
    updateDraft({ fontFamily: uploadedFont.id })
    toast.success('字体上传成功')
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '字体上传失败')
  } finally {
    isUploadingFont.value = false
    fontUploadInput.value?.clear()
  }
}

function handleFontSelectChange(value: string | number): void {
  if (typeof value !== 'string') return
  if (value === 'custom-font') {
    if (isUploadingFont.value) return
    fontUploadInput.value?.click()
    return
  }
  if (value) {
    updateDraft({ fontFamily: value })
  }
}

</script>

<style scoped>
.text-style-defaults-settings__intro {
  margin: 0 0 14px;
  color: var(--color-text-supporting);
  font-size: 13px;
  line-height: 1.6;
}
</style>
