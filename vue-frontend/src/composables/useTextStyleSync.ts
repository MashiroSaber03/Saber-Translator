import { computed, getCurrentInstance, onUnmounted, ref, watch } from 'vue'

import { getPageSummary } from '@/api/v2/content'
import { createChapterStyleApplyJob } from '@/api/v2/translation'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import {
  flushPageDocument,
  queuePageDocumentMutation,
} from '@/services/pageDocumentPersistence'
import { showToast } from '@/utils/toast'

export interface ApplySettingsOptions {
  fontSize: boolean
  fontFamily: boolean
  layoutDirection: boolean
  textColor: boolean
  fillColor: boolean
  strokeEnabled: boolean
  strokeColor: boolean
  strokeWidth: boolean
  lineSpacing: boolean
  textAlign: boolean
}

const RENDERED_STYLE_FIELDS = new Set([
  'fontSize',
  'fontFamily',
  'layoutDirection',
  'textColor',
  'strokeEnabled',
  'strokeColor',
  'strokeWidth',
  'lineSpacing',
  'textAlign',
])

export function useTextStyleSync() {
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()
  const bubbleStore = useBubbleStore()
  const taskCenterStore = useTaskCenterStore()
  const currentImage = computed(() => imageStore.currentImage)
  const isSyncingTextStyle = ref(false)
  let renderedAssetRefreshToken = 0

  if (getCurrentInstance()) {
    onUnmounted(() => {
      renderedAssetRefreshToken += 1
    })
  }

  function syncImageToSidebar(image: typeof imageStore.currentImage) {
    if (!image) return
    const current = settingsStore.settings.textStyle
    settingsStore.updateTextStyle({
      autoFontSize: image.autoFontSize ?? current.autoFontSize,
      fillColor: image.fillColor ?? current.fillColor,
      fontFamily: image.fontFamily ?? current.fontFamily,
      fontSize: image.fontSize ?? current.fontSize,
      inpaintMethod: image.inpaintMethod ?? current.inpaintMethod,
      layoutDirection: image.layoutDirection ?? current.layoutDirection,
      lineSpacing: image.lineSpacing ?? current.lineSpacing,
      strokeColor: image.strokeColor ?? current.strokeColor,
      strokeEnabled: image.strokeEnabled ?? current.strokeEnabled,
      strokeWidth: image.strokeWidth ?? current.strokeWidth,
      textAlign: image.textAlign ?? current.textAlign,
      textColor: image.textColor ?? current.textColor,
      useAutoTextColor: image.useAutoTextColor ?? current.useAutoTextColor,
    })
  }

  function syncSidebarToImage(style: typeof settingsStore.settings.textStyle) {
    if (!imageStore.currentImage) return
    imageStore.updateCurrentImage({
      autoFontSize: style.autoFontSize,
      fillColor: style.fillColor,
      fontFamily: style.fontFamily,
      fontSize: style.fontSize,
      inpaintMethod: style.inpaintMethod,
      layoutDirection: style.layoutDirection,
      lineSpacing: style.lineSpacing,
      strokeColor: style.strokeColor,
      strokeEnabled: style.strokeEnabled,
      strokeWidth: style.strokeWidth,
      textAlign: style.textAlign,
      textColor: style.textColor,
      useAutoTextColor: style.useAutoTextColor,
    })
  }

  watch(
    () => imageStore.currentImage,
    image => {
      if (!image || isSyncingTextStyle.value) return
      isSyncingTextStyle.value = true
      try {
        syncImageToSidebar(image)
      } finally {
        isSyncingTextStyle.value = false
      }
    },
  )

  watch(
    () => settingsStore.settings.textStyle,
    style => {
      if (!imageStore.currentImage || isSyncingTextStyle.value) return
      isSyncingTextStyle.value = true
      try {
        syncSidebarToImage(style)
      } finally {
        isSyncingTextStyle.value = false
      }
    },
    { deep: true },
  )

  async function persistStyle(
    pageStyleDefaultsPatch: Record<string, unknown>,
    propagateStyleFields: string[],
    defaultFontId?: string,
    expectsRender = false,
  ): Promise<void> {
    const image = currentImage.value
    if (!image || image.documentRevision === undefined) return
    const bubbles = image.bubbleStates ?? bubbleStore.bubbles
    try {
      await queuePageDocumentMutation(
        image.id,
        image.documentRevision,
        bubbles,
        {
          ...(defaultFontId ? { defaultFontId } : {}),
          pageStyleDefaultsPatch,
          propagateStyleFields,
        },
      )
    } catch (error) {
      showToast(
        `文字样式写入后端失败：${error instanceof Error ? error.message : '未知错误'}`,
        'error',
      )
      return
    }
    const committed = imageStore.images.find(candidate => candidate.id === image.id)
    if (!expectsRender || !image.translatedAssetUrl || !committed?.documentRevision) return
    void refreshRenderedAsset(image.id, committed.documentRevision)
  }

  async function refreshRenderedAsset(pageId: string, minimumRevision: number): Promise<void> {
    const token = ++renderedAssetRefreshToken
    for (let attempt = 0; attempt < 50; attempt += 1) {
      await new Promise(resolve => setTimeout(resolve, 200))
      if (token !== renderedAssetRefreshToken) return
      let summary
      try {
        summary = await getPageSummary(pageId)
      } catch {
        continue
      }
      if (token !== renderedAssetRefreshToken) return
      if (
        summary.renderStatus !== 'ready'
        || (summary.renderedRevision ?? 0) < minimumRevision
      ) continue
      const index = imageStore.images.findIndex(image => image.id === pageId)
      if (index >= 0) {
        imageStore.updateImageByIndex(index, {
          renderedRevision: summary.renderedRevision ?? undefined,
          translatedAssetUrl: summary.translatedUrl,
          translationStatus: 'completed',
        })
      }
      return
    }
  }

  async function handleTextStyleChanged(settingKey: string, newValue: unknown) {
    const image = currentImage.value
    if (!image) return
    imageStore.updateCurrentImage({
      [settingKey]: newValue,
      hasUnsavedChanges: true,
    })
    await persistStyle(
      settingKey === 'fontFamily' ? {} : { [settingKey]: newValue },
      settingKey === 'inpaintMethod' ? [] : [settingKey],
      settingKey === 'fontFamily' ? String(newValue) : undefined,
      RENDERED_STYLE_FIELDS.has(settingKey),
    )
  }

  async function handleAutoFontSizeChanged(autoFontSize: boolean) {
    const image = currentImage.value
    if (!image) return
    const fixedFontSize = settingsStore.settings.textStyle.fontSize
    imageStore.updateCurrentImage({
      autoFontSize,
      hasUnsavedChanges: true,
    })
    await persistStyle(
      { autoFontSize, fontSize: fixedFontSize },
      ['fontSize'],
      undefined,
      true,
    )
  }

  async function handleAutoTextColorChanged(useAutoTextColor: boolean) {
    const image = currentImage.value
    if (!image) return
    const textStyle = settingsStore.settings.textStyle
    imageStore.updateCurrentImage({
      hasUnsavedChanges: true,
      useAutoTextColor,
    })
    await persistStyle(
      {
        useAutoTextColor,
        ...(
          useAutoTextColor
            ? {}
            : {
                textColor: textStyle.textColor,
                fillColor: textStyle.fillColor,
              }
        ),
      },
      ['textColor', 'fillColor'],
      undefined,
      true,
    )
  }

  async function handleApplyToAll(options: ApplySettingsOptions) {
    const selectedFields = Object.entries(options)
      .filter(([, selected]) => selected)
      .map(([field]) => field)
    if (selectedFields.length === 0) {
      showToast('请至少选择一个要应用的设置项', 'warning')
      return
    }
    const image = currentImage.value
    if (!image || !image.chapterId || image.documentRevision === undefined) {
      showToast('当前页尚未写入后端章节', 'warning')
      return
    }
    try {
      await flushPageDocument(image.id)
      const committed = imageStore.images.find(candidate => candidate.id === image.id)
      if (committed?.documentRevision === undefined) {
        throw new Error('当前页文档版本不可用')
      }
      await createChapterStyleApplyJob(image.chapterId, {
        selectedFields,
        sourceDocumentRevision: committed.documentRevision,
        sourcePageId: image.id,
      })
      await taskCenterStore.refresh()
      showToast('样式应用任务已加入后端任务中心，可安全关闭页面', 'success')
    } catch (error) {
      showToast(
        `创建样式应用任务失败：${error instanceof Error ? error.message : '未知错误'}`,
        'error',
      )
    }
  }

  return {
    handleApplyToAll,
    handleAutoFontSizeChanged,
    handleAutoTextColorChanged,
    handleTextStyleChanged,
  }
}
