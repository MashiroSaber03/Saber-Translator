import { computed, ref, watch } from 'vue'

import { getPageSummary } from '@/api/v2/content'
import { createChapterStyleApplyJob } from '@/api/v2/translation'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import { queuePageDocumentMutation } from '@/services/pageDocumentPersistence'
import type { BubbleState } from '@/types/bubble'
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

const STYLE_TO_BUBBLE_FIELD: Record<string, keyof BubbleState> = {
  fillColor: 'fillColor',
  fontFamily: 'fontFamily',
  fontSize: 'fontSize',
  lineSpacing: 'lineSpacing',
  strokeColor: 'strokeColor',
  strokeEnabled: 'strokeEnabled',
  strokeWidth: 'strokeWidth',
  textAlign: 'textAlign',
  textColor: 'textColor',
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

function rgbToHex(rgb: [number, number, number] | null | undefined): string | null {
  if (!rgb || rgb.length !== 3) return null
  return `#${rgb.map(value => (
    Math.max(0, Math.min(255, Math.round(value))).toString(16).padStart(2, '0')
  )).join('')}`
}

function withStyle(
  bubbles: BubbleState[],
  settingKey: string,
  value: unknown,
): BubbleState[] {
  if (settingKey === 'layoutDirection') {
    return bubbles.map(bubble => ({
      ...bubble,
      textDirection: value === 'auto'
        ? (
            bubble.autoTextDirection === 'horizontal'
              ? 'horizontal'
              : 'vertical'
          )
        : value as 'horizontal' | 'vertical',
    }))
  }
  const target = STYLE_TO_BUBBLE_FIELD[settingKey]
  if (!target) return bubbles.map(bubble => ({ ...bubble }))
  return bubbles.map(bubble => ({ ...bubble, [target]: value }))
}

export function useTextStyleSync() {
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()
  const bubbleStore = useBubbleStore()
  const taskCenterStore = useTaskCenterStore()
  const currentImage = computed(() => imageStore.currentImage)
  const isSyncingTextStyle = ref(false)

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
    bubbles: BubbleState[],
    propagateStyleFields: string[],
    defaultFontId?: string,
  ): Promise<void> {
    const image = currentImage.value
    if (!image || image.documentRevision === undefined) return
    const baseRevision = image.documentRevision
    try {
      await queuePageDocumentMutation(
        image.id,
        baseRevision,
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
    if (
      !Object.keys(pageStyleDefaultsPatch).some(field => RENDERED_STYLE_FIELDS.has(field))
      || !image.translatedAssetUrl
    ) return
    void refreshRenderedAsset(image.id, baseRevision + 1)
  }

  async function refreshRenderedAsset(pageId: string, minimumRevision: number): Promise<void> {
    for (let attempt = 0; attempt < 50; attempt += 1) {
      await new Promise(resolve => setTimeout(resolve, 200))
      let summary
      try {
        summary = await getPageSummary(pageId)
      } catch {
        continue
      }
      if (
        summary.renderStatus !== 'ready'
        || (summary.renderedRevision ?? 0) < minimumRevision
      ) continue
      const index = imageStore.images.findIndex(image => image.id === pageId)
      if (index >= 0) {
        imageStore.updateImageByIndex(index, {
          renderedRevision: summary.renderedRevision ?? undefined,
          thumbnailTranslatedUrl: summary.thumbnailTranslatedUrl,
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
    const source = image.bubbleStates ?? bubbleStore.bubbles
    const bubbles = withStyle(source, settingKey, newValue)
    imageStore.updateCurrentImage({ bubbleStates: bubbles, hasUnsavedChanges: true })
    bubbleStore.setBubbles(bubbles)
    await persistStyle(
      { [settingKey]: newValue },
      bubbles,
      [settingKey],
      settingKey === 'fontFamily' ? String(newValue) : undefined,
    )
  }

  async function handleAutoFontSizeChanged(autoFontSize: boolean) {
    const image = currentImage.value
    if (!image) return
    const fixedFontSize = settingsStore.settings.textStyle.fontSize
    const source = image.bubbleStates ?? bubbleStore.bubbles
    const bubbles = autoFontSize
      ? source.map(bubble => ({ ...bubble }))
      : withStyle(source, 'fontSize', fixedFontSize)
    imageStore.updateCurrentImage({
      autoFontSize,
      bubbleStates: bubbles,
      hasUnsavedChanges: true,
    })
    bubbleStore.setBubbles(bubbles)
    await persistStyle(
      { autoFontSize, fontSize: fixedFontSize },
      bubbles,
      ['autoFontSize', 'fontSize'],
    )
  }

  async function handleAutoTextColorChanged(useAutoTextColor: boolean) {
    const image = currentImage.value
    if (!image) return
    const source = image.bubbleStates ?? bubbleStore.bubbles
    const bubbles = useAutoTextColor
      ? source.map(bubble => ({
          ...bubble,
          fillColor: rgbToHex(bubble.autoBgColor)
            ?? bubble.fillColor
            ?? settingsStore.settings.textStyle.fillColor,
          textColor: rgbToHex(bubble.autoFgColor)
            ?? bubble.textColor
            ?? settingsStore.settings.textStyle.textColor,
        }))
      : source.map(bubble => ({ ...bubble }))
    imageStore.updateCurrentImage({
      bubbleStates: bubbles,
      hasUnsavedChanges: true,
      useAutoTextColor,
    })
    bubbleStore.setBubbles(bubbles)
    await persistStyle(
      { useAutoTextColor },
      bubbles,
      ['useAutoTextColor'],
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
      await createChapterStyleApplyJob(image.chapterId, {
        selectedFields,
        sourceDocumentRevision: image.documentRevision,
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
    isSyncingTextStyle,
    syncImageToSidebar,
    syncSidebarToImage,
  }
}
