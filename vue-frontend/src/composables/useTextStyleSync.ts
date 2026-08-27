import { computed, getCurrentInstance, onUnmounted, ref, watch } from 'vue'

import { getPageDocument, getPageRenderStatus } from '@/api/v2/content'
import { createChapterStyleApplyJob } from '@/api/v2/translation'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import {
  flushPageDocument,
  queuePageDocumentMutation,
  registerPageDocument,
} from '@/services/pageDocumentPersistence'
import { showToast } from '@/utils/toast'
import { parseCompleteTextStyleSettings } from '@/defaults/textStyleDefaults'
import type { TextStyleMutationField, TextStyleSettings } from '@/types/settings'

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
  inlineAlign: boolean
  blockAlign: boolean
}

const RENDERED_STYLE_FIELDS = new Set<TextStyleMutationField>([
  'fontSize',
  'fontFamily',
  'layoutDirection',
  'textColor',
  'strokeEnabled',
  'strokeColor',
  'strokeWidth',
  'lineSpacing',
  'inlineAlign',
  'blockAlign',
])
const RENDER_STATUS_POLL_MS = 500
const RENDER_STATUS_TIMEOUT_MS = 30_000
const RENDER_STATUS_RETRY_LIMIT = 5

export function useTextStyleSync() {
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()
  const bubbleStore = useBubbleStore()
  const taskCenterStore = useTaskCenterStore()
  const currentImage = computed(() => imageStore.currentImage)
  const isSyncingTextStyle = ref(false)
  const isApplyingToAll = ref(false)
  const renderStatusControllers = new Map<string, AbortController>()

  if (getCurrentInstance()) {
    onUnmounted(() => {
      for (const controller of renderStatusControllers.values()) {
        controller.abort()
      }
      renderStatusControllers.clear()
    })
  }

  function syncImageToSidebar(image: typeof imageStore.currentImage) {
    if (!image) return
    settingsStore.updateTextStyle({
      autoFontSize: image.autoFontSize,
      fillColor: image.fillColor,
      fontFamily: image.fontFamily,
      fontSize: image.fontSize,
      inpaintMethod: image.inpaintMethod,
      layoutDirection: image.layoutDirection,
      lineSpacing: image.lineSpacing,
      strokeColor: image.strokeColor,
      strokeEnabled: image.strokeEnabled,
      strokeWidth: image.strokeWidth,
      inlineAlign: image.inlineAlign,
      blockAlign: image.blockAlign,
      textColor: image.textColor,
      useAutoTextColor: image.useAutoTextColor,
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
      inlineAlign: style.inlineAlign,
      blockAlign: style.blockAlign,
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
    { immediate: true },
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
      const failure = error instanceof Error ? error.message : '未知错误'
      try {
        await reloadAuthoritativePage(image.id, image.chapterId)
        showToast(`文字样式写入后端失败，已恢复后端版本：${failure}`, 'error')
      } catch (reloadError) {
        showToast(
          `文字样式写入后端失败：${failure}；重新加载失败：${
            reloadError instanceof Error ? reloadError.message : '未知错误'
          }`,
          'error',
        )
      }
      return
    }
    const committed = imageStore.images.find(candidate => candidate.id === image.id)
    if (!expectsRender || !image.translatedAssetUrl || !committed?.documentRevision) return
    imageStore.updateImageByIndex(
      imageStore.images.findIndex(candidate => candidate.id === image.id),
      { translationStatus: 'processing' },
    )
    void refreshRenderedAsset(image.id, committed.documentRevision)
  }

  async function reloadAuthoritativePage(
    pageId: string,
    expectedChapterId: string | undefined,
  ): Promise<void> {
    const document = await getPageDocument(pageId)
    if (
      document.pageId !== pageId
      || !expectedChapterId
      || document.chapterId !== expectedChapterId
    ) {
      throw new Error(`页面 ${pageId} 的后端文档身份不匹配`)
    }
    const bubbles = registerPageDocument(document)
    if (imageStore.currentImage?.id !== pageId) return
    const pageTextStyle = parseCompleteTextStyleSettings({
      ...document.pageStyleDefaults,
      ...(document.defaultFontId ? { fontFamily: document.defaultFontId } : {}),
    })
    isSyncingTextStyle.value = true
    try {
      imageStore.updateCurrentImage({
        ...pageTextStyle,
        bubbleStates: bubbles,
        documentRevision: document.documentRevision,
        hasUnsavedChanges: false,
      })
      settingsStore.updateTextStyle(pageTextStyle)
      bubbleStore.setBubbles(bubbles, true)
      bubbleStore.saveAsInitial()
    } finally {
      isSyncingTextStyle.value = false
    }
  }

  async function refreshRenderedAsset(pageId: string, minimumRevision: number): Promise<void> {
    renderStatusControllers.get(pageId)?.abort()
    const controller = new AbortController()
    renderStatusControllers.set(pageId, controller)
    const deadline = Date.now() + RENDER_STATUS_TIMEOUT_MS
    let consecutiveReadFailures = 0
    try {
      while (!controller.signal.aborted && Date.now() < deadline) {
        let status
        try {
          status = await getPageRenderStatus(pageId, controller.signal)
          if (controller.signal.aborted) return
        } catch (error) {
          if (controller.signal.aborted) return
          consecutiveReadFailures += 1
          if (consecutiveReadFailures >= RENDER_STATUS_RETRY_LIMIT) {
            showToast(
              `文字样式已保存，但无法读取后端渲染状态：${
                error instanceof Error ? error.message : '未知错误'
              }`,
              'error',
            )
            return
          }
          await new Promise(resolve => setTimeout(resolve, RENDER_STATUS_POLL_MS))
          continue
        }
        consecutiveReadFailures = 0
        if (status.pageId !== pageId) {
          showToast(`页面 ${pageId} 的渲染状态身份不匹配`, 'error')
          return
        }
        if (
          status.renderStatus === 'render_failed'
          || status.renderStatus === 'repair_failed'
        ) {
          const index = imageStore.images.findIndex(image => image.id === pageId)
          if (index >= 0) imageStore.setTranslationStatus(index, 'failed')
          showToast('文字样式已保存，但后端重渲染失败', 'error')
          return
        }
        if (
          status.renderStatus !== 'ready'
          || (status.renderedRevision ?? 0) < minimumRevision
        ) {
          await new Promise(resolve => setTimeout(resolve, RENDER_STATUS_POLL_MS))
          continue
        }
        const index = imageStore.images.findIndex(image => image.id === pageId)
        if (index >= 0) {
          imageStore.updateImageByIndex(index, {
            renderedRevision: status.renderedRevision ?? undefined,
            translatedAssetUrl: status.translatedUrl,
            translationStatus: 'completed',
          })
        }
        return
      }
      if (!controller.signal.aborted) {
        showToast('文字样式已保存，后端仍在生成最新预览', 'info')
      }
    } finally {
      if (renderStatusControllers.get(pageId) === controller) {
        renderStatusControllers.delete(pageId)
      }
    }
  }

  async function handleTextStyleChanged<Field extends TextStyleMutationField>(
    settingKey: Field,
    newValue: TextStyleSettings[Field],
  ) {
    const image = currentImage.value
    if (!image) return
    imageStore.updateCurrentImage({
      [settingKey]: newValue,
      hasUnsavedChanges: true,
    })
    const defaultFontId = settingKey === 'fontFamily' && typeof newValue === 'string'
      ? newValue
      : undefined
    await persistStyle(
      settingKey === 'fontFamily' ? {} : { [settingKey]: newValue },
      settingKey === 'inpaintMethod' ? [] : [settingKey],
      defaultFontId,
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
    if (isApplyingToAll.value) return
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
    isApplyingToAll.value = true
    try {
      await flushPageDocument(image.id)
      const committed = imageStore.images.find(candidate => candidate.id === image.id)
      if (committed?.documentRevision === undefined) {
        throw new Error('当前页文档版本不可用')
      }
      const accepted = await createChapterStyleApplyJob(image.chapterId, {
        selectedFields,
        sourceDocumentRevision: committed.documentRevision,
        sourcePageId: image.id,
      })
      for (const jobId of accepted.jobIds) taskCenterStore.trackJob(jobId)
      showToast('样式应用任务已加入后端任务中心，可安全关闭页面', 'success')
    } catch (error) {
      showToast(
        `创建样式应用任务失败：${error instanceof Error ? error.message : '未知错误'}`,
        'error',
      )
    } finally {
      isApplyingToAll.value = false
    }
  }

  return {
    handleApplyToAll,
    handleAutoFontSizeChanged,
    handleAutoTextColorChanged,
    handleTextStyleChanged,
  }
}
