import { computed, getCurrentInstance, onUnmounted, ref, watch } from 'vue'

import { getPageDocument } from '@/api/v2/content'
import { waitForPageRender } from '@/composables/useEditRender'
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
import { normalizeImageTextStyleFields } from '@/defaults/textStyleDefaults'
import { restoreFontRuns } from '@/components/edit/richTextDocument'
import { deepClone } from '@/utils/deepClone'
import type { BubbleState } from '@/types/bubble'
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

export function useTextStyleSync() {
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()
  const bubbleStore = useBubbleStore()
  const taskCenterStore = useTaskCenterStore()
  const currentImage = computed(() => imageStore.currentImage)
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

  // The panel projects page defaults. User events below are the only writes;
  // another two-way settings watcher would silently copy unrelated preferences
  // back over an accepted page document.
  watch(
    () => imageStore.currentImage ? normalizeImageTextStyleFields(imageStore.currentImage) : null,
    style => {
      if (style) settingsStore.updateTextStyle(style)
    },
    { immediate: true },
  )

  async function persistStyle(
    pageStyleDefaultsPatch: Record<string, unknown>,
    propagateStyleFields: string[],
    defaultFontId?: string,
    expectsRender = false,
    bubblesOverride?: BubbleState[],
  ): Promise<boolean> {
    const image = currentImage.value
    if (!image || image.documentRevision === undefined) return false
    const bubbles = bubblesOverride ?? image.bubbleStates ?? bubbleStore.bubbles
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
        const retained = imageStore.images.find(candidate => candidate.id === image.id)?.hasUnsavedChanges
        showToast(retained
          ? `文字样式保存失败，修改已保留，请重试：${failure}`
          : `文字样式保存失败，已重新加载后端版本：${failure}`, 'error')
      } catch (reloadError) {
        showToast(
          `文字样式写入后端失败：${failure}；重新加载失败：${
            reloadError instanceof Error ? reloadError.message : '未知错误'
          }`,
          'error',
        )
      }
      return false
    }
    const committed = imageStore.images.find(candidate => candidate.id === image.id)
    if (!expectsRender || !image.translatedAssetUrl || !committed?.documentRevision) return true
    imageStore.updateImageByIndex(
      imageStore.images.findIndex(candidate => candidate.id === image.id),
      { translationStatus: 'processing' },
    )
    void refreshRenderedAsset(image.id, committed.documentRevision)
    return true
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
    registerPageDocument(document)
  }

  async function refreshRenderedAsset(pageId: string, minimumRevision: number): Promise<void> {
    renderStatusControllers.get(pageId)?.abort()
    const controller = new AbortController()
    renderStatusControllers.set(pageId, controller)
    try {
      await waitForPageRender(pageId, minimumRevision, controller.signal)
    } catch (error) {
      if (!controller.signal.aborted) showToast(`文字样式已保存，${error instanceof Error ? error.message : '预览加载失败'}`, 'error')
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
    if (!image) return false
    imageStore.updateCurrentImage({
      [settingKey]: newValue,
      hasUnsavedChanges: true,
    })
    const defaultFontId = settingKey === 'fontFamily' && typeof newValue === 'string'
      ? newValue
      : undefined
    return persistStyle(
      settingKey === 'fontFamily' ? {} : { [settingKey]: newValue },
      settingKey === 'inpaintMethod' ? [] : [settingKey],
      defaultFontId,
      RENDERED_STYLE_FIELDS.has(settingKey),
    )
  }

  function handleMatchedFont(fontId: string): (() => void | Promise<void>) | null {
    const image = currentImage.value
    const previous = settingsStore.settings.textStyle.fontFamily
    if (!image) {
      settingsStore.updateTextStyle({ fontFamily: fontId })
      return () => {
        if (!currentImage.value && settingsStore.settings.textStyle.fontFamily === fontId) {
          settingsStore.updateTextStyle({ fontFamily: previous })
        }
      }
    }
    if (image.documentRevision === undefined) return null

    const pageId = image.id
    const previousDefault = image.fontFamily || previous
    const originals = new Map(bubbleStore.bubbles.flatMap(bubble => bubble.backendBubbleId ? [[
      bubble.backendBubbleId, {
        fontId: bubble.fontFamily,
        text: bubble.translatedText,
        runs: deepClone(bubble.richText?.runs),
      },
    ] as const] : []))
    const saved = handleTextStyleChanged('fontFamily', fontId)
    return async () => {
      if (!(await saved)) return
      const current = currentImage.value
      if (current?.id !== pageId || current.fontFamily !== fontId) return
      const bubbles = bubbleStore.bubbles.map(bubble => {
        const original = originals.get(bubble.backendBubbleId ?? '')
        if (!original || bubble.fontFamily !== fontId || bubble.translatedText !== original.text) return bubble
        return {
          ...bubble,
          fontFamily: original.fontId,
          richText: bubble.richText && original.runs ? {
            ...bubble.richText, runs: restoreFontRuns(bubble.richText.runs, original.runs),
          } : bubble.richText,
        }
      })
      // Restore individual faces and the page default in the same document
      // mutation. Propagating the default here would overwrite the faces again.
      await persistStyle({}, [], previousDefault, true, bubbles)
    }
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
    handleMatchedFont,
    handleTextStyleChanged,
  }
}
