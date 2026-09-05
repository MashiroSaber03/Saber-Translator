import { computed, onUnmounted, ref, watch, type Ref } from 'vue'
import type { BubbleColorField, BubbleState } from '@/types/bubble'
import { sampleImageColor } from '@/utils/imageColorSampling'

interface Options {
  pageId: Readonly<Ref<string | undefined>>
  bubble: Readonly<Ref<BubbleState | null | undefined>>
  bubbleIndex: Readonly<Ref<number>>
  disabled: Readonly<Ref<boolean>>
  onPick: (field: BubbleColorField, color: string) => void
  onError: (message: string) => void
}

export function useImageColorPicker(options: Options) {
  const colorPickField = ref<BubbleColorField | null>(null)
  const isPickingColor = computed(() => colorPickField.value !== null)

  function cancelColorPick(): void {
    colorPickField.value = null
  }

  function startColorPick(field: BubbleColorField): boolean {
    const { pageId, bubble, bubbleIndex, disabled } = options
    if (disabled.value || !pageId.value || !bubble.value || bubbleIndex.value < 0) return false
    colorPickField.value = field
    return true
  }

  function pickImageColor(image: HTMLImageElement | null, point: { clientX: number; clientY: number }): void {
    const field = colorPickField.value
    if (!field || !image) return
    let color: string | null
    try {
      color = sampleImageColor(image, point)
    } catch {
      cancelColorPick()
      options.onError('读取图片颜色失败，请确认图片已加载后重试')
      return
    }
    if (!color) return
    cancelColorPick()
    options.onPick(field, color)
  }

  // Sampling is synchronous; invalidate immediately when the editing target changes.
  watch([options.pageId, options.bubble, options.bubbleIndex, options.disabled], cancelColorPick, { flush: 'sync' })
  onUnmounted(cancelColorPick)

  return { colorPickField, isPickingColor, startColorPick, cancelColorPick, pickImageColor }
}
