import { ref } from 'vue'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'

export type ViewMode = 'dual' | 'original' | 'translated'
export type LayoutMode = 'horizontal' | 'vertical'

export function useEditMode() {
  const bubbleStore = useBubbleStore()
  const imageStore = useImageStore()

  const isActive = ref(false)

  function exitEditModeWithoutRender(): void {
    if (!isActive.value) return

    const currentImage = imageStore.currentImage

    if (bubbleStore.bubbleCount > 0) {
      imageStore.updateCurrentBubbleStates([...bubbleStore.bubbles])
    } else if (currentImage && Array.isArray(currentImage.bubbleStates) && currentImage.bubbleStates.length > 0) {
      // Empty state is meaningful here: the user intentionally removed every bubble.
      imageStore.updateCurrentBubbleStates([])
    }

    isActive.value = false
    bubbleStore.clearSelection()
  }

  return {
    isActive,
    exitEditModeWithoutRender
  }
}
