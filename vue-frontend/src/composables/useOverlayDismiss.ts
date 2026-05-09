import { onMounted, onUnmounted, ref, toValue, watch, type MaybeRefOrGetter } from 'vue'

interface UseOverlayDismissOptions {
  enabled?: MaybeRefOrGetter<boolean>
}

/**
 * 仅在按下和松开都发生在遮罩层上时触发关闭，避免从弹窗内容拖出后误关闭。
 */
export function useOverlayDismiss(
  onDismiss: () => void,
  options: UseOverlayDismissOptions = {},
) {
  const overlayRef = ref<HTMLElement | null>(null)
  let overlayPressStarted = false
  let isMounted = false
  let isListening = false

  const isEnabled = () => toValue(options.enabled) ?? true

  const resetOverlayDismissState = () => {
    overlayPressStarted = false
  }

  const handleOverlayMouseDown = (event: MouseEvent) => {
    if (!isEnabled() || event.button !== 0) {
      resetOverlayDismissState()
      return
    }

    overlayPressStarted = true
  }

  const handleDocumentMouseUp = (event: MouseEvent) => {
    if (!overlayPressStarted) return

    const releasedOnOverlay = event.button === 0 && event.target === overlayRef.value
    resetOverlayDismissState()

    if (releasedOnOverlay && isEnabled()) {
      onDismiss()
    }
  }

  const startListening = () => {
    if (!isMounted || isListening) return

    document.addEventListener('mouseup', handleDocumentMouseUp)
    isListening = true
  }

  const stopListening = () => {
    if (!isListening) return

    document.removeEventListener('mouseup', handleDocumentMouseUp)
    isListening = false
  }

  watch(isEnabled, (enabled) => {
    if (!enabled) {
      resetOverlayDismissState()
      stopListening()
      return
    }

    if (isMounted) {
      startListening()
    }
  })

  onMounted(() => {
    isMounted = true
    if (isEnabled()) {
      startListening()
    }
  })

  onUnmounted(() => {
    isMounted = false
    resetOverlayDismissState()
    stopListening()
  })

  return {
    overlayRef,
    handleOverlayMouseDown,
    resetOverlayDismissState,
  }
}
