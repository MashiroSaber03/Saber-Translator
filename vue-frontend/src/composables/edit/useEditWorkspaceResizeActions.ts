import { onUnmounted, ref, type Ref } from 'vue'

interface UseEditWorkspaceResizeActionsOptions {
  layoutMode: Ref<'horizontal' | 'vertical'>
  originalPanelRef: Readonly<Ref<HTMLElement | null>>
  translatedPanelRef: Readonly<Ref<HTMLElement | null>>
  editPanelRef: Readonly<Ref<HTMLElement | null>>
}

export function useEditWorkspaceResizeActions(options: UseEditWorkspaceResizeActionsOptions) {
  const isDraggingDivider = ref(false)
  const isResizingPanel = ref(false)
  const panelResizeStart = ref({ x: 0, y: 0, size: 0 })
  let dividerIsVertical = false
  let panelResizeIsVertical = false

  function usesVerticalFlow(element: HTMLElement | null): boolean {
    const direction = element ? window.getComputedStyle(element).flexDirection : ''
    if (direction === 'column' || direction === 'column-reverse') return true
    if (direction === 'row' || direction === 'row-reverse') return false
    return options.layoutMode.value === 'vertical'
  }

  function startDividerDrag(event: MouseEvent): void {
    dividerIsVertical = usesVerticalFlow(options.originalPanelRef.value?.parentElement ?? null)
    isDraggingDivider.value = true
    document.body.style.cursor = dividerIsVertical ? 'row-resize' : 'col-resize'
    document.body.style.userSelect = 'none'

    document.addEventListener('mousemove', handleDividerDrag)
    document.addEventListener('mouseup', stopDividerDrag)
    event.preventDefault()
  }

  function handleDividerDrag(event: MouseEvent): void {
    if (!isDraggingDivider.value) return

    const originalPanel = options.originalPanelRef.value
    const translatedPanel = options.translatedPanelRef.value
    const container = originalPanel?.parentElement
    if (!container || !originalPanel || !translatedPanel) return

    const containerRect = container.getBoundingClientRect()

    if (!dividerIsVertical) {
      const mouseX = event.clientX - containerRect.left
      const totalWidth = containerRect.width
      if (totalWidth <= 0) return
      const leftPercent = Math.max(20, Math.min(80, (mouseX / totalWidth) * 100))

      originalPanel.style.flex = `0 0 ${leftPercent}%`
      translatedPanel.style.flex = `0 0 ${100 - leftPercent}%`
    } else {
      const mouseY = event.clientY - containerRect.top
      const totalHeight = containerRect.height
      if (totalHeight <= 0) return
      const topPercent = Math.max(20, Math.min(80, (mouseY / totalHeight) * 100))

      originalPanel.style.flex = `0 0 ${topPercent}%`
      translatedPanel.style.flex = `0 0 ${100 - topPercent}%`
    }
  }

  function stopDividerDrag(): void {
    isDraggingDivider.value = false
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
    document.removeEventListener('mousemove', handleDividerDrag)
    document.removeEventListener('mouseup', stopDividerDrag)
  }

  function startPanelResize(event: MouseEvent): void {
    const panel = options.editPanelRef.value
    if (!panel) return

    panelResizeIsVertical = usesVerticalFlow(panel.parentElement)
    isResizingPanel.value = true
    panelResizeStart.value = {
      x: event.clientX,
      y: event.clientY,
      size: panelResizeIsVertical ? panel.offsetHeight : panel.offsetWidth,
    }

    document.body.style.cursor = panelResizeIsVertical ? 'ns-resize' : 'ew-resize'
    document.body.style.userSelect = 'none'

    document.addEventListener('mousemove', handlePanelResize)
    document.addEventListener('mouseup', stopPanelResize)
    event.preventDefault()
  }

  function handlePanelResize(event: MouseEvent): void {
    const panel = options.editPanelRef.value
    if (!isResizingPanel.value || !panel) return

    if (!panelResizeIsVertical) {
      const deltaX = panelResizeStart.value.x - event.clientX
      const newWidth = Math.max(300, Math.min(window.innerWidth * 0.6, panelResizeStart.value.size + deltaX))
      panel.style.flex = `0 0 ${newWidth}px`
      panel.style.minWidth = `${newWidth}px`
    } else {
      const deltaY = panelResizeStart.value.y - event.clientY
      const newHeight = Math.max(200, Math.min(window.innerHeight * 0.5, panelResizeStart.value.size + deltaY))
      panel.style.flex = `0 0 ${newHeight}px`
      panel.style.height = `${newHeight}px`
    }
  }

  function stopPanelResize(): void {
    isResizingPanel.value = false
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
    document.removeEventListener('mousemove', handlePanelResize)
    document.removeEventListener('mouseup', stopPanelResize)
  }

  onUnmounted(() => {
    if (isDraggingDivider.value) {
      stopDividerDrag()
    }
    if (isResizingPanel.value) {
      stopPanelResize()
    }
  })

  return {
    startDividerDrag,
    handleDividerDrag,
    stopDividerDrag,
    startPanelResize,
    handlePanelResize,
    stopPanelResize,
  }
}
