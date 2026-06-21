import { ref, type Ref } from 'vue'

interface UseEditWorkspaceResizeActionsOptions {
  layoutMode: Ref<'horizontal' | 'vertical'>
  originalViewportRef: Readonly<Ref<HTMLElement | null>>
  editPanelRef: Readonly<Ref<HTMLElement | null>>
}

export function useEditWorkspaceResizeActions(options: UseEditWorkspaceResizeActionsOptions) {
  const isDraggingDivider = ref(false)
  const dividerStartPos = ref(0)
  const isResizingPanel = ref(false)
  const panelResizeStart = ref({ x: 0, y: 0, size: 0 })

  function startDividerDrag(event: MouseEvent): void {
    isDraggingDivider.value = true
    dividerStartPos.value = options.layoutMode.value === 'horizontal' ? event.clientX : event.clientY
    document.body.style.cursor = options.layoutMode.value === 'horizontal' ? 'col-resize' : 'row-resize'
    document.body.style.userSelect = 'none'

    document.addEventListener('mousemove', handleDividerDrag)
    document.addEventListener('mouseup', stopDividerDrag)
    event.preventDefault()
  }

  function handleDividerDrag(event: MouseEvent): void {
    if (!isDraggingDivider.value) return

    const container = options.originalViewportRef.value?.parentElement?.parentElement
    if (!container) return

    const containerRect = container.getBoundingClientRect()

    if (options.layoutMode.value === 'horizontal') {
      const mouseX = event.clientX - containerRect.left
      const totalWidth = containerRect.width
      const leftPercent = Math.max(20, Math.min(80, (mouseX / totalWidth) * 100))

      const originalPanel = container.querySelector('.original-panel') as HTMLElement
      const translatedPanel = container.querySelector('.translated-panel') as HTMLElement
      if (originalPanel && translatedPanel) {
        originalPanel.style.flex = `0 0 ${leftPercent}%`
        translatedPanel.style.flex = `0 0 ${100 - leftPercent}%`
      }
    } else {
      const mouseY = event.clientY - containerRect.top
      const totalHeight = containerRect.height
      const topPercent = Math.max(20, Math.min(80, (mouseY / totalHeight) * 100))

      const originalPanel = container.querySelector('.original-panel') as HTMLElement
      const translatedPanel = container.querySelector('.translated-panel') as HTMLElement
      if (originalPanel && translatedPanel) {
        originalPanel.style.flex = `0 0 ${topPercent}%`
        translatedPanel.style.flex = `0 0 ${100 - topPercent}%`
      }
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
    isResizingPanel.value = true
    const panel = options.editPanelRef.value
    if (!panel) return

    panelResizeStart.value = {
      x: event.clientX,
      y: event.clientY,
      size: options.layoutMode.value === 'horizontal' ? panel.offsetWidth : panel.offsetHeight,
    }

    document.body.style.cursor = options.layoutMode.value === 'horizontal' ? 'ew-resize' : 'ns-resize'
    document.body.style.userSelect = 'none'

    document.addEventListener('mousemove', handlePanelResize)
    document.addEventListener('mouseup', stopPanelResize)
    event.preventDefault()
  }

  function handlePanelResize(event: MouseEvent): void {
    const panel = options.editPanelRef.value
    if (!isResizingPanel.value || !panel) return

    if (options.layoutMode.value === 'horizontal') {
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

  return {
    startDividerDrag,
    handleDividerDrag,
    stopDividerDrag,
    startPanelResize,
    handlePanelResize,
    stopPanelResize,
  }
}
