import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  BubbleState,
  BubbleCoords,
  BubbleStateOverrides,
  BubbleStateUpdates
} from '@/types/bubble'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'

import {
  createBubbleState as createBubbleStateFromFactory,
  cloneBubbleStates,
  getTextlinesPerBubbleFromStates,
  isValidBubbleState,
  detectTextDirection
} from '@/utils/bubbleFactory'

export { cloneBubbleStates, isValidBubbleState }

export function createBubbleState(overrides?: BubbleStateOverrides | BubbleCoords): BubbleState {
  if (Array.isArray(overrides)) {
    return createBubbleStateFromFactory({ coords: overrides })
  }
  return createBubbleStateFromFactory(overrides)
}


export const useBubbleStore = defineStore('bubble', () => {
  const bubbles = ref<BubbleState[]>([])
  const selectedIndex = ref<number>(-1)
  const selectedIndices = ref<number[]>([])
  const initialStates = ref<BubbleState[]>([])

  const selectedBubble = computed<BubbleState | null>(() => {
    if (selectedIndex.value >= 0 && selectedIndex.value < bubbles.value.length) {
      return bubbles.value[selectedIndex.value] ?? null
    }
    return null
  })

  const bubbleCount = computed<number>(() => bubbles.value.length)
  const hasBubbles = computed<boolean>(() => bubbles.value.length > 0)
  const hasSelection = computed<boolean>(() => selectedIndex.value >= 0)
  const isMultiSelect = computed<boolean>(() => selectedIndices.value.length > 1)
  const selectedBubbles = computed<BubbleState[]>(() => {
    return selectedIndices.value
      .filter((i) => i >= 0 && i < bubbles.value.length)
      .map((i) => bubbles.value[i])
      .filter((b): b is BubbleState => b !== undefined)
  })

  function syncToCurrentImage(): void {
    const imageStore = useImageStore()
    const currentImage = imageStore.currentImage
    if (currentImage) {
      const clonedBubbles = cloneBubbleStates(bubbles.value)
      currentImage.bubbleStates = clonedBubbles
      currentImage.bubbleCoords = clonedBubbles.map((bubble) => bubble.coords)
      currentImage.bubbleAngles = clonedBubbles.map((bubble) => bubble.rotationAngle || 0)
      currentImage.originalTexts = clonedBubbles.map((bubble) => bubble.originalText || '')
      currentImage.bubbleTexts = clonedBubbles.map((bubble) => bubble.translatedText || '')
      currentImage.textboxTexts = clonedBubbles.map((bubble) => bubble.textboxText || '')
      currentImage.textlinesPerBubble = getTextlinesPerBubbleFromStates(clonedBubbles)
      currentImage.ocrResults = clonedBubbles.map((bubble) => bubble.ocrResult || {
        text: bubble.originalText || '',
        confidence: null,
        confidenceSupported: false,
        engine: '',
        primaryEngine: '',
        fallbackUsed: false
      })
      currentImage.hasUnsavedChanges = true
    }
  }

  function setBubbles(newBubbles: BubbleState[], skipSync: boolean = false): void {
    bubbles.value = newBubbles
    initialStates.value = cloneBubbleStates(newBubbles)
    clearSelection()
    if (!skipSync) {
      syncToCurrentImage()
    }
  }

  function addBubble(coords: BubbleCoords, overrides?: Partial<BubbleState>): BubbleState {
    const autoDirection = detectTextDirection(coords)

    const settingsStore = useSettingsStore()
    const textStyle = settingsStore.settings.textStyle

    const layoutDirection = textStyle.layoutDirection
    const bubbleTextDirection =
      (layoutDirection === 'vertical' || layoutDirection === 'horizontal')
        ? layoutDirection
        : (autoDirection === 'vertical' || autoDirection === 'horizontal')
          ? autoDirection
          : 'vertical' as const

    const newBubble = createBubbleState({
      coords,
      translatedText: '',
      autoTextDirection: autoDirection,
      fontSize: textStyle.fontSize,
      fontFamily: textStyle.fontFamily,
      textDirection: bubbleTextDirection,
      textColor: textStyle.textColor,
      fillColor: textStyle.fillColor,
      inpaintMethod: textStyle.inpaintMethod,
      strokeEnabled: textStyle.strokeEnabled,
      strokeColor: textStyle.strokeColor,
      strokeWidth: textStyle.strokeWidth,
      lineSpacing: textStyle.lineSpacing,
      textAlign: textStyle.textAlign,
      rotationAngle: 0,
      position: { x: 0, y: 0 },
      ...overrides
    })
    bubbles.value.push(newBubble)
    syncToCurrentImage()
    return newBubble
  }

  function deleteBubble(index: number): boolean {
    if (index < 0 || index >= bubbles.value.length) {
      return false
    }

    bubbles.value.splice(index, 1)

    if (selectedIndex.value === index) {
      selectedIndex.value = -1
    } else if (selectedIndex.value > index) {
      selectedIndex.value--
    }

    selectedIndices.value = selectedIndices.value
      .filter((i) => i !== index)
      .map((i) => (i > index ? i - 1 : i))

    syncToCurrentImage()
    return true
  }

  function deleteSelected(): void {
    if (selectedIndices.value.length === 0 && selectedIndex.value < 0) {
      return
    }

    const indicesToDelete = [...new Set([...selectedIndices.value, selectedIndex.value])]
      .filter((i) => i >= 0)
      .sort((a, b) => b - a)

    for (const index of indicesToDelete) {
      bubbles.value.splice(index, 1)
    }

    clearSelection()
    syncToCurrentImage()
  }

  function clearBubbles(): void {
    bubbles.value = []
    initialStates.value = []
    clearSelection()
    syncToCurrentImage()
  }

  function clearBubblesLocal(): void {
    bubbles.value = []
    initialStates.value = []
    clearSelection()
  }

  function selectBubble(index: number): void {
    if (index >= -1 && index < bubbles.value.length) {
      selectedIndex.value = index
      selectedIndices.value = index >= 0 ? [index] : []
    }
  }

  function toggleMultiSelect(index: number): void {
    if (index < 0 || index >= bubbles.value.length) return

    const existingIndex = selectedIndices.value.indexOf(index)
    if (existingIndex >= 0) {
      selectedIndices.value.splice(existingIndex, 1)
      if (selectedIndex.value === index) {
        selectedIndex.value = selectedIndices.value[0] ?? -1
      }
    } else {
      selectedIndices.value.push(index)
      selectedIndex.value = index
    }
  }

  function clearSelection(): void {
    selectedIndex.value = -1
    selectedIndices.value = []
  }

  function clearMultiSelect(): void {
    if (selectedIndex.value >= 0) {
      selectedIndices.value = [selectedIndex.value]
    } else {
      selectedIndices.value = []
    }
  }

  function selectNext(): boolean {
    if (bubbles.value.length === 0) return false
    const nextIndex = selectedIndex.value < bubbles.value.length - 1 ? selectedIndex.value + 1 : 0
    selectBubble(nextIndex)
    return true
  }

  function selectPrevious(): boolean {
    if (bubbles.value.length === 0) return false
    const prevIndex = selectedIndex.value > 0 ? selectedIndex.value - 1 : bubbles.value.length - 1
    selectBubble(prevIndex)
    return true
  }

  function updateBubble(index: number, updates: BubbleStateUpdates): boolean {
    if (index < 0 || index >= bubbles.value.length) {
      return false
    }

    const bubble = bubbles.value[index]
    if (bubble) {
      if (updates.coords) {
        updates.autoTextDirection = detectTextDirection(updates.coords)
      }
      Object.assign(bubble, updates)
      syncToCurrentImage()
      return true
    }
    return false
  }

  function updateSelectedBubble(updates: BubbleStateUpdates): boolean {
    if (selectedIndex.value < 0) {
      return false
    }
    return updateBubble(selectedIndex.value, updates)
  }

  function updateAllSelected(updates: BubbleStateUpdates): void {
    const indices = selectedIndices.value.length > 0
      ? selectedIndices.value
      : (selectedIndex.value >= 0 ? [selectedIndex.value] : [])

    for (const index of indices) {
      const bubble = bubbles.value[index]
      if (bubble) {
        const updatesWithAutoDirection = { ...updates }
        if (updates.coords) {
          updatesWithAutoDirection.autoTextDirection = detectTextDirection(updates.coords)
        }
        Object.assign(bubble, updatesWithAutoDirection)
      }
    }
    syncToCurrentImage()
  }

  function updateAllBubbles(updates: BubbleStateUpdates): void {
    for (let i = 0; i < bubbles.value.length; i++) {
      const bubble = bubbles.value[i]
      if (bubble) {
        const updatesWithAutoDirection = { ...updates }
        if (updates.coords) {
          updatesWithAutoDirection.autoTextDirection = detectTextDirection(updates.coords)
        }
        Object.assign(bubble, updatesWithAutoDirection)
      }
    }
    syncToCurrentImage()
  }

  function hasChanges(): boolean {
    if (bubbles.value.length !== initialStates.value.length) {
      return true
    }

    for (let i = 0; i < bubbles.value.length; i++) {
      const current = bubbles.value[i]
      const initial = initialStates.value[i]

      if (!current || !initial) continue

      if (
        current.translatedText !== initial.translatedText ||
        current.textboxText !== initial.textboxText ||
        current.fontSize !== initial.fontSize ||
        current.fontFamily !== initial.fontFamily ||
        current.textDirection !== initial.textDirection ||
        current.textColor !== initial.textColor ||
        current.fillColor !== initial.fillColor ||
        current.rotationAngle !== initial.rotationAngle ||
        current.strokeEnabled !== initial.strokeEnabled ||
        current.strokeColor !== initial.strokeColor ||
        current.strokeWidth !== initial.strokeWidth ||
        current.lineSpacing !== initial.lineSpacing ||
        current.textAlign !== initial.textAlign ||
        current.inpaintMethod !== initial.inpaintMethod ||
        JSON.stringify(current.coords) !== JSON.stringify(initial.coords)
      ) {
        return true
      }
    }

    return false
  }

  function resetToInitial(): void {
    bubbles.value = cloneBubbleStates(initialStates.value)
    clearSelection()
    syncToCurrentImage()
  }

  function saveAsInitial(): void {
    initialStates.value = cloneBubbleStates(bubbles.value)
  }

  function toApiRequest(): {
    bubble_coords: BubbleCoords[]
    bubble_texts: string[]
    textbox_texts: string[]
    font_sizes: number[]
    font_families: string[]
    text_directions: string[]
    text_colors: string[]
    fill_colors: string[]
    rotation_angles: number[]
    stroke_enabled: boolean[]
    stroke_colors: string[]
    stroke_widths: number[]
    line_spacings: number[]
    text_aligns: string[]
    inpaint_methods: string[]
  } {
    return {
      bubble_coords: bubbles.value.map(b => b.coords),
      bubble_texts: bubbles.value.map(b => b.translatedText),
      textbox_texts: bubbles.value.map(b => b.textboxText),
      font_sizes: bubbles.value.map(b => b.fontSize),
      font_families: bubbles.value.map(b => b.fontFamily),
      text_directions: bubbles.value.map(b => {
        if (b.textDirection === 'vertical' || b.textDirection === 'horizontal') {
          return b.textDirection === 'vertical' ? 'v' : 'h'
        }
        if (b.autoTextDirection === 'vertical' || b.autoTextDirection === 'horizontal') {
          return b.autoTextDirection === 'vertical' ? 'v' : 'h'
        }
        return 'v'
      }),
      text_colors: bubbles.value.map(b => b.textColor),
      fill_colors: bubbles.value.map(b => b.fillColor),
      rotation_angles: bubbles.value.map(b => b.rotationAngle),
      stroke_enabled: bubbles.value.map(b => b.strokeEnabled),
      stroke_colors: bubbles.value.map(b => b.strokeColor),
      stroke_widths: bubbles.value.map(b => b.strokeWidth),
      line_spacings: bubbles.value.map(b => b.lineSpacing),
      text_aligns: bubbles.value.map(b => b.textAlign),
      inpaint_methods: bubbles.value.map(b => b.inpaintMethod)
    }
  }

  function serialize(): string {
    return JSON.stringify(bubbles.value)
  }

  function deserialize(json: string): boolean {
    try {
      const parsed = JSON.parse(json)
      if (!Array.isArray(parsed)) {
        return false
      }

      const validStates: BubbleState[] = []
      for (const item of parsed) {
        if (isValidBubbleState(item)) {
          validStates.push(item as BubbleState)
        }
      }

      setBubbles(validStates)
      return true
    } catch {
      return false
    }
  }

  return {
    bubbles,
    selectedIndex,
    selectedIndices,
    initialStates,

    selectedBubble,
    bubbleCount,
    hasBubbles,
    hasSelection,
    isMultiSelect,
    selectedBubbles,

    setBubbles,
    addBubble,
    deleteBubble,
    deleteSelected,
    clearBubbles,
    clearBubblesLocal,

    selectBubble,
    toggleMultiSelect,
    clearSelection,
    clearMultiSelect,
    selectNext,
    selectPrevious,

    updateBubble,
    updateSelectedBubble,
    updateAllSelected,
    updateAllBubbles,

    hasChanges,
    resetToInitial,
    saveAsInitial,

    toApiRequest,
    serialize,
    deserialize
  }
})
