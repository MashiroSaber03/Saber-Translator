import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  BubbleState,
  BubbleCoords,
  BubbleStateUpdates
} from '@/types/bubble'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'

import {
  createBubbleState,
  cloneBubbleStates,
  detectTextDirection
} from '@/utils/bubbleFactory'

function bubbleIdentity(bubble: BubbleState | undefined): string | null {
  if (!bubble) return null
  if (bubble.backendBubbleId) return `backend:${bubble.backendBubbleId}`
  if (bubble.clientMutationId) return `client:${bubble.clientMutationId}`
  return null
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

  function syncToCurrentImage(): void {
    const imageStore = useImageStore()
    const currentImage = imageStore.currentImage
    if (currentImage) {
      const clonedBubbles = cloneBubbleStates(bubbles.value)
      currentImage.bubbleStates = clonedBubbles
      currentImage.hasUnsavedChanges = true
    }
  }

  function setBubbles(newBubbles: BubbleState[], skipSync: boolean = false): void {
    const primaryIdentity = bubbleIdentity(selectedBubble.value ?? undefined)
    const selectedIdentities = selectedIndices.value
      .map(index => bubbleIdentity(bubbles.value[index]))
      .filter((identity): identity is string => Boolean(identity))
    const ownedBubbles = cloneBubbleStates(newBubbles)
    bubbles.value = ownedBubbles
    initialStates.value = cloneBubbleStates(ownedBubbles)
    const indexByIdentity = new Map(
      ownedBubbles
        .map((bubble, index) => [bubbleIdentity(bubble), index] as const)
        .filter((entry): entry is readonly [string, number] => Boolean(entry[0])),
    )
    selectedIndices.value = selectedIdentities
      .map(identity => indexByIdentity.get(identity))
      .filter((index): index is number => index !== undefined)
    selectedIndex.value = primaryIdentity
      ? (indexByIdentity.get(primaryIdentity) ?? -1)
      : -1
    if (!skipSync) {
      syncToCurrentImage()
    }
  }

  function addBubble(coords: BubbleCoords, overrides?: Partial<BubbleState>): BubbleState {
    const integerCoords = coords.map(value => Math.round(value)) as BubbleCoords
    const autoDirection = detectTextDirection(integerCoords)

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
      inlineAlign: textStyle.inlineAlign,
      blockAlign: textStyle.blockAlign,
      rotationAngle: 0,
      position: { x: 0, y: 0 },
      ...overrides,
      coords: integerCoords,
    })
    bubbles.value.push(newBubble)
    syncToCurrentImage()
    return newBubble
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

  function updateAllBubbles(updates: BubbleStateUpdates): void {
    for (let i = 0; i < bubbles.value.length; i++) {
      const bubble = bubbles.value[i]
      if (bubble) {
        Object.assign(bubble, updates)
      }
    }
    syncToCurrentImage()
  }

  function saveAsInitial(): void {
    initialStates.value = cloneBubbleStates(bubbles.value)
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

    setBubbles,
    addBubble,
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
    updateAllBubbles,
    saveAsInitial,
  }
})
