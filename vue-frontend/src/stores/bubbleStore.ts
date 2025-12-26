/**
 * 气泡状态管理 Store
 * 管理编辑模式下的气泡状态、选择、增删改操作
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  BubbleState,
  BubbleCoords,
  BubbleStateUpdates
} from '@/types/bubble'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'

// 从 bubbleFactory 统一导入 store 内部使用的工厂函数
import {
  createBubbleState,
  cloneBubbleStates,
  isValidBubbleState,
  detectTextDirection
} from '@/utils/bubbleFactory'

// ============================================================
// Store 定义
// ============================================================

export const useBubbleStore = defineStore('bubble', () => {
  // ============================================================
  // 状态定义
  // ============================================================

  /** 气泡状态数组 */
  const bubbles = ref<BubbleState[]>([])

  /** 当前选中的气泡索引（-1 表示未选中） */
  const selectedIndex = ref<number>(-1)

  /** 多选的气泡索引数组 */
  const selectedIndices = ref<number[]>([])

  /** 初始气泡状态（用于检测变更） */
  const initialStates = ref<BubbleState[]>([])

  // ============================================================
  // 拖动状态（共享给所有BubbleOverlay组件）
  // ============================================================

  /** 是否正在拖动 */
  const isDragging = ref(false)
  /** 拖动的气泡索引 */
  const draggingIndex = ref(-1)
  /** 拖动X偏移 */
  const dragOffsetX = ref(0)
  /** 拖动Y偏移 */
  const dragOffsetY = ref(0)
  /** 拖动初始X */
  const dragInitialX = ref(0)
  /** 拖动初始Y */
  const dragInitialY = ref(0)

  /** 是否正在调整大小 */
  const isResizing = ref(false)
  /** 调整大小的气泡索引 */
  const resizingIndex = ref(-1)
  /** 调整大小当前坐标 */
  const resizeCurrentCoords = ref<[number, number, number, number] | null>(null)

  /** 是否正在旋转 */
  const isRotating = ref(false)
  /** 旋转的气泡索引 */
  const rotatingIndex = ref(-1)
  /** 旋转当前角度 */
  const rotateCurrentAngle = ref(0)

  // ============================================================
  // 计算属性
  // ============================================================

  /** 当前选中的气泡 */
  const selectedBubble = computed<BubbleState | null>(() => {
    if (selectedIndex.value >= 0 && selectedIndex.value < bubbles.value.length) {
      return bubbles.value[selectedIndex.value] ?? null
    }
    return null
  })

  /** 气泡总数 */
  const bubbleCount = computed<number>(() => bubbles.value.length)

  /** 是否有气泡 */
  const hasBubbles = computed<boolean>(() => bubbles.value.length > 0)

  /** 是否有选中的气泡 */
  const hasSelection = computed<boolean>(() => selectedIndex.value >= 0)

  /** 是否为多选模式 */
  const isMultiSelect = computed<boolean>(() => selectedIndices.value.length > 1)

  /** 所有选中的气泡 */
  const selectedBubbles = computed<BubbleState[]>(() => {
    return selectedIndices.value
      .filter((i) => i >= 0 && i < bubbles.value.length)
      .map((i) => bubbles.value[i])
      .filter((b): b is BubbleState => b !== undefined)
  })

  // ============================================================
  // 同步方法（复刻旧版 updateSingleBubbleState 的同步逻辑）
  // ============================================================

  /**
   * 将当前气泡状态同步到 imageStore.currentImage.bubbleStates
   * 复刻旧版 state.js 中 updateSingleBubbleState 的同步逻辑
   * 确保切换图片后气泡状态不丢失
   */
  function syncToCurrentImage(): void {
    const imageStore = useImageStore()
    const currentImage = imageStore.currentImage
    if (currentImage) {
      currentImage.bubbleStates = cloneBubbleStates(bubbles.value)
      currentImage.hasUnsavedChanges = true
      console.log('气泡状态已同步到当前图片')
    }
  }

  // ============================================================
  // 气泡管理方法
  // ============================================================

  /**
   * 设置气泡数组
   * @param newBubbles - 新的气泡数组
   * @param skipSync - 是否跳过同步到 imageStore（加载时使用）
   */
  function setBubbles(newBubbles: BubbleState[], skipSync: boolean = false): void {
    bubbles.value = newBubbles
    initialStates.value = cloneBubbleStates(newBubbles)
    clearSelection()
    console.log(`气泡数组已设置，共 ${newBubbles.length} 个气泡`)
    // 设置初始数据时通常不需要同步（避免覆盖），但可以选择同步
    if (!skipSync) {
      syncToCurrentImage()
    }
  }

  /**
   * 添加气泡
   * 【复刻原版】从 settingsStore 读取当前 UI 设置作为新气泡的默认值
   * @param coords - 气泡坐标
   * @param overrides - 可选的覆盖属性
   * @returns 新添加的气泡
   */
  function addBubble(coords: BubbleCoords, overrides?: Partial<BubbleState>): BubbleState {
    // 自动计算排版方向
    const autoDirection = detectTextDirection(coords)

    // 【复刻原版 edit_mode.js addNewBubble】从 settingsStore 读取当前 UI 设置
    const settingsStore = useSettingsStore()
    const textStyle = settingsStore.settings.textStyle

    // 处理自动排版：如果选择"auto"，回退到默认的 'vertical'
    const layoutDirection = textStyle.layoutDirection
    const bubbleTextDirection = layoutDirection === 'auto' ? 'vertical' : (layoutDirection || 'vertical')

    const newBubble = createBubbleState({
      coords,
      translatedText: '',
      autoTextDirection: autoDirection,
      // 从当前 UI 设置读取默认值
      fontSize: textStyle.fontSize,
      fontFamily: textStyle.fontFamily,
      textDirection: bubbleTextDirection as 'vertical' | 'horizontal' | 'auto',
      textColor: textStyle.textColor,
      fillColor: textStyle.fillColor,
      inpaintMethod: textStyle.inpaintMethod,
      strokeEnabled: textStyle.strokeEnabled,
      strokeColor: textStyle.strokeColor,
      strokeWidth: textStyle.strokeWidth,
      rotationAngle: 0,
      position: { x: 0, y: 0 },
      // 允许 overrides 覆盖上述默认值
      ...overrides
    })
    bubbles.value.push(newBubble)
    console.log(`已添加气泡，当前共 ${bubbles.value.length} 个`)
    // 【同步】添加气泡后同步到 currentImage
    syncToCurrentImage()
    return newBubble
  }

  /**
   * 删除指定索引的气泡
   * @param index - 气泡索引
   * @returns 是否删除成功
   */
  function deleteBubble(index: number): boolean {
    if (index < 0 || index >= bubbles.value.length) {
      console.warn(`删除失败: 无效的索引 ${index}`)
      return false
    }

    bubbles.value.splice(index, 1)

    // 调整选中索引
    if (selectedIndex.value === index) {
      selectedIndex.value = -1
    } else if (selectedIndex.value > index) {
      selectedIndex.value--
    }

    // 调整多选索引
    selectedIndices.value = selectedIndices.value
      .filter((i) => i !== index)
      .map((i) => (i > index ? i - 1 : i))

    console.log(`已删除气泡，当前共 ${bubbles.value.length} 个`)
    // 【同步】删除气泡后同步到 currentImage
    syncToCurrentImage()
    return true
  }

  /**
   * 删除所有选中的气泡
   */
  function deleteSelected(): void {
    if (selectedIndices.value.length === 0 && selectedIndex.value < 0) {
      return
    }

    // 获取要删除的索引（去重并排序）
    const indicesToDelete = [...new Set([...selectedIndices.value, selectedIndex.value])]
      .filter((i) => i >= 0)
      .sort((a, b) => b - a) // 从大到小排序，避免索引偏移问题

    // 从后往前删除
    for (const index of indicesToDelete) {
      bubbles.value.splice(index, 1)
    }

    // 清除选择
    clearSelection()
    console.log(`已删除 ${indicesToDelete.length} 个气泡`)
    // 【同步】批量删除后同步到 currentImage
    syncToCurrentImage()
  }

  /**
   * 清除所有气泡
   */
  function clearBubbles(): void {
    bubbles.value = []
    initialStates.value = []
    clearSelection()
    console.log('所有气泡已清除')
    // 【同步】清除后同步到 currentImage
    syncToCurrentImage()
  }

  /**
   * 【复刻原版】仅清除本地气泡状态，不同步到 imageStore
   * 用于加载图片时：如果原图 bubbleStates 为 null，不应该把它写成 []
   * 这保持了 null（未处理）和 []（用户主动清空）的语义区分
   */
  function clearBubblesLocal(): void {
    bubbles.value = []
    initialStates.value = []
    clearSelection()
    console.log('本地气泡状态已清除（不同步到imageStore）')
  }

  // ============================================================
  // 选择管理方法
  // ============================================================

  /**
   * 选择气泡
   * @param index - 气泡索引
   */
  function selectBubble(index: number): void {
    if (index >= -1 && index < bubbles.value.length) {
      selectedIndex.value = index
      // 单选时清除多选
      selectedIndices.value = index >= 0 ? [index] : []
      console.log(`已选中气泡 ${index}`)
    }
  }

  /**
   * 切换多选状态
   * @param index - 气泡索引
   */
  function toggleMultiSelect(index: number): void {
    if (index < 0 || index >= bubbles.value.length) return

    const existingIndex = selectedIndices.value.indexOf(index)
    if (existingIndex >= 0) {
      // 已选中，取消选择
      selectedIndices.value.splice(existingIndex, 1)
      if (selectedIndex.value === index) {
        selectedIndex.value = selectedIndices.value[0] ?? -1
      }
    } else {
      // 未选中，添加到多选
      selectedIndices.value.push(index)
      selectedIndex.value = index
    }
    console.log(`多选状态: ${selectedIndices.value.join(', ')}`)
  }

  /**
   * 清除选择
   */
  function clearSelection(): void {
    selectedIndex.value = -1
    selectedIndices.value = []
  }

  /**
   * 清除多选（保留主选择）
   */
  function clearMultiSelect(): void {
    if (selectedIndex.value >= 0) {
      selectedIndices.value = [selectedIndex.value]
    } else {
      selectedIndices.value = []
    }
  }

  /**
   * 选择下一个气泡
   * @returns 是否成功
   */
  function selectNext(): boolean {
    if (bubbles.value.length === 0) return false
    const nextIndex = selectedIndex.value < bubbles.value.length - 1 ? selectedIndex.value + 1 : 0
    selectBubble(nextIndex)
    return true
  }

  /**
   * 选择上一个气泡
   * @returns 是否成功
   */
  function selectPrevious(): boolean {
    if (bubbles.value.length === 0) return false
    const prevIndex = selectedIndex.value > 0 ? selectedIndex.value - 1 : bubbles.value.length - 1
    selectBubble(prevIndex)
    return true
  }

  /**
   * 更新指定索引的气泡状态
   * @param index - 气泡索引
   * @param updates - 要更新的属性
   * @returns 是否更新成功
   */
  function updateBubble(index: number, updates: BubbleStateUpdates): boolean {
    if (index < 0 || index >= bubbles.value.length) {
      console.warn(`更新失败: 无效的索引 ${index}`)
      return false
    }

    const bubble = bubbles.value[index]
    if (bubble) {
      Object.assign(bubble, updates)
      console.log(`已更新气泡 ${index}`)
      // 【同步】复刻旧版逻辑：每次更新单个气泡时同步到 currentImage
      syncToCurrentImage()
      return true
    }
    return false
  }

  /**
   * 更新当前选中的气泡
   * @param updates - 要更新的属性
   * @returns 是否更新成功
   */
  function updateSelectedBubble(updates: BubbleStateUpdates): boolean {
    if (selectedIndex.value < 0) {
      console.warn('更新失败: 没有选中的气泡')
      return false
    }
    return updateBubble(selectedIndex.value, updates)
  }

  /**
   * 批量更新所有选中的气泡
   * @param updates - 要更新的属性
   */
  function updateAllSelected(updates: BubbleStateUpdates): void {
    const indices = selectedIndices.value.length > 0
      ? selectedIndices.value
      : (selectedIndex.value >= 0 ? [selectedIndex.value] : [])

    for (const index of indices) {
      const bubble = bubbles.value[index]
      if (bubble) {
        Object.assign(bubble, updates)
      }
    }
    // 【同步】批量更新后统一同步一次
    syncToCurrentImage()
    console.log(`已批量更新 ${indices.length} 个气泡`)
  }

  /**
   * 更新所有气泡的指定属性
   * @param updates - 要更新的属性
   */
  function updateAllBubbles(updates: BubbleStateUpdates): void {
    for (let i = 0; i < bubbles.value.length; i++) {
      const bubble = bubbles.value[i]
      if (bubble) {
        Object.assign(bubble, updates)
      }
    }
    // 【修复问题4】批量更新后同步到 currentImage，确保样式落盘
    syncToCurrentImage()
    console.log(`已更新所有 ${bubbles.value.length} 个气泡`)
  }

  /**
   * 检查气泡状态是否有变更
   * @returns 是否有变更
   */
  function hasChanges(): boolean {
    if (bubbles.value.length !== initialStates.value.length) {
      return true
    }

    for (let i = 0; i < bubbles.value.length; i++) {
      const current = bubbles.value[i]
      const initial = initialStates.value[i]

      // 跳过无效的状态
      if (!current || !initial) continue

      // 检查关键属性是否变更
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
        current.inpaintMethod !== initial.inpaintMethod ||
        JSON.stringify(current.coords) !== JSON.stringify(initial.coords)
      ) {
        return true
      }
    }

    return false
  }

  /**
   * 重置气泡状态到初始状态
   */
  function resetToInitial(): void {
    bubbles.value = cloneBubbleStates(initialStates.value)
    clearSelection()
    console.log('气泡状态已重置到初始状态')
  }

  /**
   * 保存当前状态为初始状态
   */
  function saveAsInitial(): void {
    initialStates.value = cloneBubbleStates(bubbles.value)
    console.log('当前状态已保存为初始状态')
  }

  // ============================================================
  // 序列化方法
  // ============================================================

  /**
   * 将气泡状态转换为 API 请求格式
   * @returns API 请求格式的数据
   */
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
    inpaint_methods: string[]
  } {
    return {
      bubble_coords: bubbles.value.map(b => b.coords),
      bubble_texts: bubbles.value.map(b => b.translatedText),
      textbox_texts: bubbles.value.map(b => b.textboxText),
      font_sizes: bubbles.value.map(b => b.fontSize),
      font_families: bubbles.value.map(b => b.fontFamily),
      text_directions: bubbles.value.map(b => {
        if (b.textDirection === 'auto') {
          return b.autoTextDirection === 'vertical' ? 'v' : 'h'
        }
        return b.textDirection === 'vertical' ? 'v' : 'h'
      }),
      text_colors: bubbles.value.map(b => b.textColor),
      fill_colors: bubbles.value.map(b => b.fillColor),
      rotation_angles: bubbles.value.map(b => b.rotationAngle),
      stroke_enabled: bubbles.value.map(b => b.strokeEnabled),
      stroke_colors: bubbles.value.map(b => b.strokeColor),
      stroke_widths: bubbles.value.map(b => b.strokeWidth),
      inpaint_methods: bubbles.value.map(b => b.inpaintMethod)
    }
  }

  /**
   * 序列化气泡状态为 JSON 字符串
   * @returns JSON 字符串
   */
  function serialize(): string {
    return JSON.stringify(bubbles.value)
  }

  /**
   * 从 JSON 字符串反序列化气泡状态
   * @param json - JSON 字符串
   * @returns 是否成功
   */
  function deserialize(json: string): boolean {
    try {
      const parsed = JSON.parse(json)
      if (!Array.isArray(parsed)) {
        console.error('反序列化失败: 数据不是数组')
        return false
      }

      // 验证每个气泡状态
      const validStates: BubbleState[] = []
      for (const item of parsed) {
        if (isValidBubbleState(item)) {
          validStates.push(item as BubbleState)
        } else {
          console.warn('跳过无效的气泡状态:', item)
        }
      }

      setBubbles(validStates)
      return true
    } catch (error) {
      console.error('反序列化失败:', error)
      return false
    }
  }

  // ============================================================
  // 返回 Store 接口
  // ============================================================

  return {
    // 状态
    bubbles,
    selectedIndex,
    selectedIndices,
    initialStates,

    // 拖动状态（共享）
    isDragging,
    draggingIndex,
    dragOffsetX,
    dragOffsetY,
    dragInitialX,
    dragInitialY,
    isResizing,
    resizingIndex,
    resizeCurrentCoords,
    isRotating,
    rotatingIndex,
    rotateCurrentAngle,

    // 计算属性
    selectedBubble,
    bubbleCount,
    hasBubbles,
    hasSelection,
    isMultiSelect,
    selectedBubbles,

    // 气泡管理
    setBubbles,
    addBubble,
    deleteBubble,
    deleteSelected,
    clearBubbles,
    clearBubblesLocal,

    // 选择管理
    selectBubble,
    toggleMultiSelect,
    clearSelection,
    clearMultiSelect,
    selectNext,
    selectPrevious,

    // 气泡更新
    updateBubble,
    updateSelectedBubble,
    updateAllSelected,
    updateAllBubbles,

    // 状态检测
    hasChanges,
    resetToInitial,
    saveAsInitial,

    // 序列化
    toApiRequest,
    serialize,
    deserialize
  }
})