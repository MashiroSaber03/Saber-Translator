/**
 * 响应式布局组合式函数
 * 提供屏幕尺寸检测、断点判断和布局自适应功能
 * 
 * 功能：
 * - 屏幕尺寸实时监测
 * - 断点判断（移动端、平板、桌面）
 * - 侧边栏显示/隐藏控制
 * - 布局模式自动切换
 * 
 * @module useResponsive
 */

import { ref, computed, onMounted, onUnmounted, readonly } from 'vue'

// ============================================================
// 断点常量定义
// ============================================================

/** 断点配置 */
export const BREAKPOINTS = {
  /** 超小屏幕（手机竖屏） */
  XS: 360,
  /** 小屏幕（手机横屏） */
  SM: 480,
  /** 中等屏幕（平板竖屏） */
  MD: 768,
  /** 大屏幕（平板横屏/小桌面） */
  LG: 1024,
  /** 超大屏幕（桌面） */
  XL: 1200,
  /** 超超大屏幕（大桌面） */
  XXL: 1440,
} as const

/** 设备类型 */
export type DeviceType = 'mobile' | 'tablet' | 'desktop'

/** 屏幕方向 */
export type Orientation = 'portrait' | 'landscape'

/** 布局模式 */
export type LayoutMode = 'compact' | 'normal' | 'wide'

// ============================================================
// 响应式状态
// ============================================================

/** 窗口宽度 */
const windowWidth = ref(typeof window !== 'undefined' ? window.innerWidth : 1024)

/** 窗口高度 */
const windowHeight = ref(typeof window !== 'undefined' ? window.innerHeight : 768)

/** 侧边栏是否可见（移动端控制） */
const sidebarVisible = ref(true)

/** 是否为触摸设备 */
const isTouchDevice = ref(false)

/** 监听器是否已初始化 */
let isInitialized = false

/** 监听器引用计数 */
let listenerCount = 0

// ============================================================
// 事件处理函数
// ============================================================

/**
 * 处理窗口大小变化
 */
function handleResize() {
  windowWidth.value = window.innerWidth
  windowHeight.value = window.innerHeight
  
  // 在桌面端自动显示侧边栏
  if (windowWidth.value >= BREAKPOINTS.MD) {
    sidebarVisible.value = true
  }
}

/**
 * 检测是否为触摸设备
 */
function detectTouchDevice() {
  isTouchDevice.value = 
    'ontouchstart' in window ||
    navigator.maxTouchPoints > 0 ||
    // @ts-ignore - 兼容旧版浏览器
    navigator.msMaxTouchPoints > 0
}

/**
 * 初始化监听器
 */
function initListeners() {
  if (typeof window === 'undefined') return
  
  if (!isInitialized) {
    window.addEventListener('resize', handleResize)
    detectTouchDevice()
    isInitialized = true
  }
  listenerCount++
}

/**
 * 清理监听器
 */
function cleanupListeners() {
  listenerCount--
  if (listenerCount <= 0 && isInitialized) {
    window.removeEventListener('resize', handleResize)
    isInitialized = false
    listenerCount = 0
  }
}

// ============================================================
// 组合式函数
// ============================================================

/**
 * 响应式布局组合式函数
 * 
 * @example
 * ```ts
 * const { 
 *   isMobile, 
 *   isTablet, 
 *   isDesktop,
 *   sidebarVisible,
 *   toggleSidebar 
 * } = useResponsive()
 * ```
 */
export function useResponsive() {
  // 生命周期
  onMounted(() => {
    initListeners()
    handleResize()
  })
  
  onUnmounted(() => {
    cleanupListeners()
  })
  
  // ============================================================
  // 计算属性
  // ============================================================
  
  /** 是否为超小屏幕（< 360px） */
  const isXs = computed(() => windowWidth.value < BREAKPOINTS.XS)
  
  /** 是否为小屏幕（< 480px） */
  const isSm = computed(() => windowWidth.value < BREAKPOINTS.SM)
  
  /** 是否为中等屏幕（< 768px） */
  const isMd = computed(() => windowWidth.value < BREAKPOINTS.MD)
  
  /** 是否为大屏幕（< 1024px） */
  const isLg = computed(() => windowWidth.value < BREAKPOINTS.LG)
  
  /** 是否为超大屏幕（< 1200px） */
  const isXl = computed(() => windowWidth.value < BREAKPOINTS.XL)
  
  /** 是否为移动端（< 768px） */
  const isMobile = computed(() => windowWidth.value < BREAKPOINTS.MD)
  
  /** 是否为平板（768px - 1024px） */
  const isTablet = computed(() => 
    windowWidth.value >= BREAKPOINTS.MD && windowWidth.value < BREAKPOINTS.LG
  )
  
  /** 是否为桌面端（>= 1024px） */
  const isDesktop = computed(() => windowWidth.value >= BREAKPOINTS.LG)
  
  /** 设备类型 */
  const deviceType = computed<DeviceType>(() => {
    if (isMobile.value) return 'mobile'
    if (isTablet.value) return 'tablet'
    return 'desktop'
  })
  
  /** 屏幕方向 */
  const orientation = computed<Orientation>(() => 
    windowWidth.value > windowHeight.value ? 'landscape' : 'portrait'
  )
  
  /** 是否为横屏 */
  const isLandscape = computed(() => orientation.value === 'landscape')
  
  /** 是否为竖屏 */
  const isPortrait = computed(() => orientation.value === 'portrait')
  
  /** 布局模式 */
  const layoutMode = computed<LayoutMode>(() => {
    if (windowWidth.value < BREAKPOINTS.MD) return 'compact'
    if (windowWidth.value < BREAKPOINTS.XL) return 'normal'
    return 'wide'
  })
  
  /** 是否应该显示侧边栏 */
  const shouldShowSidebar = computed(() => {
    // 桌面端始终显示
    if (isDesktop.value) return true
    // 移动端/平板根据状态显示
    return sidebarVisible.value
  })
  
  /** 侧边栏宽度（根据屏幕尺寸自适应） */
  const sidebarWidth = computed(() => {
    if (windowWidth.value < BREAKPOINTS.SM) return '100%'
    if (windowWidth.value < BREAKPOINTS.MD) return '280px'
    if (windowWidth.value < BREAKPOINTS.LG) return '240px'
    return '280px'
  })
  
  /** 内容区域是否需要偏移（侧边栏显示时） */
  const contentOffset = computed(() => {
    if (!shouldShowSidebar.value) return '0'
    if (isMobile.value) return '0' // 移动端侧边栏覆盖显示
    return sidebarWidth.value
  })
  
  // ============================================================
  // 方法
  // ============================================================
  
  /**
   * 切换侧边栏显示状态
   */
  function toggleSidebar() {
    sidebarVisible.value = !sidebarVisible.value
  }
  
  /**
   * 显示侧边栏
   */
  function showSidebar() {
    sidebarVisible.value = true
  }
  
  /**
   * 隐藏侧边栏
   */
  function hideSidebar() {
    sidebarVisible.value = false
  }
  
  /**
   * 检查当前宽度是否大于等于指定断点
   * @param breakpoint 断点值
   */
  function isAbove(breakpoint: number): boolean {
    return windowWidth.value >= breakpoint
  }
  
  /**
   * 检查当前宽度是否小于指定断点
   * @param breakpoint 断点值
   */
  function isBelow(breakpoint: number): boolean {
    return windowWidth.value < breakpoint
  }
  
  /**
   * 检查当前宽度是否在指定范围内
   * @param min 最小值
   * @param max 最大值
   */
  function isBetween(min: number, max: number): boolean {
    return windowWidth.value >= min && windowWidth.value < max
  }
  
  /**
   * 根据断点返回对应的值
   * @param values 断点值映射
   */
  function responsive<T>(values: {
    xs?: T
    sm?: T
    md?: T
    lg?: T
    xl?: T
    xxl?: T
    default: T
  }): T {
    if (windowWidth.value >= BREAKPOINTS.XXL && values.xxl !== undefined) return values.xxl
    if (windowWidth.value >= BREAKPOINTS.XL && values.xl !== undefined) return values.xl
    if (windowWidth.value >= BREAKPOINTS.LG && values.lg !== undefined) return values.lg
    if (windowWidth.value >= BREAKPOINTS.MD && values.md !== undefined) return values.md
    if (windowWidth.value >= BREAKPOINTS.SM && values.sm !== undefined) return values.sm
    if (windowWidth.value >= BREAKPOINTS.XS && values.xs !== undefined) return values.xs
    return values.default
  }
  
  // ============================================================
  // 返回值
  // ============================================================
  
  return {
    // 尺寸
    windowWidth: readonly(windowWidth),
    windowHeight: readonly(windowHeight),
    
    // 断点判断
    isXs,
    isSm,
    isMd,
    isLg,
    isXl,
    
    // 设备类型
    isMobile,
    isTablet,
    isDesktop,
    deviceType,
    
    // 方向
    orientation,
    isLandscape,
    isPortrait,
    
    // 布局
    layoutMode,
    sidebarWidth,
    contentOffset,
    
    // 侧边栏
    sidebarVisible: readonly(sidebarVisible),
    shouldShowSidebar,
    toggleSidebar,
    showSidebar,
    hideSidebar,
    
    // 触摸设备
    isTouchDevice: readonly(isTouchDevice),
    
    // 工具方法
    isAbove,
    isBelow,
    isBetween,
    responsive,
    
    // 常量
    BREAKPOINTS,
  }
}

// 导出默认实例（用于非组件场景）
export default useResponsive
