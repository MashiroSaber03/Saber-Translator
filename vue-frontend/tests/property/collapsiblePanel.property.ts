/**
 * 折叠面板状态属性测试
 * 
 * **Feature: vue-frontend-migration, Property 47: 折叠面板状态一致性**
 * **Validates: Requirements 29.1**
 * 
 * 测试内容：
 * - 面板展开/折叠状态切换正确
 * - 多个面板独立状态管理正确
 * - 面板状态持久化正确（可选）
 */

import { describe, it, expect, beforeEach } from 'vitest'
import * as fc from 'fast-check'

// ============================================================
// 类型定义
// ============================================================

/**
 * 折叠面板状态
 */
interface CollapsiblePanelState {
  /** 面板ID */
  id: string
  /** 是否展开 */
  isExpanded: boolean
}

/**
 * 面板管理器状态
 */
interface PanelManagerState {
  /** 所有面板状态 */
  panels: Map<string, boolean>
}

// ============================================================
// 面板状态管理函数（模拟组件逻辑）
// ============================================================

/**
 * 创建面板管理器
 */
function createPanelManager(): PanelManagerState {
  return {
    panels: new Map()
  }
}

/**
 * 注册面板
 * @param manager 面板管理器
 * @param panelId 面板ID
 * @param initialExpanded 初始展开状态
 */
function registerPanel(
  manager: PanelManagerState,
  panelId: string,
  initialExpanded: boolean = true
): void {
  manager.panels.set(panelId, initialExpanded)
}

/**
 * 切换面板状态
 * @param manager 面板管理器
 * @param panelId 面板ID
 * @returns 切换后的状态
 */
function togglePanel(manager: PanelManagerState, panelId: string): boolean {
  const currentState = manager.panels.get(panelId)
  if (currentState === undefined) {
    // 面板不存在，注册并设为展开
    manager.panels.set(panelId, true)
    return true
  }
  const newState = !currentState
  manager.panels.set(panelId, newState)
  return newState
}

/**
 * 获取面板状态
 * @param manager 面板管理器
 * @param panelId 面板ID
 * @returns 面板是否展开，不存在返回 undefined
 */
function getPanelState(manager: PanelManagerState, panelId: string): boolean | undefined {
  return manager.panels.get(panelId)
}

/**
 * 设置面板状态
 * @param manager 面板管理器
 * @param panelId 面板ID
 * @param isExpanded 是否展开
 */
function setPanelState(
  manager: PanelManagerState,
  panelId: string,
  isExpanded: boolean
): void {
  manager.panels.set(panelId, isExpanded)
}

/**
 * 展开所有面板
 * @param manager 面板管理器
 */
function expandAllPanels(manager: PanelManagerState): void {
  for (const panelId of manager.panels.keys()) {
    manager.panels.set(panelId, true)
  }
}

/**
 * 折叠所有面板
 * @param manager 面板管理器
 */
function collapseAllPanels(manager: PanelManagerState): void {
  for (const panelId of manager.panels.keys()) {
    manager.panels.set(panelId, false)
  }
}

/**
 * 序列化面板状态（用于持久化）
 * @param manager 面板管理器
 * @returns JSON 字符串
 */
function serializePanelState(manager: PanelManagerState): string {
  const obj: Record<string, boolean> = {}
  for (const [key, value] of manager.panels.entries()) {
    obj[key] = value
  }
  return JSON.stringify(obj)
}

/**
 * 反序列化面板状态
 * @param json JSON 字符串
 * @returns 面板管理器
 */
function deserializePanelState(json: string): PanelManagerState {
  const manager = createPanelManager()
  try {
    const obj = JSON.parse(json) as Record<string, boolean>
    // 使用 Object.keys 而不是 Object.entries，避免原型链属性问题
    for (const key of Object.keys(obj)) {
      // 确保是自有属性且值是布尔类型
      if (Object.prototype.hasOwnProperty.call(obj, key) && typeof obj[key] === 'boolean') {
        manager.panels.set(key, obj[key])
      }
    }
  } catch {
    // 解析失败，返回空管理器
  }
  return manager
}

// ============================================================
// 生成器
// ============================================================

/**
 * 生成有效的面板ID
 */
const validPanelIdArb = fc.stringOf(
  fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz0123456789-_'.split('')),
  { minLength: 1, maxLength: 20 }
)

/**
 * 生成面板状态
 */
const panelStateArb: fc.Arbitrary<CollapsiblePanelState> = fc.record({
  id: validPanelIdArb,
  isExpanded: fc.boolean()
})

/**
 * 生成面板状态数组
 */
const panelStatesArb = fc.array(panelStateArb, { minLength: 1, maxLength: 10 })

// ============================================================
// 属性测试
// ============================================================

describe('折叠面板状态属性测试', () => {
  let manager: PanelManagerState

  beforeEach(() => {
    manager = createPanelManager()
  })

  /**
   * Property 47.1: 面板切换状态正确
   * 对于任意面板，切换一次后状态应该取反
   */
  it('Property 47.1: 面板切换状态正确', () => {
    fc.assert(
      fc.property(
        validPanelIdArb,
        fc.boolean(),
        (panelId, initialState) => {
          // 重置管理器
          manager = createPanelManager()
          
          // 注册面板
          registerPanel(manager, panelId, initialState)
          
          // 验证初始状态
          expect(getPanelState(manager, panelId)).toBe(initialState)
          
          // 切换状态
          const newState = togglePanel(manager, panelId)
          
          // 验证切换后状态取反
          expect(newState).toBe(!initialState)
          expect(getPanelState(manager, panelId)).toBe(!initialState)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 47.2: 双击切换恢复原状态
   * 对于任意面板，切换两次后应该恢复原状态
   */
  it('Property 47.2: 双击切换恢复原状态', () => {
    fc.assert(
      fc.property(
        validPanelIdArb,
        fc.boolean(),
        (panelId, initialState) => {
          // 重置管理器
          manager = createPanelManager()
          
          // 注册面板
          registerPanel(manager, panelId, initialState)
          
          // 切换两次
          togglePanel(manager, panelId)
          togglePanel(manager, panelId)
          
          // 验证恢复原状态
          expect(getPanelState(manager, panelId)).toBe(initialState)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 47.3: 多个面板独立状态管理
   * 切换一个面板不应影响其他面板的状态
   */
  it('Property 47.3: 多个面板独立状态管理', () => {
    fc.assert(
      fc.property(
        panelStatesArb,
        fc.nat({ max: 9 }),
        (panels, toggleIndex) => {
          // 重置管理器
          manager = createPanelManager()
          
          // 确保面板ID唯一
          const uniquePanels = panels.filter(
            (panel, index, self) => 
              self.findIndex(p => p.id === panel.id) === index
          )
          
          if (uniquePanels.length < 2) {
            return true // 跳过面板数量不足的情况
          }
          
          // 注册所有面板
          for (const panel of uniquePanels) {
            registerPanel(manager, panel.id, panel.isExpanded)
          }
          
          // 记录所有面板的初始状态
          const initialStates = new Map<string, boolean>()
          for (const panel of uniquePanels) {
            initialStates.set(panel.id, panel.isExpanded)
          }
          
          // 选择要切换的面板
          const targetIndex = toggleIndex % uniquePanels.length
          const targetPanel = uniquePanels[targetIndex]
          if (!targetPanel) return true
          
          // 切换目标面板
          togglePanel(manager, targetPanel.id)
          
          // 验证其他面板状态未变
          for (const panel of uniquePanels) {
            if (panel.id !== targetPanel.id) {
              expect(getPanelState(manager, panel.id)).toBe(initialStates.get(panel.id))
            }
          }
          
          // 验证目标面板状态已变
          expect(getPanelState(manager, targetPanel.id)).toBe(!initialStates.get(targetPanel.id))
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 47.4: 面板状态序列化往返一致性
   * 序列化后反序列化应该得到相同的状态
   */
  it('Property 47.4: 面板状态序列化往返一致性', () => {
    fc.assert(
      fc.property(
        panelStatesArb,
        (panels) => {
          // 重置管理器
          manager = createPanelManager()
          
          // 确保面板ID唯一
          const uniquePanels = panels.filter(
            (panel, index, self) => 
              self.findIndex(p => p.id === panel.id) === index
          )
          
          // 注册所有面板
          for (const panel of uniquePanels) {
            registerPanel(manager, panel.id, panel.isExpanded)
          }
          
          // 序列化
          const serialized = serializePanelState(manager)
          
          // 反序列化
          const restored = deserializePanelState(serialized)
          
          // 验证状态一致
          for (const panel of uniquePanels) {
            expect(restored.panels.get(panel.id)).toBe(panel.isExpanded)
          }
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 47.5: 展开所有面板后所有面板都展开
   */
  it('Property 47.5: 展开所有面板后所有面板都展开', () => {
    fc.assert(
      fc.property(
        panelStatesArb,
        (panels) => {
          // 重置管理器
          manager = createPanelManager()
          
          // 确保面板ID唯一
          const uniquePanels = panels.filter(
            (panel, index, self) => 
              self.findIndex(p => p.id === panel.id) === index
          )
          
          // 注册所有面板（可能有些折叠）
          for (const panel of uniquePanels) {
            registerPanel(manager, panel.id, panel.isExpanded)
          }
          
          // 展开所有面板
          expandAllPanels(manager)
          
          // 验证所有面板都展开
          for (const panel of uniquePanels) {
            expect(getPanelState(manager, panel.id)).toBe(true)
          }
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 47.6: 折叠所有面板后所有面板都折叠
   */
  it('Property 47.6: 折叠所有面板后所有面板都折叠', () => {
    fc.assert(
      fc.property(
        panelStatesArb,
        (panels) => {
          // 重置管理器
          manager = createPanelManager()
          
          // 确保面板ID唯一
          const uniquePanels = panels.filter(
            (panel, index, self) => 
              self.findIndex(p => p.id === panel.id) === index
          )
          
          // 注册所有面板（可能有些展开）
          for (const panel of uniquePanels) {
            registerPanel(manager, panel.id, panel.isExpanded)
          }
          
          // 折叠所有面板
          collapseAllPanels(manager)
          
          // 验证所有面板都折叠
          for (const panel of uniquePanels) {
            expect(getPanelState(manager, panel.id)).toBe(false)
          }
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 47.7: 设置面板状态正确
   */
  it('Property 47.7: 设置面板状态正确', () => {
    fc.assert(
      fc.property(
        validPanelIdArb,
        fc.boolean(),
        fc.boolean(),
        (panelId, initialState, newState) => {
          // 重置管理器
          manager = createPanelManager()
          
          // 注册面板
          registerPanel(manager, panelId, initialState)
          
          // 设置新状态
          setPanelState(manager, panelId, newState)
          
          // 验证状态正确
          expect(getPanelState(manager, panelId)).toBe(newState)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
