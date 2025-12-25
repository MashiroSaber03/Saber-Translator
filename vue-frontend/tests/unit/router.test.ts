/**
 * 路由配置单元测试
 * 验证路由配置的正确性
 */
import { describe, it, expect } from 'vitest'
import router from '@/router'

describe('路由配置', () => {
  it('应该包含书架路由', () => {
    const route = router.getRoutes().find(r => r.name === 'bookshelf')
    expect(route).toBeDefined()
    expect(route?.path).toBe('/')
  })

  it('应该包含翻译路由', () => {
    const route = router.getRoutes().find(r => r.name === 'translate')
    expect(route).toBeDefined()
    expect(route?.path).toBe('/translate')
  })

  it('应该包含阅读器路由', () => {
    const route = router.getRoutes().find(r => r.name === 'reader')
    expect(route).toBeDefined()
    expect(route?.path).toBe('/reader')
  })

  it('应该包含漫画分析路由', () => {
    const route = router.getRoutes().find(r => r.name === 'insight')
    expect(route).toBeDefined()
    expect(route?.path).toBe('/insight')
  })
})
