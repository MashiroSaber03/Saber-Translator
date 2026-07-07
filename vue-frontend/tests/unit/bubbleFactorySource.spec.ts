import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('bubble factory source contract', () => {
  it('keeps the shared bubble utility compact and clone logic centralized', () => {
    const content = source('src/utils/bubbleFactory.ts')

    for (const staleNarration of [
      '/**',
      '@param',
      '@returns',
      '气泡状态工厂函数',
      '默认气泡状态值',
      '创建气泡状态',
      '根据气泡宽高比自动检测排版方向',
      '从后端响应创建气泡状态数组',
      '将气泡状态数组转换为 API 请求格式',
      '更新单个气泡状态',
      '批量更新所有气泡状态',
      '深拷贝气泡状态数组',
      '深拷贝单个气泡状态',
      '验证气泡状态是否有效',
      '检查点是否在气泡矩形内',
      '检查点是否在多边形内',
      '获取默认气泡设置',
      '初始化气泡状态数组',
    ]) {
      expect(content).not.toContain(staleNarration)
    }

    expect(content).toContain('function cloneBubbleStateFields')
    expect(content.match(/function cloneBubbleStateFields/g)).toHaveLength(1)
  })

  it('keeps the bubble factory property suite focused on behavior contracts', () => {
    const content = source('tests/property/bubbleFactory.property.ts')

    for (const staleNarration of [
      '气泡工厂函数属性测试',
      '测试数据生成器',
      '其他工厂函数测试',
      '验证逻辑测试',
      '宽高比判断测试',
      '点击检测一致性',
      '状态初始化一致性',
      '测试 createBubbleState',
      '测试 isPointInPolygon',
      '生成有效',
      '验证',
      '// ============================================================',
      '/' + '**',
    ]) {
      expect(content).not.toContain(staleNarration)
    }
  })
})
