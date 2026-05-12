import { describe, expect, it } from 'vitest'

import { isUsableImagePrompt, normalizeImagePrompt } from './promptValidation'

describe('promptValidation', () => {
  it('rejects empty and failure-marker prompts', () => {
    expect(isUsableImagePrompt('')).toBe(false)
    expect(isUsableImagePrompt('   ')).toBe(false)
    expect(isUsableImagePrompt('生成失败: 网络错误')).toBe(false)
    expect(isUsableImagePrompt('提示词生成失败: 未知错误')).toBe(false)
    expect(isUsableImagePrompt('出场角色：男主')).toBe(true)
  })

  it('normalizes prompts to at most five non-empty lines', () => {
    const normalized = normalizeImagePrompt(`
      出场角色：男主
      核心动作/情绪：奔跑
      场景：走廊
      关键对白：等等我
      风格约束：保持原作漫画线条、脸型、上色、页面密度和分镜节奏。
      多余说明：这一行应该被截断
    `)

    expect(normalized.split('\n')).toHaveLength(5)
    expect(normalized).not.toContain('多余说明')
  })
})
