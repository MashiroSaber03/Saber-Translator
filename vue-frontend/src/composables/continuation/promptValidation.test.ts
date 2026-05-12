import { describe, expect, it } from 'vitest'

import { hasUsableStoryContent, isUsableImagePrompt, normalizeImagePrompt } from './promptValidation'

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

  it('allows the continuation-driven final prompt to keep role form and style lines', () => {
    const normalized = normalizeImagePrompt(`
      上一页剧情：第一页里主角离开教室。
      本页承接：接续上一页剧情，本页发生以下内容。
      本页剧情：第二页里主角走到走廊，遇到同学。
      关键对白：Hero：咦？
      出场角色：Hero
      角色形态：
      - Hero：Battle Form
      风格约束：保持原作漫画线条、脸型、上色、页面密度和分镜节奏。
    `)

    expect(normalized).toContain('角色形态：')
    expect(normalized).toContain('风格约束：保持原作漫画线条、脸型、上色、页面密度和分镜节奏。')
  })

  it('detects usable story content from continuation-driven fields', () => {
    expect(hasUsableStoryContent({
      continuity_text: '第一页里主角离开教室。',
      story_text: '第二页里主角走到走廊，遇到同学。',
      dialogue_text: 'Hero：咦？',
    })).toBe(true)

    expect(hasUsableStoryContent({
      continuity_text: '',
      story_text: '',
      dialogue_text: '（无）',
    })).toBe(false)
  })
})
