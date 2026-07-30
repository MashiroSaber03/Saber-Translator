import { describe, expect, it } from 'vitest'
import { parseCharacterStudioAgentOutput } from '@/stores/characterStudioAgentOutput'

describe('character studio agent output', () => {
  it('keeps the native domain patch format', () => {
    const result = parseCharacterStudioAgentOutput(
      '```json:patch\n{"set":{"identity.description":"新的简介"}}\n```',
    )

    expect(result.patch).toEqual({
      set: { 'identity.description': '新的简介' },
    })
  })

  it('rejects alternate patch dialects instead of maintaining a compatibility layer', () => {
    expect(() => parseCharacterStudioAgentOutput(
      '```json:patch\n'
      + '[{"op":"replace","path":"/identity/description","value":"新的简介"}]\n'
      + '```',
    )).toThrow('卡片助手 patch 必须为对象')
  })
})
