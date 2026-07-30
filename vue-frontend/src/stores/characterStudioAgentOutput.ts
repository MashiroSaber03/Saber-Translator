import type { CharacterStudioAgentPatchV2 } from '@/types/characterStudio'

export interface CharacterStudioAgentOutput {
  patch: CharacterStudioAgentPatchV2 | null
  htmlPreview: string
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parseAgentPatch(content: string): CharacterStudioAgentPatchV2 | null {
  const match = content.match(/```json:patch\s*([\s\S]*?)```/i)
  if (!match) return null
  try {
    const parsed: unknown = JSON.parse(match[1]!.trim())
    if (isRecord(parsed)) return parsed as CharacterStudioAgentPatchV2
    throw new Error('卡片助手 patch 必须为对象')
  } catch (error) {
    if (error instanceof SyntaxError) return null
    throw error
  }
}

function parseAgentHtmlPreview(content: string): string {
  const match = content.match(/```html\s*([\s\S]*?)```/i)
  return match?.[1]?.trim() || ''
}

export function parseCharacterStudioAgentOutput(content: string): CharacterStudioAgentOutput {
  return {
    patch: parseAgentPatch(content),
    htmlPreview: parseAgentHtmlPreview(content),
  }
}
