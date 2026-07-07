import type { CharacterStudioAgentPatchV2 } from '@/types/characterStudio'

export interface CharacterStudioAgentOutput {
  patch: CharacterStudioAgentPatchV2 | null
  htmlPreview: string
}

function parseAgentPatch(content: string): CharacterStudioAgentPatchV2 | null {
  const match = content.match(/```json:patch\s*([\s\S]*?)```/i)
  if (!match) return null
  try {
    return JSON.parse(match[1]!.trim()) as CharacterStudioAgentPatchV2
  } catch {
    return null
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
