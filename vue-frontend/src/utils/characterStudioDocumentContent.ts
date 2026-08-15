import type { V2StudioDocumentContent } from '@/api/v2/studio'
import type { CharacterStudioDocument } from '@/types/characterStudio'

export function characterStudioDocumentContent(
  document: CharacterStudioDocument,
): V2StudioDocumentContent {
  return {
    origin: document.origin,
    status: document.status,
    meta: {
      title: document.meta.title,
      tags: document.meta.tags,
    },
    identity: document.identity,
    coreMessages: document.coreMessages,
    lorebook: document.lorebook,
    regexScripts: document.regexScripts,
    stateTasks: document.stateTasks,
    exportArtifacts: document.exportArtifacts,
  }
}
