import {
  downloadCharacterStudioExport,
  downloadCharacterStudioWorldbook,
  exportCharacterStudioChatSession,
} from '@/api/characterStudio'
import { triggerBlobDownload } from '@/utils/browserDownload'

export async function downloadStudioChatTranscript(
  sessionId: string,
): Promise<void> {
  const { blob, filename } = await exportCharacterStudioChatSession(sessionId)
  triggerBlobDownload(blob, filename)
}

export async function downloadStudioDocumentExport(
  docId: string,
  format: string,
): Promise<void> {
  const { blob, filename } = format === 'worldbook'
    ? await downloadCharacterStudioWorldbook(docId)
    : await downloadCharacterStudioExport(docId, format)
  triggerBlobDownload(blob, filename)
}
