import {
  downloadCharacterStudioExport,
  downloadCharacterStudioWorldbook,
  exportCharacterStudioChatSession,
} from '@/api/characterStudio'
import { triggerBlobDownload } from '@/utils/browserDownload'

export async function downloadStudioChatTranscript(
  bookId: string,
  docId: string,
  sessionId: string,
): Promise<void> {
  const { blob, filename } = await exportCharacterStudioChatSession(bookId, docId, sessionId)
  triggerBlobDownload(blob, filename)
}

export async function downloadStudioDocumentExport(
  bookId: string,
  docId: string,
  format: string,
): Promise<void> {
  const { blob, filename } = format === 'worldbook'
    ? await downloadCharacterStudioWorldbook(bookId, docId)
    : await downloadCharacterStudioExport(bookId, docId, format)
  triggerBlobDownload(blob, filename)
}
