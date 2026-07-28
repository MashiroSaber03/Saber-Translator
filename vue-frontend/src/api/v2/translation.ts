import { apiClient } from '@/api/client'
import { newIdempotencyKey } from './content'

export interface V2TranslationBatchAccepted {
  batchId: string
  jobIds: string[]
  status: 'queued'
}

export interface TranslationJobConfig {
  executionMode: 'parallel' | 'sequential'
  mode: 'hq' | 'proofread' | 'remove_text' | 'standard'
  skipCompleted?: boolean
  sourceLanguage: string
  targetLanguage: string
}

export async function createChapterTranslationJob(
  chapterId: string,
  pageIds: string[],
  config: TranslationJobConfig,
): Promise<V2TranslationBatchAccepted> {
  return apiClient.post<V2TranslationBatchAccepted>(
    `/api/v2/chapters/${encodeURIComponent(chapterId)}/translation-jobs`,
    { config, pageIds },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}
