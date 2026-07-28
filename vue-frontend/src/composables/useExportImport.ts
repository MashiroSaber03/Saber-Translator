import { computed, getCurrentInstance, onUnmounted, ref } from 'vue'

import { jobsApi, type V2JobDetail } from '@/api/v2/jobs'
import {
  commitChapterTextImport,
  createChapterExportJob,
  getChapterTextExportUrl,
  previewChapterTextImport,
} from '@/api/v2/translation'
import { useImageStore } from '@/stores/imageStore'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import type { ImageData } from '@/types/image'
import { triggerUrlDownload } from '@/utils/browserDownload'
import { useToast } from '@/utils/toast'

export const DOWNLOAD_FORMATS = ['zip', 'pdf', 'cbz'] as const
export type DownloadFormat = (typeof DOWNLOAD_FORMATS)[number]
export type DownloadImageType = 'translated' | 'original'

export interface DownloadImageEntry {
  index: number
  type: DownloadImageType
}

interface JobArtifact {
  assetId: string
  expiresAt: string | null
  kind: string
  url: string
}

type JobWithArtifacts = V2JobDetail & { artifacts?: JobArtifact[] }

export function resolveDownloadFileName(
  originalFileName: string,
  imageIndex: number,
  type: DownloadImageType,
): string {
  const fileName = originalFileName || `image_${imageIndex}.png`
  return `${type}_${fileName.replace(/\.[^/.]+$/, '')}.png`
}

export function collectDownloadImageEntries(images: ImageData[]): DownloadImageEntry[] {
  return images.flatMap((image, index) => {
    if (image.translatedDataURL) return [{ index, type: 'translated' as const }]
    if (image.originalDataURL) return [{ index, type: 'original' as const }]
    return []
  })
}

function downloadUrl(url: string, filename: string): string {
  const separator = url.includes('?') ? '&' : '?'
  return `${url}${separator}download=1&filename=${encodeURIComponent(filename)}`
}

function chapterIdFor(images: ImageData[]): string | null {
  const chapterIds = new Set(
    images.map(image => image.chapterId).filter((value): value is string => Boolean(value)),
  )
  return chapterIds.size === 1 ? [...chapterIds][0] ?? null : null
}

function progressValue(progress: Record<string, unknown>, field: string): number {
  const value = Number(progress[field])
  return Number.isFinite(value) ? value : 0
}

export function useExportImport() {
  const imageStore = useImageStore()
  const taskCenterStore = useTaskCenterStore()
  const toast = useToast()

  const isDownloading = ref(false)
  const downloadProgress = ref(0)
  const downloadProgressText = ref('')
  const isImporting = ref(false)
  const importProgress = ref(0)
  const importProgressText = ref('')
  let importResetTimer: ReturnType<typeof setTimeout> | null = null
  let downloadResetTimer: ReturnType<typeof setTimeout> | null = null

  const canExportText = computed(() => imageStore.hasImages)
  const canImportText = computed(() => imageStore.hasImages)
  const canDownload = computed(() => imageStore.hasImages)

  function resetLater(kind: 'download' | 'import') {
    const current = kind === 'download' ? downloadResetTimer : importResetTimer
    if (current) clearTimeout(current)
    const timer = setTimeout(() => {
      if (kind === 'download') {
        downloadResetTimer = null
        downloadProgress.value = 0
        downloadProgressText.value = ''
      } else {
        importResetTimer = null
        importProgress.value = 0
        importProgressText.value = ''
      }
    }, 2000)
    if (kind === 'download') downloadResetTimer = timer
    else importResetTimer = timer
  }

  function exportText(): void {
    const chapterId = chapterIdFor(imageStore.images)
    if (!chapterId) {
      toast.warning('当前图片不属于同一个后端章节')
      return
    }
    triggerUrlDownload(getChapterTextExportUrl(chapterId))
    toast.success('后端文本导出已开始')
  }

  async function importText(file: File): Promise<void> {
    const chapterId = chapterIdFor(imageStore.images)
    if (!chapterId) {
      toast.warning('当前图片不属于同一个后端章节')
      return
    }
    isImporting.value = true
    importProgress.value = 5
    importProgressText.value = '后端正在校验文本文件'
    try {
      const preview = await previewChapterTextImport(chapterId, file)
      importProgress.value = 50
      importProgressText.value = '正在核对页面与气泡版本'
      const confirmed = preview.pages.filter(page => (
        page.status === 'match'
        && page.changes.length > 0
        && page.baseDocumentRevision !== null
        && page.sourceAssetId
        && page.sourceChecksum
      ))
      if (confirmed.length === 0) {
        if (preview.conflictedPages > 0) {
          toast.warning(`没有可安全导入的页面；${preview.conflictedPages} 页存在版本冲突`)
        } else {
          toast.info('文件内容与当前章节一致，无需导入')
        }
        return
      }
      const accepted = await commitChapterTextImport(chapterId, confirmed)
      await taskCenterStore.refresh()
      importProgress.value = 100
      importProgressText.value = '文本导入任务已进入后端队列'
      const conflictSuffix = preview.conflictedPages > 0
        ? `；跳过 ${preview.conflictedPages} 页冲突`
        : ''
      toast.success(
        `已提交 ${confirmed.length} 页文本导入，可安全关闭页面${conflictSuffix}`,
      )
      if (!accepted.jobIds[0]) throw new Error('后端没有返回文本导入任务')
    } catch (error) {
      toast.error(`导入失败：${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isImporting.value = false
      resetLater('import')
    }
  }

  function downloadCurrentImage(): void {
    const image = imageStore.currentImage
    const assetUrl = image?.translatedAssetUrl
      || image?.sourceAssetUrl
      || image?.translatedDataURL
      || image?.originalDataURL
    if (!image || !assetUrl) {
      toast.warning('没有可下载的图片')
      return
    }
    const type: DownloadImageType = (
      image.translatedAssetUrl || image.translatedDataURL
    ) ? 'translated' : 'original'
    const filename = resolveDownloadFileName(
      image.fileName,
      imageStore.currentImageIndex,
      type,
    )
    triggerUrlDownload(downloadUrl(assetUrl, filename))
    toast.success(`下载已开始：${filename}`)
  }

  async function waitForExport(jobId: string): Promise<JobWithArtifacts> {
    for (let attempt = 0; attempt < 3600; attempt += 1) {
      const job = await jobsApi.get(jobId) as JobWithArtifacts
      const total = progressValue(job.progress, 'totalItems')
      const complete = (
        progressValue(job.progress, 'completedItems')
        + progressValue(job.progress, 'failedItems')
      )
      downloadProgress.value = total > 0
        ? Math.min(95, Math.round(complete / total * 90) + 5)
        : 5
      downloadProgressText.value = `后端正在生成导出文件：${complete}/${total || 1}`
      if (job.status === 'completed') return job
      if (['cancelled', 'completed_with_errors', 'failed'].includes(job.status)) {
        throw new Error(`导出任务状态：${job.status}`)
      }
      await new Promise(resolve => setTimeout(resolve, 500))
    }
    throw new Error('导出仍在后端运行，请稍后从任务中心下载')
  }

  async function downloadAllImages(format: DownloadFormat = 'zip'): Promise<void> {
    const chapterId = chapterIdFor(imageStore.images)
    if (!chapterId) {
      toast.warning('当前图片不属于同一个后端章节')
      return
    }
    isDownloading.value = true
    downloadProgress.value = 2
    downloadProgressText.value = '正在创建后端导出任务'
    try {
      const accepted = await createChapterExportJob(
        chapterId,
        format,
        imageStore.images.map(image => image.id),
      )
      await taskCenterStore.refresh()
      const jobId = accepted.jobIds[0]
      if (!jobId) throw new Error('后端没有返回导出任务')
      toast.info('导出任务已进入后端队列，可安全关闭页面', 0)
      const job = await waitForExport(jobId)
      const artifact = job.artifacts?.[0]
      if (!artifact) throw new Error('导出任务未生成可下载文件')
      downloadProgress.value = 100
      downloadProgressText.value = '导出完成，下载已开始'
      triggerUrlDownload(
        downloadUrl(artifact.url, `chapter-export.${format}`),
      )
      toast.success('后端导出完成，下载已开始')
    } catch (error) {
      toast.error(`下载失败：${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isDownloading.value = false
      resetLater('download')
    }
  }

  if (getCurrentInstance()) {
    onUnmounted(() => {
      if (importResetTimer) clearTimeout(importResetTimer)
      if (downloadResetTimer) clearTimeout(downloadResetTimer)
    })
  }

  return {
    canDownload,
    canExportText,
    canImportText,
    downloadAllImages,
    downloadCurrentImage,
    downloadProgress,
    downloadProgressText,
    exportText,
    importProgress,
    importProgressText,
    importText,
    isDownloading,
    isImporting,
  }
}
