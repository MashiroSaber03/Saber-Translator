import { getCurrentInstance, onUnmounted, ref } from 'vue'

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
export type DownloadImageType = 'translated' | 'clean' | 'original'

export function resolveDownloadFileName(
  originalFileName: string,
  imageIndex: number,
  type: DownloadImageType,
): string {
  const fileName = originalFileName || `image_${imageIndex}.png`
  return `${type}_${fileName.replace(/\.[^/.]+$/, '')}.png`
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
  let downloadResetTimer: ReturnType<typeof setTimeout> | null = null
  let disposed = false
  let activeExportWait: AbortController | null = null

  function resetDownloadProgressLater() {
    if (downloadResetTimer) clearTimeout(downloadResetTimer)
    downloadResetTimer = setTimeout(() => {
      downloadResetTimer = null
      downloadProgress.value = 0
      downloadProgressText.value = ''
    }, 2000)
  }

  function exportText(): void {
    const chapterId = chapterIdFor(imageStore.images)
    if (!chapterId) {
      toast.warning('当前图片不属于同一个后端章节')
      return
    }
    triggerUrlDownload(
      getChapterTextExportUrl(chapterId),
      `chapter-${chapterId}-text.json`,
    )
    toast.success('后端文本导出已开始')
  }

  async function importText(file: File): Promise<void> {
    const chapterId = chapterIdFor(imageStore.images)
    if (!chapterId) {
      toast.warning('当前图片不属于同一个后端章节')
      return
    }
    try {
      const preview = await previewChapterTextImport(chapterId, file)
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
      const conflictSuffix = preview.conflictedPages > 0
        ? `；跳过 ${preview.conflictedPages} 页冲突`
        : ''
      toast.success(
        `已提交 ${confirmed.length} 页文本导入，可安全关闭页面${conflictSuffix}`,
      )
      if (!accepted.jobIds[0]) throw new Error('后端没有返回文本导入任务')
    } catch (error) {
      toast.error(`导入失败：${error instanceof Error ? error.message : String(error)}`)
    }
  }

  function downloadCurrentImage(): void {
    const image = imageStore.currentImage
    const assetUrl = (
      image?.translatedAssetUrl
      || image?.cleanAssetUrl
      || image?.sourceAssetUrl
    )
    if (!image || !assetUrl) {
      toast.warning('没有可下载的图片')
      return
    }
    const type: DownloadImageType = image.translatedAssetUrl
      ? 'translated'
      : image.cleanAssetUrl
        ? 'clean'
        : 'original'
    const filename = resolveDownloadFileName(
      image.fileName,
      imageStore.currentImageIndex,
      type,
    )
    triggerUrlDownload(downloadUrl(assetUrl, filename))
    toast.success(`下载已开始：${filename}`)
  }

  async function waitForExport(jobId: string) {
    activeExportWait?.abort()
    const controller = new AbortController()
    activeExportWait = controller
    try {
      return await taskCenterStore.waitForJob(jobId, {
        signal: controller.signal,
        onProgress(progress) {
          const total = progressValue(progress, 'totalItems')
          const complete = (
            progressValue(progress, 'completedItems')
            + progressValue(progress, 'failedItems')
          )
          downloadProgress.value = total > 0
            ? Math.min(95, Math.round(complete / total * 90) + 5)
            : 5
          downloadProgressText.value = `后端正在生成导出文件：${complete}/${total || 1}`
        },
      })
    } finally {
      if (activeExportWait === controller) activeExportWait = null
    }
  }

  async function downloadAllImages(format: DownloadFormat = 'zip'): Promise<void> {
    if (isDownloading.value) {
      toast.info('已有导出任务正在等待后端完成')
      return
    }
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
      const jobId = accepted.jobIds[0]
      if (!jobId) throw new Error('后端没有返回导出任务')
      toast.info('导出任务已进入后端队列，可安全关闭页面', 0)
      const job = await waitForExport(jobId)
      const artifact = job.artifacts[0]
      if (!artifact) throw new Error('导出任务未生成可下载文件')
      downloadProgress.value = 100
      downloadProgressText.value = '导出完成，下载已开始'
      triggerUrlDownload(
        downloadUrl(artifact.url, `chapter-export.${format}`),
      )
      if (job.status === 'completed_with_errors') {
        toast.warning('后端导出已完成，但有部分页面失败；已下载可用结果')
      } else {
        toast.success('后端导出完成，下载已开始')
      }
    } catch (error) {
      if (!disposed) {
        toast.error(`下载失败：${error instanceof Error ? error.message : String(error)}`)
      }
    } finally {
      isDownloading.value = false
      resetDownloadProgressLater()
    }
  }

  if (getCurrentInstance()) {
    onUnmounted(() => {
      disposed = true
      activeExportWait?.abort()
      activeExportWait = null
      if (downloadResetTimer) clearTimeout(downloadResetTimer)
    })
  }

  return {
    downloadAllImages,
    downloadCurrentImage,
    downloadProgress,
    downloadProgressText,
    exportText,
    importText,
    isDownloading,
  }
}
