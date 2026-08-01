import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import * as insightApi from '@/api/insight'
import { useInsightStore } from '@/stores/insightStore'
import type { TimelineData } from './timelineTypes'
import type { InsightTimelineResponse } from '@/types/insight'

type TimelinePayload = Partial<TimelineData> & {
}

type TimelineApiResponse = InsightTimelineResponse & TimelinePayload

function getTimelinePayload(response: TimelineApiResponse): TimelinePayload {
  return response.timeline && typeof response.timeline === 'object'
    ? response.timeline as TimelinePayload
    : {}
}

function normalizeTimelineResponse(response: TimelineApiResponse): TimelineData {
  const payload = getTimelinePayload(response)
  return {
    mode: payload.mode || 'simple',
    groups: payload.groups || [],
    stats: payload.stats,
    story_summary: payload.story_summary || '',
    main_characters: payload.main_characters || [],
    plot_arcs: payload.plot_arcs || [],
    plot_threads: payload.plot_threads || [],
    events: payload.events || [],
    cached: payload.cached,
  }
}

export function useTimelinePanel() {
  const insightStore = useInsightStore()
  const isLoading = ref(false)
  const isRegenerating = ref(false)
  const timelineData = ref<TimelineData | null>(null)
  const expandedGroups = ref<Set<string>>(new Set())
  const errorMessage = ref('')
  const pendingMessage = ref('')
  let dataRequestId = 0
  let loadRequestId = 0
  let regenerateRequestId = 0
  let isMounted = true

  const hasTimelineData = computed(() => {
    if (!timelineData.value) return false
    const hasGroups = Boolean(timelineData.value.groups?.length)
    const hasArcs = Boolean(timelineData.value.plot_arcs?.length)
    return hasGroups || hasArcs
  })

  const totalEvents = computed(() => timelineData.value?.stats?.total_events || 0)
  const totalPages = computed(() => timelineData.value?.stats?.total_pages || 0)
  const mainCharacters = computed(() => timelineData.value?.main_characters || [])
  const plotArcs = computed(() => timelineData.value?.plot_arcs || [])
  const plotThreads = computed(() => timelineData.value?.plot_threads || [])
  const storySummary = computed(() => timelineData.value?.story_summary || '')
  const expandedGroupIds = computed(() => Array.from(expandedGroups.value))

  const isEnhancedData = computed(() => {
    return timelineData.value?.mode === 'enhanced'
      || Boolean(timelineData.value?.story_summary)
      || Boolean(timelineData.value?.plot_arcs?.length)
  })

  async function loadTimeline(): Promise<void> {
    const bookId = insightStore.currentBookId
    if (!bookId) return

    const requestId = ++dataRequestId
    const loadingId = ++loadRequestId

    isLoading.value = true
    errorMessage.value = ''

    try {
      const response = await insightApi.getTimeline(bookId) as TimelineApiResponse
      if (!isMounted || dataRequestId !== requestId || insightStore.currentBookId !== bookId) return

      if (response.success) {
        timelineData.value = normalizeTimelineResponse(response)
        pendingMessage.value = ''
      } else {
        if (!pendingMessage.value) {
          errorMessage.value = response.error || '加载时间线失败'
        }
      }
    } catch (error) {
      if (!isMounted || dataRequestId !== requestId || insightStore.currentBookId !== bookId) return
      errorMessage.value = error instanceof Error ? error.message : '加载失败'
    } finally {
      if (isMounted && loadRequestId === loadingId) {
        isLoading.value = false
      }
    }
  }

  async function regenerateTimeline(): Promise<void> {
    const bookId = insightStore.currentBookId
    if (!bookId) return

    const requestId = ++dataRequestId
    const regeneratingId = ++regenerateRequestId

    isRegenerating.value = true
    errorMessage.value = ''
    pendingMessage.value = ''

    try {
      const response = await insightApi.regenerateTimeline(bookId) as TimelineApiResponse
      if (!isMounted || dataRequestId !== requestId || insightStore.currentBookId !== bookId) return

      if (response.success) {
        if (response.task_id) {
          timelineData.value = null
          pendingMessage.value = response.message || '时间线生成已进入任务中心，完成后将自动加载。'
        } else {
          timelineData.value = normalizeTimelineResponse(response)
          insightStore.triggerDataRefresh()
        }
      } else {
        errorMessage.value = '重新生成失败'
      }
    } catch (error) {
      if (!isMounted || dataRequestId !== requestId || insightStore.currentBookId !== bookId) return
      errorMessage.value = error instanceof Error ? error.message : '重新生成失败'
    } finally {
      if (isMounted && regenerateRequestId === regeneratingId) {
        isRegenerating.value = false
      }
    }
  }

  function getThumbnailUrl(pageNum: number): string {
    if (!insightStore.currentBookId) return ''
    return insightApi.getThumbnailUrl(insightStore.currentBookId, pageNum)
  }

  function showPageDetail(pageNum: number): void {
    insightStore.selectPage(pageNum)
  }

  function toggleGroup(groupId: string): void {
    const nextGroups = new Set(expandedGroups.value)
    if (nextGroups.has(groupId)) {
      nextGroups.delete(groupId)
    } else {
      nextGroups.add(groupId)
    }
    expandedGroups.value = nextGroups
  }

  onMounted(() => {
    if (insightStore.currentBookId) {
      loadTimeline()
    }
  })

  watch(() => insightStore.currentBookId, (newBookId) => {
    if (newBookId) {
      timelineData.value = null
      expandedGroups.value = new Set()
      pendingMessage.value = ''
      loadTimeline()
    } else {
      dataRequestId++
      timelineData.value = null
      expandedGroups.value = new Set()
      isLoading.value = false
      isRegenerating.value = false
      pendingMessage.value = ''
    }
  })

  watch(() => insightStore.dataRefreshKey, (newKey) => {
    if (newKey > 0 && insightStore.currentBookId) {
      loadTimeline()
    }
  })

  onUnmounted(() => {
    isMounted = false
    dataRequestId += 1
    loadRequestId += 1
    regenerateRequestId += 1
  })

  return {
    errorMessage,
    expandedGroupIds,
    getThumbnailUrl,
    hasTimelineData,
    isEnhancedData,
    isLoading,
    isRegenerating,
    loadTimeline,
    mainCharacters,
    plotArcs,
    plotThreads,
    pendingMessage,
    regenerateTimeline,
    showPageDetail,
    storySummary,
    timelineData,
    toggleGroup,
    totalEvents,
    totalPages,
  }
}
