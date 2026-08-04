import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import * as insightApi from '@/api/insight'
import { useInsightStore } from '@/stores/insightStore'
import type { TimelineData } from '@/types/insight'

function normalizeTimeline(data: TimelineData): TimelineData {
  return {
    mode: data.mode || 'simple',
    groups: data.groups || [],
    stats: data.stats,
    story_summary: data.story_summary || '',
    main_characters: data.main_characters || [],
    plot_arcs: data.plot_arcs || [],
    plot_threads: data.plot_threads || [],
    events: data.events || [],
    cached: data.cached,
    next_event_cursor: data.next_event_cursor,
    next_character_cursor: data.next_character_cursor,
  }
}

export function useTimelinePanel() {
  const insightStore = useInsightStore()
  const isLoading = ref(false)
  const isLoadingMore = ref(false)
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
  const hasMoreTimeline = computed(() => Boolean(
    timelineData.value?.next_event_cursor != null
    || timelineData.value?.next_character_cursor != null
  ))

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
      const timeline = await insightApi.getTimeline(bookId)
      if (!isMounted || dataRequestId !== requestId || insightStore.currentBookId !== bookId) return

      if (timeline) {
        timelineData.value = normalizeTimeline(timeline)
        pendingMessage.value = ''
      } else if (!pendingMessage.value) {
        timelineData.value = null
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
      await insightApi.regenerateTimeline(bookId)
      if (!isMounted || dataRequestId !== requestId || insightStore.currentBookId !== bookId) return

      timelineData.value = null
      pendingMessage.value = '时间线生成已进入任务中心，完成后将自动加载。'
    } catch (error) {
      if (!isMounted || dataRequestId !== requestId || insightStore.currentBookId !== bookId) return
      errorMessage.value = error instanceof Error ? error.message : '重新生成失败'
    } finally {
      if (isMounted && regenerateRequestId === regeneratingId) {
        isRegenerating.value = false
      }
    }
  }

  async function loadMoreTimeline(): Promise<void> {
    const bookId = insightStore.currentBookId
    const current = timelineData.value
    if (!bookId || !current || !hasMoreTimeline.value || isLoadingMore.value) return
    const requestId = dataRequestId
    isLoadingMore.value = true
    errorMessage.value = ''
    try {
      const next = await insightApi.getTimeline(bookId, {
        eventCursor: current.next_event_cursor ?? current.stats?.total_events ?? 0,
        characterCursor: current.next_character_cursor
          ?? current.main_characters?.at(-1)?.name,
      })
      if (!next || !isMounted || dataRequestId !== requestId || insightStore.currentBookId !== bookId) {
        return
      }
      const knownGroups = new Set(current.groups?.map(group => group.id))
      const knownCharacters = new Set(current.main_characters?.map(character => character.name))
      timelineData.value = normalizeTimeline({
        ...current,
        groups: [
          ...(current.groups ?? []),
          ...(next.groups ?? []).filter(group => !knownGroups.has(group.id)),
        ],
        events: [...(current.events ?? []), ...(next.events ?? [])],
        main_characters: [
          ...(current.main_characters ?? []),
          ...(next.main_characters ?? []).filter(character => !knownCharacters.has(character.name)),
        ],
        stats: next.stats ?? current.stats,
        next_event_cursor: next.next_event_cursor,
        next_character_cursor: next.next_character_cursor,
      })
    } catch (error) {
      if (isMounted && dataRequestId === requestId) {
        errorMessage.value = error instanceof Error ? error.message : '加载更多时间线失败'
      }
    } finally {
      if (isMounted && dataRequestId === requestId) isLoadingMore.value = false
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
      isLoadingMore.value = false
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
    isLoadingMore,
    isRegenerating,
    loadTimeline,
    loadMoreTimeline,
    hasMoreTimeline,
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
