import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import * as insightApi from '@/api/insight'
import { useInsightStore } from '@/stores/insightStore'
import type { TimelineData } from '@/types/insight'

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
    return timelineData.value !== null
  })

  const mainCharacters = computed(() => timelineData.value?.main_characters ?? [])
  const plotArcs = computed(() => timelineData.value?.plot_arcs ?? [])
  const plotThreads = computed(() => timelineData.value?.plot_threads ?? [])
  const storySummary = computed(() => timelineData.value?.story_summary ?? '')
  const expandedGroupIds = computed(() => Array.from(expandedGroups.value))
  const hasMoreTimeline = computed(() =>
    Boolean(
      timelineData.value?.next_event_cursor != null ||
      timelineData.value?.next_character_cursor != null
    )
  )

  const isEnhancedData = computed(() => {
    return timelineData.value?.mode === 'enhanced'
  })

  async function loadTimeline(): Promise<void> {
    const bookId = insightStore.currentBookId
    if (!bookId) return

    const requestId = ++dataRequestId
    const loadingId = ++loadRequestId

    isLoading.value = true
    isLoadingMore.value = false
    errorMessage.value = ''

    try {
      const timeline = await insightApi.getTimeline(bookId)
      if (!isMounted || dataRequestId !== requestId || insightStore.currentBookId !== bookId) return

      if (timeline) {
        if (
          timelineData.value &&
          timelineData.value.timeline_version_id !== timeline.timeline_version_id
        ) {
          expandedGroups.value = new Set()
        }
        timelineData.value = timeline
        pendingMessage.value = ''
      } else if (!pendingMessage.value) {
        timelineData.value = null
      }
    } catch (error) {
      if (!isMounted || dataRequestId !== requestId || insightStore.currentBookId !== bookId) return
      errorMessage.value = error instanceof Error ? error.message : '加载失败'
    } finally {
      if (isMounted && loadRequestId === loadingId && insightStore.currentBookId === bookId) {
        isLoading.value = false
      }
    }
  }

  async function regenerateTimeline(): Promise<void> {
    const bookId = insightStore.currentBookId
    if (!bookId || isRegenerating.value || pendingMessage.value) return

    const requestId = ++dataRequestId
    const regeneratingId = ++regenerateRequestId

    isRegenerating.value = true
    errorMessage.value = ''
    pendingMessage.value = ''

    try {
      await insightApi.regenerateTimeline(bookId)
      if (!isMounted || dataRequestId !== requestId || insightStore.currentBookId !== bookId) return

      pendingMessage.value = '时间线生成已进入任务中心，完成后将自动加载。'
    } catch (error) {
      if (!isMounted || dataRequestId !== requestId || insightStore.currentBookId !== bookId) return
      errorMessage.value = error instanceof Error ? error.message : '重新生成失败'
    } finally {
      if (
        isMounted &&
        regenerateRequestId === regeneratingId &&
        insightStore.currentBookId === bookId
      ) {
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
        eventCursor: current.next_event_cursor ?? undefined,
        characterCursor: current.next_character_cursor ?? undefined,
      })
      if (
        !next ||
        !isMounted ||
        dataRequestId !== requestId ||
        insightStore.currentBookId !== bookId
      ) {
        return
      }
      if (next.timeline_version_id !== current.timeline_version_id) {
        timelineData.value = next
        expandedGroups.value = new Set()
        return
      }
      const knownGroups = new Set(current.groups.map(group => group.id))
      const knownEvents = new Set(current.events.map(event => event.eventId))
      const knownCharacters = new Set(
        current.main_characters.map(character => character.character_id)
      )
      timelineData.value = {
        ...current,
        groups: [...current.groups, ...next.groups.filter(group => !knownGroups.has(group.id))],
        events: [
          ...current.events,
          ...next.events.filter(event => !knownEvents.has(event.eventId)),
        ],
        main_characters: [
          ...current.main_characters,
          ...next.main_characters.filter(character => !knownCharacters.has(character.character_id)),
        ],
        page_thumbnails: { ...current.page_thumbnails, ...next.page_thumbnails },
        stats: next.stats,
        next_event_cursor: next.next_event_cursor,
        next_character_cursor: next.next_character_cursor,
      }
    } catch (error) {
      if (isMounted && dataRequestId === requestId && insightStore.currentBookId === bookId) {
        errorMessage.value = error instanceof Error ? error.message : '加载更多时间线失败'
      }
    } finally {
      if (isMounted && dataRequestId === requestId && insightStore.currentBookId === bookId) {
        isLoadingMore.value = false
      }
    }
  }

  function getThumbnailUrl(pageNum: number): string {
    return timelineData.value?.page_thumbnails[pageNum] ?? ''
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

  watch(
    () => insightStore.currentBookId,
    newBookId => {
      dataRequestId += 1
      loadRequestId += 1
      regenerateRequestId += 1
      timelineData.value = null
      expandedGroups.value = new Set()
      isLoading.value = false
      isRegenerating.value = false
      isLoadingMore.value = false
      errorMessage.value = ''
      pendingMessage.value = ''
      if (newBookId) loadTimeline()
    }
  )

  watch(
    () => insightStore.dataRefreshKey,
    newKey => {
      if (newKey > 0 && insightStore.currentBookId) {
        loadTimeline()
      }
    }
  )

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
  }
}
