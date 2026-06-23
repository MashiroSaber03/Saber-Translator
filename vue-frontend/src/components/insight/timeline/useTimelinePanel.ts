import { computed, onMounted, ref, watch } from 'vue'
import * as insightApi from '@/api/insight'
import { useInsightStore } from '@/stores/insightStore'
import type { TimelineData } from './timelineTypes'

function normalizeTimelineResponse(response: any): TimelineData {
  return {
    mode: response.mode || 'simple',
    groups: response.groups || [],
    stats: response.stats,
    story_summary: response.story_summary || response.summary?.one_sentence || response.summary || '',
    main_characters: response.main_characters || response.characters || [],
    plot_arcs: response.plot_arcs || response.story_arcs || [],
    plot_threads: response.plot_threads || [],
    events: response.events || [],
    cached: response.cached,
  }
}

export function useTimelinePanel() {
  const insightStore = useInsightStore()
  const isLoading = ref(false)
  const isRegenerating = ref(false)
  const timelineData = ref<TimelineData | null>(null)
  const expandedGroups = ref<Set<string>>(new Set())
  const errorMessage = ref('')

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
    if (!insightStore.currentBookId) return

    isLoading.value = true
    errorMessage.value = ''

    try {
      const response = await insightApi.getTimeline(insightStore.currentBookId) as any
      if (response.success) {
        timelineData.value = normalizeTimelineResponse(response)
      } else {
        errorMessage.value = response.error || '加载时间线失败'
      }
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : '加载失败'
    } finally {
      isLoading.value = false
    }
  }

  async function regenerateTimeline(): Promise<void> {
    if (!insightStore.currentBookId) return

    isRegenerating.value = true
    errorMessage.value = ''

    try {
      const response = await insightApi.regenerateTimeline(insightStore.currentBookId) as any
      if (response.success) {
        timelineData.value = normalizeTimelineResponse(response)
        insightStore.triggerDataRefresh()
      } else {
        errorMessage.value = '重新生成失败'
      }
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : '重新生成失败'
    } finally {
      isRegenerating.value = false
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
      loadTimeline()
    }
  })

  watch(() => insightStore.dataRefreshKey, (newKey) => {
    if (newKey > 0 && insightStore.currentBookId) {
      loadTimeline()
    }
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
    regenerateTimeline,
    showPageDetail,
    storySummary,
    timelineData,
    toggleGroup,
    totalEvents,
    totalPages,
  }
}
