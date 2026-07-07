import { watch, type WatchSource } from 'vue'

type InsightSettingsDraftOptions<TConfig> = {
  sources: WatchSource<unknown>[]
  buildDraft: () => TConfig
  applyDraft: (config: TConfig) => void
  loadDraft: () => TConfig
  emitDraft: (config: TConfig) => void
  syncRequestId?: WatchSource<number | undefined>
  deep?: boolean
}

export function useInsightSettingsDraft<TConfig>(options: InsightSettingsDraftOptions<TConfig>) {
  function emitCurrentDraft(): void {
    options.emitDraft(options.buildDraft())
  }

  function syncFromSource(): void {
    options.applyDraft(options.loadDraft())
    emitCurrentDraft()
  }

  watch(options.sources, emitCurrentDraft, {
    deep: options.deep,
    immediate: true,
  })

  if (options.syncRequestId) {
    watch(options.syncRequestId, syncFromSource)
  }

  return {
    emitCurrentDraft,
    syncFromSource,
  }
}
