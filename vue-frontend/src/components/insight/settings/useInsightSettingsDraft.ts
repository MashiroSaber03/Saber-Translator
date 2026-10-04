import { watch, type WatchSource } from 'vue'

type InsightSettingsDraftOptions<TConfig> = {
  sources: WatchSource<unknown>[]
  buildDraft: () => TConfig
  emitDraft: (config: TConfig) => void
  deep?: boolean
}

export function useInsightSettingsDraft<TConfig>(options: InsightSettingsDraftOptions<TConfig>) {
  watch(options.sources, () => options.emitDraft(options.buildDraft()), {
    deep: options.deep,
    immediate: true,
  })
}
