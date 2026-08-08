<script setup lang="ts">
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import StudioPreviewWorkspaceHeader from './StudioPreviewWorkspaceHeader.vue'
import StudioPreviewWorkspacePanel from './StudioPreviewWorkspacePanel.vue'
import type { CharacterStudioChatMessage } from '@/types/characterStudio'

defineProps<{
  latestRuntimeMessage: CharacterStudioChatMessage | null
  summarizeLog: (item: Record<string, unknown>) => string
}>()
</script>

<template>
  <StudioPreviewWorkspacePanel class="runtime-workspace">
    <StudioPreviewWorkspaceHeader
      title="运行日志"
      description="查看最新一轮的变量快照、世界书命中、正则命中与任务执行记录。"
    />
    <div class="runtime-workspace__main">
      <template v-if="latestRuntimeMessage">
        <div class="runtime-workspace__grid">
          <section class="runtime-workspace__card">
            <h5 class="runtime-workspace__card-title">变量快照</h5>
            <pre class="runtime-workspace__card-code">{{ JSON.stringify(latestRuntimeMessage.variables_snapshot || {}, null, 2) }}</pre>
          </section>
          <section class="runtime-workspace__card">
            <h5 class="runtime-workspace__card-title">运行日志</h5>
            <div v-if="latestRuntimeMessage.runtime_log.length > 0" class="runtime-workspace__log-list">
              <div
                v-for="(item, index) in latestRuntimeMessage.runtime_log"
                :key="`runtime-${index}`"
                class="runtime-workspace__log-item"
              >
                {{ summarizeLog(item) }}
              </div>
            </div>
            <ProductEmptyState
              v-else
              icon-name="bar-chart"
              role="note"
              size="compact"
              title="当前还没有运行日志"
            />
          </section>
        </div>
      </template>
      <div v-else class="runtime-workspace__empty-panel">
        <ProductEmptyState
          icon-name="bar-chart"
          role="note"
          size="compact"
          title="发送消息后查看运行结果"
        />
      </div>
    </div>
  </StudioPreviewWorkspacePanel>
</template>

<style scoped>
.runtime-workspace {
  gap: 12px;
  min-height: 0;
}

.runtime-workspace__card-title {
  margin: 8px 0 0;
  color: var(--color-text-heading);
}

.runtime-workspace__main {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  min-height: 0;
}

.runtime-workspace__grid {
  display: grid;
  flex: 1 1 auto;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
  gap: 12px;
  width: 100%;
  min-height: 0;
}

.runtime-workspace__card {
  display: flex;
  flex-direction: column;
  min-height: 0;
  padding: 16px;
  border: 1px solid color-mix(in srgb, var(--color-border-default) 60%, transparent);
  border-radius: 18px;
  background: color-mix(in srgb, var(--color-surface-card) 86%, transparent);
}

.runtime-workspace__card-code {
  flex: 1 1 auto;
  max-height: 280px;
  min-height: 0;
  margin: 10px 0 0;
  overflow: auto;
  color: var(--studio-text-strong);
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}

.runtime-workspace__empty-panel {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
  padding: 12px;
  overflow: auto;
  border: 1px solid var(--studio-border-default);
  border-radius: 20px;
  background: linear-gradient(180deg, color-mix(in srgb, var(--color-surface-app) 95%, transparent), color-mix(in srgb, var(--color-surface-neutral-muted) 90%, transparent));
  align-items: center;
  justify-content: center;
}

.runtime-workspace__log-list {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  gap: 8px;
  min-height: 0;
  margin-top: 10px;
  overflow: auto;
}

.runtime-workspace__log-item {
  padding: 10px 12px;
  border-radius: 12px;
  background: color-mix(in srgb, var(--color-action-brand) 6%, transparent);
  color: var(--studio-text-default);
  font-size: 12px;
  line-height: 1.6;
}

</style>
