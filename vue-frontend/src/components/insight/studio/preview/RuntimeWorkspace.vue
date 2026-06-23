<script setup lang="ts">
import type { CharacterStudioChatMessage } from '@/types/characterStudio'

defineProps<{
  latestRuntimeMessage: CharacterStudioChatMessage | null
  summarizeLog: (item: Record<string, unknown>) => string
}>()
</script>

<template>
  <section class="workspace-card runtime-workspace">
    <div class="assistant-head">
      <div>
        <h4>运行日志</h4>
        <p>查看最新一轮的变量快照、世界书命中、正则命中与任务执行记录。</p>
      </div>
    </div>
    <div class="runtime-main">
      <template v-if="latestRuntimeMessage">
        <div class="runtime-grid">
          <section class="runtime-card">
            <h5>变量快照</h5>
            <pre>{{ JSON.stringify(latestRuntimeMessage.variables_snapshot || {}, null, 2) }}</pre>
          </section>
          <section class="runtime-card">
            <h5>运行日志</h5>
            <div v-if="latestRuntimeMessage.runtime_log.length > 0" class="log-list">
              <div
                v-for="(item, index) in latestRuntimeMessage.runtime_log"
                :key="`runtime-${index}`"
                class="log-item"
              >
                {{ summarizeLog(item) }}
              </div>
            </div>
            <div v-else class="empty-copy">当前还没有运行日志。</div>
          </section>
        </div>
      </template>
      <div v-else class="messages-panel runtime-empty-panel">
        <div class="empty-copy">发送消息后，这里会显示最新一轮的运行结果。</div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.workspace-card {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  width: 100%;
  min-height: 0;
  padding: 14px;
  border: 1px solid var(--studio-border-default);
  border-radius: 24px;
  background: var(--character-studio-preview-shell-surface-raised);
  box-shadow: 0 24px 40px var(--studio-shadow-floating);
}

.runtime-workspace {
  gap: 12px;
  min-height: 0;
}

.assistant-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.assistant-head h4,
.runtime-card h5 {
  margin: 8px 0 0;
  color: var(--character-studio-preview-shell-text-primary);
}

.assistant-head p {
  margin: 8px 0 0;
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.7;
}

.runtime-main {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  min-height: 0;
}

.runtime-grid {
  display: grid;
  flex: 1 1 auto;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  width: 100%;
  min-height: 0;
}

.runtime-card {
  display: flex;
  flex-direction: column;
  min-height: 0;
  padding: 16px;
  border: 1px solid var(--character-studio-preview-workspace-border-default);
  border-radius: 18px;
  background: var(--character-studio-preview-workspace-surface-tint);
}

.runtime-card pre {
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

.messages-panel {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
  padding: 12px;
  overflow: auto;
  border: 1px solid var(--studio-border-default);
  border-radius: 20px;
  background: linear-gradient(180deg, var(--character-studio-preview-workspace-surface-base), var(--character-studio-preview-workspace-surface-raised));
}

.runtime-empty-panel {
  align-items: center;
  justify-content: center;
}

.log-list {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  gap: 8px;
  min-height: 0;
  margin-top: 10px;
  overflow: auto;
}

.log-item {
  padding: 10px 12px;
  border-radius: 12px;
  background: var(--character-studio-preview-details-surface-base);
  color: var(--studio-text-default);
  font-size: 12px;
  line-height: 1.6;
}

.empty-copy {
  color: var(--studio-text-subtle);
  font-size: 13px;
  line-height: 1.7;
}

@media (--breakpoint-studio-down) {
  .runtime-grid {
    grid-template-columns: 1fr;
  }
}
</style>
