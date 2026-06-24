<script setup lang="ts">
import type { PlotThread } from './timelineTypes'

defineProps<{
  plotThreads: PlotThread[]
  storySummary: string
}>()
</script>

<template>
  <div class="timeline-summary-card">
    <h4>📖 故事概要</h4>
    <p class="one-sentence">{{ storySummary }}</p>
    <div v-if="plotThreads.length" class="themes">
      <span>主题：</span>
      <span
        v-for="thread in plotThreads.slice(0, 5)"
        :key="thread.id"
        class="theme-tag"
      >
        {{ thread.name }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.timeline-summary-card {
  background: linear-gradient(135deg, var(--insight-action-primary) 0%, var(--insight-action-primary-strong) 100%);
  color: var(--color-text-inverse);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 24px;
}

.timeline-summary-card h4 {
  margin: 0 0 12px;
  font-size: 16px;
  font-weight: 600;
}

.one-sentence {
  font-size: 15px;
  line-height: 1.6;
  margin-bottom: 12px;
}

.themes {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.theme-tag {
  background: var(--timeline-panel-summary-tag-surface);
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 12px;
}
</style>
