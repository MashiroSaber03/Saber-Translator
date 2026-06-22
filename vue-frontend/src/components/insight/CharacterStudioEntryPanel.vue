<template>
  <div class="entry-card">
    <div>
      <div class="eyebrow">角色工坊 2.0</div>
      <h3>角色工坊已升级为独立工作台</h3>
      <p>新的工作台会在独立页面中提供角色候选、世界书树、问候语、正则脚本、状态任务、聊天预览和卡片助手的完整闭环体验。</p>
    </div>
    <div class="actions">
      <UiButton variant="toolbar" class="action-primary" @click="openStudio">打开角色工坊</UiButton>
    </div>
  </div>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import { useRouter } from 'vue-router'
import { useInsightStore } from '@/stores/insightStore'

const router = useRouter()
const insightStore = useInsightStore()

function openStudio() {
  if (!insightStore.currentBookId) return
  void router.push({
    name: 'character-studio',
    query: { book: insightStore.currentBookId },
  })
}
</script>

<style scoped>
.entry-card {
  /* owner tokens: character-studio-entry-panel */
  --character-studio-entry-panel-accent-primary: rgba(86, 138, 225, .08);
  --character-studio-entry-panel-accent-secondary: rgba(79, 136, 240, .12);
  --character-studio-entry-panel-accent-muted: rgba(255, 255, 255, 0);
  --character-studio-entry-panel-border-default: rgba(36, 76, 130, .14);
  --character-studio-entry-panel-shadow-default: rgba(25, 49, 80, .1);
  --character-studio-entry-panel-shadow-raised: rgba(38, 91, 184, .24);
  --character-studio-entry-panel-surface-base: #2960c1;
  --character-studio-entry-panel-surface-raised: #447fe5;
  --character-studio-entry-panel-text-primary: #5778a4;
  --character-studio-entry-panel-text-secondary: #566d86;

  width: 100%;
  min-height: 180px;
  padding: 28px 30px;
  border-radius: 28px;
  background:
    radial-gradient(circle at top right, var(--character-studio-entry-panel-accent-primary), transparent 24%),
    linear-gradient(180deg, var(--character-studio-entry-panel-accent-secondary), var(--character-studio-entry-panel-accent-muted));
  border: 1px solid var(--character-studio-entry-panel-border-default);
  box-shadow: 0 24px 48px var(--character-studio-entry-panel-shadow-default);
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: center;
}

.eyebrow {
  font-size: 11px;
  letter-spacing: 0.12em;
  color: var(--character-studio-entry-panel-text-primary);
  font-weight: 600;
}

.entry-card h3 {
  margin: 10px 0 0;
  font-size: 24px;
}

.entry-card p {
  margin: 12px 0 0;
  color: var(--character-studio-entry-panel-text-secondary);
  max-width: 760px;
  line-height: 1.7;
}

.action-primary {
  border: none;
  border-radius: 14px;
  padding: 12px 20px;
  font-size: 14px;
  cursor: pointer;
  color: var(--color-text-inverse);
  background: linear-gradient(135deg, var(--character-studio-entry-panel-surface-base), var(--character-studio-entry-panel-surface-raised));
  box-shadow: 0 10px 22px var(--character-studio-entry-panel-shadow-raised);
}

@media (--breakpoint-lg-down) {
  .entry-card {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
