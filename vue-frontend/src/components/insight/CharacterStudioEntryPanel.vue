<template>
  <div class="entry-card">
    <div>
      <div class="eyebrow">角色工坊</div>
      <h3>独立角色工作台</h3>
      <p>在独立页面中集中管理角色候选、世界书树、问候语、正则脚本、状态任务、聊天预览和卡片助手。</p>
    </div>
    <div class="actions">
      <UiButton variant="toolbar" class="entry-card__action" @click="openStudio">打开角色工坊</UiButton>
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
  --character-studio-entry-panel-card-background: radial-gradient(circle at top right, rgba(86, 138, 225, .08), transparent 24%), linear-gradient(180deg, rgba(79, 136, 240, .12), rgba(255, 255, 255, 0));
  --character-studio-entry-panel-card-border: rgba(36, 76, 130, .14);
  --character-studio-entry-panel-card-shadow: rgba(25, 49, 80, .1);
  --character-studio-entry-panel-action-background: linear-gradient(135deg, #2960c1, #447fe5);
  --character-studio-entry-panel-action-shadow: rgba(38, 91, 184, .24);
  --character-studio-entry-panel-eyebrow-text: #5778a4;
  --character-studio-entry-panel-description-text: #566d86;

  width: 100%;
  min-height: 180px;
  padding: 28px 30px;
  border-radius: 28px;
  background: var(--character-studio-entry-panel-card-background);
  border: 1px solid var(--character-studio-entry-panel-card-border);
  box-shadow: 0 24px 48px var(--character-studio-entry-panel-card-shadow);
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: center;
}

.eyebrow {
  font-size: 11px;
  letter-spacing: 0;
  color: var(--character-studio-entry-panel-eyebrow-text);
  font-weight: 600;
}

.entry-card h3 {
  margin: 10px 0 0;
  font-size: 24px;
}

.entry-card p {
  margin: 12px 0 0;
  color: var(--character-studio-entry-panel-description-text);
  max-width: 760px;
  line-height: 1.7;
}

.entry-card__action {
  border: none;
  border-radius: 14px;
  padding: 12px 20px;
  font-size: 14px;
  cursor: pointer;
  color: var(--color-text-inverse);
  background: var(--character-studio-entry-panel-action-background);
  box-shadow: 0 10px 22px var(--character-studio-entry-panel-action-shadow);
}

@media (--breakpoint-lg-down) {
  .entry-card {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
