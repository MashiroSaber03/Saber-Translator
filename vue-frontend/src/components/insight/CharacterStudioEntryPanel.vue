<template>
  <div class="entry-card">
    <div>
      <div class="eyebrow">角色工坊 2.0</div>
      <h3>角色工坊已升级为独立工作台</h3>
      <p>新的工作台会在独立页面中提供角色候选、世界书树、问候语、正则脚本、状态任务、聊天预览和卡片助手的完整闭环体验。</p>
    </div>
    <div class="actions">
      <button class="primary-btn" @click="openStudio">打开角色工坊</button>
    </div>
  </div>
</template>

<script setup lang="ts">
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
  margin: 20px;
  padding: 28px 30px;
  border-radius: 28px;
  background:
    radial-gradient(circle at top right, rgba(79, 136, 240, 0.12), transparent 24%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(246, 249, 254, 0.88));
  border: 1px solid rgba(36, 76, 130, 0.14);
  box-shadow: 0 24px 48px rgba(25, 49, 80, 0.1);
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: center;
}

.eyebrow {
  font-size: 11px;
  letter-spacing: 0.12em;
  color: #5778a4;
  font-weight: 600;
}

.entry-card h3 {
  margin: 10px 0 0;
  font-size: 24px;
}

.entry-card p {
  margin: 12px 0 0;
  color: #566d86;
  max-width: 760px;
  line-height: 1.7;
}

.primary-btn {
  border: none;
  border-radius: 14px;
  padding: 12px 20px;
  font-size: 14px;
  cursor: pointer;
  color: #fff;
  background: linear-gradient(135deg, #2960c1, #447fe5);
  box-shadow: 0 10px 22px rgba(38, 91, 184, 0.24);
}

@media (max-width: 900px) {
  .entry-card {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
