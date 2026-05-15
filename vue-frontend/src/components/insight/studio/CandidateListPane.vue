<template>
  <div class="pane">
    <div class="pane-head">
      <h3>分析候选</h3>
      <span>{{ candidates.length }}</span>
    </div>
    <div v-if="!hasTimeline" class="empty-copy">当前书还没有增强时间线。你仍然可以空白新建或导入角色卡。</div>
    <div v-else-if="candidates.length === 0" class="empty-copy">没有可用候选角色。</div>
    <div v-else class="list">
      <div v-for="item in candidates" :key="item.name" class="candidate-row">
        <div class="candidate-main">
          <strong>{{ item.name }}</strong>
          <div class="candidate-meta">
            首登 {{ item.first_appearance || '-' }} 页 · 对话 {{ item.dialogue_count }} · 关键页 {{ item.sample_pages.slice(0, 3).join(' / ') || '-' }}
          </div>
        </div>
        <button class="create-btn" :disabled="!!creatingCandidateName" @click="$emit('create', item.name)">
          {{ creatingCandidateName === item.name ? '创建中...' : '创建' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { CharacterStudioCandidate } from '@/types/characterStudio'

defineProps<{
  candidates: CharacterStudioCandidate[]
  hasTimeline: boolean
  creatingCandidateName: string
}>()

defineEmits<{
  (e: 'create', candidateName: string): void
}>()
</script>

<style scoped>
.pane {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.pane-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.pane-head h3 {
  margin: 0;
  font-size: 14px;
}

.pane-head span {
  font-size: 12px;
  color: #6d839f;
}

.empty-copy {
  color: #6d839f;
  font-size: 13px;
  line-height: 1.6;
  padding: 8px 0;
}

.list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.candidate-row {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: center;
  border-radius: 16px;
  padding: 12px;
  background: rgba(255, 255, 255, 0.72);
}

.candidate-main strong {
  display: block;
  color: #122b47;
  font-size: 13px;
}

.candidate-meta {
  margin-top: 6px;
  color: #6d839f;
  font-size: 11px;
  line-height: 1.5;
}

.create-btn {
  border: none;
  border-radius: 12px;
  padding: 8px 12px;
  background: rgba(37, 99, 199, 0.1);
  color: #1f5fc3;
  cursor: pointer;
}

.create-btn:disabled {
  opacity: 0.72;
  cursor: wait;
}
</style>
