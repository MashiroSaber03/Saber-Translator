<template>
  <div class="workbench">
    <div class="hero-block">
      <div class="hero-head">
        <div>
          <h3>主问候</h3>
          <p>角色进入对话时最先展示的开场白。它决定了语气、场景和第一印象。</p>
        </div>
        <button class="ghost-btn" :disabled="generating" @click="$emit('generate')">
          {{ generating ? '生成中...' : '批量生成' }}
        </button>
      </div>
      <textarea :value="firstMessage" rows="6" @input="$emit('update:firstMessage', ($event.target as HTMLTextAreaElement).value)"></textarea>
    </div>

    <div class="list-block">
      <div class="list-head">
        <div>
          <h3>备用问候</h3>
          <p>维护多种开场方式，可随时采用为主问候或继续打磨。</p>
        </div>
        <button class="secondary-btn" @click="$emit('add')">添加备用问候</button>
      </div>

      <div v-if="alternates.length === 0" class="empty-copy">还没有备用问候，建议生成 3-5 条不同场景的开场白。</div>

      <div v-else class="alternate-list">
        <article v-for="(item, index) in alternates" :key="`alt-${index}`" class="alternate-card">
          <div class="alternate-head">
            <div class="title">
              <span class="index-chip">#{{ index + 1 }}</span>
              <strong>备用问候</strong>
            </div>
            <div class="actions">
              <button class="ghost-btn small" @click="$emit('promote', item)">设为主问候</button>
              <button class="ghost-btn small" :disabled="index === 0" @click="$emit('move', index, -1)">上移</button>
              <button class="ghost-btn small" :disabled="index === alternates.length - 1" @click="$emit('move', index, 1)">下移</button>
              <button class="danger-btn small" @click="$emit('remove', index)">删除</button>
            </div>
          </div>
          <textarea :value="item" rows="4" @input="$emit('update:item', index, ($event.target as HTMLTextAreaElement).value)"></textarea>
        </article>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  firstMessage: string
  alternates: string[]
  generating: boolean
}>()

defineEmits<{
  (e: 'update:firstMessage', value: string): void
  (e: 'update:item', index: number, value: string): void
  (e: 'add'): void
  (e: 'remove', index: number): void
  (e: 'move', index: number, direction: -1 | 1): void
  (e: 'promote', value: string): void
  (e: 'generate'): void
}>()
</script>

<style scoped>
.workbench {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.hero-block,
.list-block {
  border-radius: 20px;
  padding: 18px;
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(25, 55, 94, 0.08);
}

.hero-head,
.list-head,
.alternate-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.hero-head h3,
.list-head h3 {
  margin: 0;
}

.hero-head p,
.list-head p {
  margin: 6px 0 0;
  color: #607794;
  font-size: 13px;
  line-height: 1.6;
}

textarea {
  width: 100%;
  margin-top: 14px;
  border: 1px solid rgba(28, 55, 94, 0.12);
  background: rgba(245, 249, 254, 0.92);
  border-radius: 16px;
  padding: 14px;
  color: #183351;
  resize: vertical;
  font-size: 13px;
  line-height: 1.7;
}

.alternate-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 14px;
}

.alternate-card {
  border: 1px solid rgba(28, 55, 94, 0.08);
  border-radius: 18px;
  padding: 14px;
  background: rgba(247, 250, 254, 0.96);
}

.title {
  display: flex;
  gap: 8px;
  align-items: center;
}

.index-chip {
  border-radius: 999px;
  padding: 3px 8px;
  background: rgba(37, 99, 199, 0.1);
  color: #1f5fc3;
  font-size: 11px;
}

.actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.secondary-btn,
.ghost-btn,
.danger-btn {
  border: none;
  border-radius: 12px;
  cursor: pointer;
}

.secondary-btn,
.ghost-btn {
  padding: 10px 14px;
  background: rgba(20, 56, 106, 0.07);
  color: #234977;
}

.danger-btn {
  padding: 10px 14px;
  background: rgba(217, 55, 55, 0.12);
  color: #b83535;
}

.secondary-btn:disabled,
.ghost-btn:disabled,
.danger-btn:disabled {
  opacity: 0.68;
  cursor: not-allowed;
}

.small {
  padding: 7px 10px;
  font-size: 12px;
}

.empty-copy {
  margin-top: 14px;
  color: #6d839f;
  font-size: 13px;
}

@media (max-width: 900px) {
  .hero-head,
  .list-head,
  .alternate-head {
    flex-direction: column;
  }
}
</style>
