<template>
  <div class="workbench">
    <div class="hero-block">
      <div class="hero-head">
        <div>
          <h3>主问候</h3>
          <p>角色进入对话时最先展示的开场白。它决定了语气、场景和第一印象。</p>
        </div>
        <UiButton variant="toolbar" class="action-ghost" :disabled="generating" @click="$emit('generate')">
          {{ generating ? '生成中...' : '批量生成' }}
        </UiButton>
      </div>
      <UiTextarea class="workbench-textarea" :value="firstMessage" rows="6" @input="$emit('update:firstMessage', ($event.target as HTMLTextAreaElement).value)" />
    </div>

    <div class="list-block">
      <div class="list-head">
        <div>
          <h3>备用问候</h3>
          <p>维护多种开场方式，可随时采用为主问候或继续打磨。</p>
        </div>
        <UiButton variant="toolbar" class="action-secondary" @click="$emit('add')">添加备用问候</UiButton>
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
              <UiButton variant="toolbar" class="action-ghost" @click="$emit('promote', item)" size="sm">设为主问候</UiButton>
              <UiButton variant="toolbar" class="action-ghost" :disabled="index === 0" @click="$emit('move', index, -1)" size="sm">上移</UiButton>
              <UiButton variant="toolbar" class="action-ghost" :disabled="index === alternates.length - 1" @click="$emit('move', index, 1)" size="sm">下移</UiButton>
              <UiButton variant="toolbar" class="action-danger" @click="$emit('remove', index)" size="sm">删除</UiButton>
            </div>
          </div>
          <UiTextarea class="workbench-textarea" :value="item" rows="4" @input="$emit('update:item', index, ($event.target as HTMLTextAreaElement).value)" />
        </article>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiButton from '@/components/ui/UiButton.vue'
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
  --greeting-workbench-block-border: rgba(25, 55, 94, .08);
  --greeting-workbench-block-background: rgba(255, 255, 255, .82);
  --greeting-workbench-alternate-card-background: rgba(247, 250, 254, .96);
  --ui-textarea-border: 1px solid var(--studio-border-strong);
  --ui-textarea-background: var(--studio-surface-soft);
  --ui-textarea-radius: 16px;
  --ui-textarea-padding: 14px;
  --ui-textarea-color: var(--studio-text-strong);
  --ui-textarea-font-size: 13px;
  --ui-textarea-line-height: 1.7;

  display: flex;
  flex-direction: column;
  gap: 18px;
}

.hero-block,
.list-block {
  border-radius: 20px;
  padding: 18px;
  background: var(--greeting-workbench-block-background);
  border: 1px solid var(--greeting-workbench-block-border);
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
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.6;
}

.workbench-textarea {
  margin-top: 14px;
}

.alternate-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 14px;
}

.alternate-card {
  border: 1px solid var(--studio-border-default);
  border-radius: 18px;
  padding: 14px;
  background: var(--greeting-workbench-alternate-card-background);
}

.title {
  display: flex;
  gap: 8px;
  align-items: center;
}

.index-chip {
  border-radius: 999px;
  padding: 3px 8px;
  background: var(--studio-surface-tint);
  color: var(--color-text-primary-strong);
  font-size: 11px;
}

.actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.action-secondary,
.action-ghost,
.action-danger {
  border: none;
  border-radius: 12px;
  cursor: pointer;
}

.action-secondary,
.action-ghost {
  padding: 10px 14px;
  background: var(--studio-surface-muted);
  color: var(--studio-text-default);
}

.action-danger {
  padding: 10px 14px;
  background: var(--color-surface-danger-soft);
  color: var(--studio-text-danger);
}

.action-secondary:disabled,
.action-ghost:disabled,
.action-danger:disabled {
  opacity: 0.68;
  cursor: not-allowed;
}

.small {
  padding: 7px 10px;
  font-size: 12px;
}

.empty-copy {
  margin-top: 14px;
  color: var(--studio-text-subtle);
  font-size: 13px;
}

@media (--breakpoint-lg-down) {
  .hero-head,
  .list-head,
  .alternate-head {
    flex-direction: column;
  }
}
</style>
