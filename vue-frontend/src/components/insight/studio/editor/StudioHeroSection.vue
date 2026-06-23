<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import type { CharacterStudioDocument, CharacterStudioEditorPendingState } from '@/types/characterStudio'

defineProps<{
  avatarUrl: string
  document: CharacterStudioDocument
  formatOrigin: (origin: CharacterStudioDocument['origin']['type']) => string
  isGenerationLocked: boolean
  isGenerating: (section: string) => boolean
  pendingState: CharacterStudioEditorPendingState
}>()

defineEmits<{
  (event: 'delete'): void
  (event: 'generate', section: string): void
}>()
</script>

<template>
  <section class="overview-hero">
    <div class="hero-main">
      <div class="avatar-shell">
        <img v-if="avatarUrl" :src="avatarUrl" alt="角色头像">
        <div v-else class="avatar-placeholder">{{ document.identity.name.slice(0, 1) || '角' }}</div>
      </div>
      <div class="hero-copy">
        <div class="hero-kicker">当前角色</div>
        <h2>{{ document.meta.title || document.identity.name }}</h2>
        <p>{{ document.identity.description || '当前角色还没有完善简介，建议先使用“AI 一键补全整卡”，再回到分区里精修。' }}</p>
        <div class="hero-meta">
          <span class="meta-pill">{{ formatOrigin(document.origin.type) }}</span>
          <span v-if="document.origin.source_character" class="meta-pill">来源: {{ document.origin.source_character }}</span>
          <span v-if="document.meta.tags.length > 0" class="meta-pill">{{ document.meta.tags.length }} 个标签</span>
          <span v-if="document.status.frozen_sections.length > 0" class="meta-pill">已钉住 {{ document.status.frozen_sections.length }} 区块</span>
        </div>
      </div>
    </div>
    <div class="hero-actions">
      <UiButton
        variant="toolbar"
        class="action-primary"
        :disabled="isGenerationLocked"
        @click="$emit('generate', 'full')"
      >
        {{ isGenerating('full') ? '整卡补全中...' : 'AI 一键补全整卡' }}
      </UiButton>
      <UiButton
        variant="toolbar"
        class="action-ghost"
        :disabled="isGenerationLocked"
        @click="$emit('generate', 'review')"
      >
        {{ isGenerating('review') ? '审查中...' : 'AI 审查当前角色' }}
      </UiButton>
      <UiButton
        variant="toolbar"
        class="action-danger"
        :disabled="pendingState.deleting"
        @click="$emit('delete')"
      >
        {{ pendingState.deleting ? '删除中...' : '删除文档' }}
      </UiButton>
    </div>
  </section>
</template>

<style scoped>
.overview-hero {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  padding: 22px;
  border: 1px solid var(--studio-border-default);
  border-radius: 28px;
  background: var(--character-studio-editor-surface-base);
  box-shadow: 0 26px 42px var(--studio-shadow-floating);
}

.hero-main {
  display: flex;
  gap: 18px;
  min-width: 0;
}

.avatar-shell {
  flex-shrink: 0;
  width: 116px;
  height: 164px;
  overflow: hidden;
  border-radius: 24px;
  background: linear-gradient(180deg, var(--studio-surface-tint-strong), var(--character-studio-editor-surface-raised));
}

.avatar-shell img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.avatar-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  color: var(--color-text-primary-strong);
  font-weight: 700;
  font-size: 32px;
}

.hero-copy {
  min-width: 0;
}

.hero-kicker {
  color: var(--character-studio-editor-text-muted);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.hero-copy h2 {
  margin: 10px 0 0;
  color: var(--character-studio-editor-text-primary);
  font-size: 30px;
}

.hero-copy p {
  margin: 12px 0 0;
  color: var(--studio-text-muted);
  line-height: 1.8;
}

.hero-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
}

.meta-pill {
  padding: 5px 10px;
  border-radius: 999px;
  background: var(--studio-surface-muted);
  color: var(--studio-text-default);
  font-size: 11px;
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  align-content: flex-start;
  justify-content: flex-end;
  gap: 10px;
}

.action-ghost,
.action-primary,
.action-danger {
  border: none;
  border-radius: 14px;
  cursor: pointer;
}

.action-ghost {
  padding: 11px 14px;
  background: var(--studio-surface-muted);
  color: var(--studio-text-default);
}

.action-primary {
  padding: 11px 16px;
  background: linear-gradient(135deg, var(--character-studio-editor-surface-hover), var(--character-studio-editor-surface-active));
  box-shadow: 0 12px 24px var(--character-studio-editor-shadow-default);
  color: var(--color-text-inverse);
}

.action-danger {
  padding: 11px 16px;
  background: var(--color-surface-danger-soft);
  color: var(--studio-text-danger);
}

.action-ghost:disabled,
.action-primary:disabled,
.action-danger:disabled {
  opacity: 0.62;
  cursor: not-allowed;
}

@media (--breakpoint-studio-down) {
  .overview-hero {
    flex-direction: column;
  }
}

@media (--breakpoint-preview-down) {
  .hero-main {
    flex-direction: column;
  }
}
</style>
