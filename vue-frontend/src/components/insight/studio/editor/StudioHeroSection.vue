<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductAvatar from '@/components/product/ProductAvatar.vue'
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
  <section class="studio-hero-section">
    <div class="studio-hero-section__main">
      <ProductAvatar
        class="studio-hero-section__avatar"
        :image-src="avatarUrl"
        label="角色头像"
        :fallback-text="document.identity.name"
        size="hero"
        shape="portrait"
      />
      <div class="studio-hero-section__copy">
        <div class="studio-hero-section__kicker">当前角色</div>
        <h2 class="studio-hero-section__title">{{ document.meta.title || document.identity.name }}</h2>
        <p class="studio-hero-section__description">{{ document.identity.description || '当前角色还没有完善简介，建议先使用“AI 一键补全整卡”，再回到分区里精修。' }}</p>
        <div class="studio-hero-section__meta">
          <span class="studio-hero-section__meta-pill">{{ formatOrigin(document.origin.type) }}</span>
          <span v-if="document.origin.source_character" class="studio-hero-section__meta-pill">来源: {{ document.origin.source_character }}</span>
          <span v-if="document.meta.tags.length > 0" class="studio-hero-section__meta-pill">{{ document.meta.tags.length }} 个标签</span>
          <span v-if="document.status.frozen_sections.length > 0" class="studio-hero-section__meta-pill">已钉住 {{ document.status.frozen_sections.length }} 区块</span>
        </div>
      </div>
    </div>
    <ProductActionRow class="studio-hero-section__actions" aria-label="角色概览操作">
      <UiButton
        variant="primary"
        :disabled="isGenerationLocked"
        @click="$emit('generate', 'full')"
      >
        {{ isGenerating('full') ? '整卡补全中...' : 'AI 一键补全整卡' }}
      </UiButton>
      <UiButton
        variant="secondary"
        :disabled="isGenerationLocked"
        @click="$emit('generate', 'review')"
      >
        {{ isGenerating('review') ? '审查中...' : 'AI 审查当前角色' }}
      </UiButton>
      <UiButton
        variant="secondary"
        tone="danger"
        :disabled="pendingState.deleting"
        @click="$emit('delete')"
      >
        {{ pendingState.deleting ? '删除中...' : '删除文档' }}
      </UiButton>
    </ProductActionRow>
  </section>
</template>

<style scoped>
.studio-hero-section {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  padding: 22px;
  border: 1px solid var(--studio-border-default);
  border-radius: 28px;
  background: color-mix(in srgb, var(--color-surface-card) 88%, transparent);
  box-shadow: 0 26px 42px var(--studio-shadow-floating);
}

.studio-hero-section__main {
  display: flex;
  gap: 18px;
  min-width: 0;
}

.studio-hero-section__avatar {
  --product-avatar-background: linear-gradient(180deg, var(--studio-surface-tint-strong), color-mix(in srgb, var(--color-text-heading) 4%, transparent));
  --product-avatar-color: var(--color-text-primary-strong);
}

.studio-hero-section__copy {
  min-width: 0;
}

.studio-hero-section__kicker {
  color: color-mix(in srgb, var(--color-action-primary) 27%, color-mix(in srgb, var(--color-action-brand-strong) 17.808%, var(--color-text-subtle)));
  font-size: 11px;
  letter-spacing: 0;
  text-transform: uppercase;
}

.studio-hero-section__title {
  margin: 10px 0 0;
  color: var(--color-text-heading);
  font-size: 30px;
}

.studio-hero-section__description {
  margin: 12px 0 0;
  color: var(--studio-text-muted);
  line-height: 1.8;
}

.studio-hero-section__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
}

.studio-hero-section__meta-pill {
  padding: 5px 10px;
  border-radius: 999px;
  background: var(--studio-surface-muted);
  color: var(--studio-text-default);
  font-size: 11px;
}

.studio-hero-section__actions {
  align-content: flex-start;
  justify-content: flex-end;
}

@media (--breakpoint-studio-down) {
  .studio-hero-section {
    flex-direction: column;
  }
}

@media (--breakpoint-preview-down) {
  .studio-hero-section__main {
    flex-direction: column;
  }
}
</style>
