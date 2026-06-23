<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'

defineProps<{
  canUseBookConstraints: boolean
}>()

defineEmits<{
  (event: 'openGlossary'): void
  (event: 'openNonTranslate'): void
}>()
</script>

<template>
  <div class="book-constraints-panel">
    <div class="book-constraints-title">书籍约束</div>
    <div class="book-constraints-hint">
      术语表和禁翻表按单本漫画保存，不与其他书共享。
    </div>
    <div class="book-constraints-actions">
      <UiButton
        variant="toolbar"
        type="button"
        class="settings-button secondary-button"
        :disabled="!canUseBookConstraints"
        @click="$emit('openGlossary')"
      >
        术语表
      </UiButton>
      <UiButton
        variant="toolbar"
        type="button"
        class="settings-button secondary-button"
        :disabled="!canUseBookConstraints"
        @click="$emit('openNonTranslate')"
      >
        禁翻表
      </UiButton>
    </div>
    <div v-if="!canUseBookConstraints" class="book-constraints-disabled-note">
      仅书架模式可用
    </div>
  </div>
</template>

<style scoped>
.book-constraints-panel {
  margin-top: 14px;
  padding: 12px;
  border: 1px solid var(--settings-sidebar-workflow-border-default);
  border-radius: 12px;
  background: var(--settings-sidebar-workflow-surface-subtle);
}

.book-constraints-title {
  color: var(--settings-sidebar-workflow-text-secondary);
  font-weight: 700;
  font-size: 15px;
}

.book-constraints-hint {
  margin-top: 6px;
  color: var(--settings-sidebar-workflow-text-muted);
  font-size: 12px;
  line-height: 1.4;
}

.book-constraints-actions {
  display: flex;
  gap: 10px;
  margin-top: 12px;
}

.book-constraints-actions .settings-button {
  flex: 1;
}

.secondary-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 38px;
  padding: 0 14px;
  border: 1px solid var(--settings-sidebar-workflow-border-strong);
  border-radius: 8px;
  background: var(--color-surface-plain);
  color: var(--settings-sidebar-workflow-text-subtle);
  font-weight: 600;
  font-size: 13px;
}

.secondary-button:disabled {
  background: var(--settings-sidebar-workflow-surface-hover);
  color: var(--settings-sidebar-workflow-text-supporting);
  cursor: not-allowed;
}

.secondary-button:hover:not(:disabled) {
  background: var(--settings-sidebar-workflow-surface-active);
}

.book-constraints-disabled-note {
  margin-top: 8px;
  color: var(--settings-sidebar-workflow-text-supporting);
  font-size: 12px;
}
</style>
