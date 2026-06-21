<template>
  <header class="studio-topbar">
    <div class="topbar-left">
      <UiButton variant="toolbar" class="back-btn" @click="$emit('back')">返回分析</UiButton>
      <UiButton variant="toolbar" class="action-ghost" @click="$emit('open-resource')">角色资源</UiButton>
      <div class="title-block">
        <div class="title-row">
          <h1>角色工坊 2.0</h1>
          <span v-if="busy && busyLabel" class="status-pill busy-pill">{{ busyLabel }}</span>
        </div>
        <div class="meta-row">
          <span v-if="bookTitle" class="status-pill">当前书籍：{{ bookTitle }}</span>
          <span class="status-pill" :class="{ empty: !documentTitle }">
            {{ documentTitle ? `当前角色：${documentTitle}` : '当前角色：未选择' }}
          </span>
          <span v-if="documentOrigin" class="status-pill">{{ documentOrigin }}</span>
        </div>
      </div>
    </div>

    <div class="topbar-right">
      <UiButton variant="toolbar" class="action-ghost" @click="$emit('open-export')">导出区</UiButton>
      <UiButton variant="toolbar" class="action-ghost" :disabled="!hasDocument || validatePending" @click="$emit('validate')">
        {{ validatePending ? '诊断中...' : '诊断' }}
      </UiButton>
      <UiButton variant="toolbar" class="action-primary" :disabled="!hasDocument || savePending" @click="$emit('save')">
        {{ savePending ? '保存中...' : '保存' }}
      </UiButton>
    </div>
  </header>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
defineProps<{
  bookTitle: string
  documentTitle: string
  documentOrigin: string
  hasDocument: boolean
  busy: boolean
  busyLabel: string
  savePending: boolean
  validatePending: boolean
}>()

defineEmits<{
  (e: 'back'): void
  (e: 'save'): void
  (e: 'validate'): void
  (e: 'open-resource'): void
  (e: 'open-export'): void
}>()
</script>

<style scoped>
.studio-topbar {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
  padding: 10px 20px;
  position: sticky;
  top: 0;
  z-index: var(--z-app-header);
  border-bottom: 1px solid var(--color-border-studio);
  background: var(--studio-topbar-surface-base);
  backdrop-filter: blur(18px);
}

.topbar-left,
.topbar-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.topbar-left {
  min-width: 0;
  flex: 1;
}

.topbar-right {
  flex-shrink: 0;
  justify-content: flex-end;
  flex-wrap: wrap;
}

.title-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
  flex: 0 1 420px;
  max-width: 420px;
  padding: 8px 12px;
  border-radius: 18px;
  background: var(--color-surface-raised);
  border: 1px solid var(--color-border-studio);
}

.title-row {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.title-row h1 {
  margin: 0;
  font-size: 20px;
  line-height: 1.1;
  color: var(--studio-topbar-text-primary);
  white-space: nowrap;
}

.meta-row {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-items: center;
}

.status-pill {
  border-radius: 999px;
  padding: 4px 9px;
  background: var(--studio-topbar-surface-raised);
  color: var(--color-text-studio);
  font-size: 11px;
  line-height: 1.2;
}

.status-pill.empty {
  color: var(--color-text-studio-subtle);
}

.busy-pill {
  background: var(--color-surface-studio-tint2);
  color: var(--color-text-primary-strong);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 160px;
  flex-shrink: 1;
}

.back-btn,
.action-ghost,
.action-primary {
  border: none;
  border-radius: 14px;
  font-size: 14px;
  line-height: 1.2;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
  white-space: nowrap;
}

.back-btn,
.action-ghost {
  padding: 11px 15px;
  background: var(--color-surface-studio-muted);
  color: var(--color-text-studio);
}

.back-btn:hover,
.action-ghost:hover {
  transform: translateY(-1px);
}

.action-primary {
  padding: 11px 18px;
  background: linear-gradient(135deg, var(--studio-topbar-surface-muted), var(--studio-topbar-surface-subtle));
  color: var(--color-text-inverse);
  box-shadow: 0 12px 24px var(--studio-topbar-shadow-default);
}

.back-btn:disabled,
.action-ghost:disabled,
.action-primary:disabled {
  opacity: 0.68;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

@media (--breakpoint-lg-down) {
  .studio-topbar {
    padding: 12px 16px;
  }

  .topbar-left,
  .topbar-right {
    width: 100%;
    flex-wrap: wrap;
  }

  .title-block {
    flex: 1 1 100%;
    max-width: none;
  }

  .title-row {
    flex-wrap: wrap;
  }

  .busy-pill {
    max-width: none;
  }
}
</style>
