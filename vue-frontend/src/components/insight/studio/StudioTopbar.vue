<template>
  <header class="studio-topbar">
    <div class="topbar-left">
      <button class="icon-btn drawer-btn" title="打开导航" @click="$emit('toggle-left-drawer')">☰</button>
      <button class="back-btn" @click="$emit('back')">返回分析</button>
      <div class="title-block">
        <div class="kicker">漫画分析 / 角色工坊</div>
        <h1>角色工坊 2.0</h1>
        <p>{{ subtitle }}</p>
        <div class="inline-status">
          <span v-if="bookTitle" class="status-pill">当前书籍：{{ bookTitle }}</span>
          <span class="status-pill" :class="{ empty: !documentTitle }">
            {{ documentTitle ? `当前角色：${documentTitle}` : '当前角色：未选择' }}
          </span>
          <span v-if="documentOrigin" class="status-pill">{{ documentOrigin }}</span>
          <span v-if="busy && busyLabel" class="status-pill busy-pill">{{ busyLabel }}</span>
        </div>
      </div>
    </div>

    <div class="topbar-right">
      <button class="icon-btn drawer-btn" title="打开运行时侧栏" @click="$emit('toggle-right-drawer')">⌘</button>
      <button class="ghost-btn" @click="$emit('open-export')">导出区</button>
      <button class="ghost-btn" :disabled="!hasDocument" @click="$emit('toggle-preview')">
        {{ previewCollapsed ? '展开预览' : '收起预览' }}
      </button>
      <button class="ghost-btn" :disabled="!hasDocument || validatePending" @click="$emit('validate')">
        {{ validatePending ? '诊断中...' : '诊断' }}
      </button>
      <button class="primary-btn" :disabled="!hasDocument || savePending" @click="$emit('save')">
        {{ savePending ? '保存中...' : '保存' }}
      </button>
    </div>
  </header>
</template>

<script setup lang="ts">
defineProps<{
  subtitle: string
  bookTitle: string
  documentTitle: string
  documentOrigin: string
  hasDocument: boolean
  previewCollapsed: boolean
  busy: boolean
  busyLabel: string
  savePending: boolean
  validatePending: boolean
}>()

defineEmits<{
  (e: 'back'): void
  (e: 'save'): void
  (e: 'validate'): void
  (e: 'open-export'): void
  (e: 'toggle-preview'): void
  (e: 'toggle-left-drawer'): void
  (e: 'toggle-right-drawer'): void
}>()
</script>

<style scoped>
.studio-topbar {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: center;
  padding: 16px 24px;
  position: sticky;
  top: 0;
  z-index: 40;
  border-bottom: 1px solid rgba(28, 55, 94, 0.08);
  background: rgba(248, 251, 255, 0.9);
  backdrop-filter: blur(18px);
}

.topbar-left,
.topbar-right {
  display: flex;
  align-items: center;
  gap: 12px;
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
}

.kicker {
  font-size: 11px;
  letter-spacing: 0.12em;
  color: #6f84a2;
  font-weight: 600;
}

.title-block h1 {
  margin: 0;
  font-size: 24px;
  line-height: 1.1;
  color: #102741;
}

.title-block p {
  margin: 0;
  color: #5f7591;
  font-size: 13px;
}

.inline-status {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.status-pill {
  border-radius: 999px;
  padding: 5px 10px;
  background: rgba(20, 56, 106, 0.06);
  color: #234977;
  font-size: 11px;
}

.status-pill.empty {
  color: #6d839f;
}

.busy-pill {
  background: rgba(37, 99, 199, 0.12);
  color: #1f5fc3;
}

.back-btn,
.ghost-btn,
.primary-btn,
.icon-btn {
  border: none;
  border-radius: 14px;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
  white-space: nowrap;
}

.back-btn,
.ghost-btn {
  padding: 11px 15px;
  background: rgba(20, 56, 106, 0.07);
  color: #234977;
}

.back-btn:hover,
.ghost-btn:hover,
.icon-btn:hover {
  transform: translateY(-1px);
}

.primary-btn {
  padding: 11px 18px;
  background: linear-gradient(135deg, #2563c7, #4d86ee);
  color: #fff;
  box-shadow: 0 12px 24px rgba(37, 99, 199, 0.22);
}

.back-btn:disabled,
.ghost-btn:disabled,
.primary-btn:disabled,
.icon-btn:disabled {
  opacity: 0.68;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.icon-btn {
  width: 42px;
  height: 42px;
  background: rgba(20, 56, 106, 0.07);
  color: #234977;
  font-size: 18px;
}

.drawer-btn {
  display: none;
}

@media (max-width: 1280px) {
  .drawer-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .inline-status {
    display: none;
  }

  .topbar-right {
    max-width: 100%;
  }
}

@media (max-width: 900px) {
  .topbar-left,
  .topbar-right {
    width: 100%;
    flex-wrap: wrap;
  }
}
</style>
