<template>
  <header class="studio-topbar">
    <div class="topbar-left">
      <button class="back-btn" @click="$emit('back')">返回分析</button>
      <button class="ghost-btn" @click="$emit('open-resource')">角色资源</button>
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
      <button class="ghost-btn" :disabled="!hasDocument" @click="$emit('open-chat')">继续聊天</button>
      <button class="ghost-btn" :disabled="!hasDocument || busy" @click="$emit('new-chat')">新对话</button>
      <button class="ghost-btn" :disabled="!hasDocument" @click="$emit('open-prompt')">查看提示词</button>
      <button class="ghost-btn" @click="$emit('open-export')">导出区</button>
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
  (e: 'open-chat'): void
  (e: 'new-chat'): void
  (e: 'open-prompt'): void
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
  z-index: 40;
  border-bottom: 1px solid rgba(28, 55, 94, 0.08);
  background: rgba(248, 251, 255, 0.9);
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
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(28, 55, 94, 0.08);
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
  color: #102741;
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
  background: rgba(20, 56, 106, 0.06);
  color: #234977;
  font-size: 11px;
  line-height: 1.2;
}

.status-pill.empty {
  color: #6d839f;
}

.busy-pill {
  background: rgba(37, 99, 199, 0.12);
  color: #1f5fc3;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 160px;
  flex-shrink: 1;
}

.back-btn,
.ghost-btn,
.primary-btn {
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
.ghost-btn:hover {
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
.primary-btn:disabled {
  opacity: 0.68;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

@media (max-width: 900px) {
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
