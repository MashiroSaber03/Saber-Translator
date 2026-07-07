<template>
  <div class="edit-toolbar-help">
    <UiButton
      variant="toolbar"
      class="edit-toolbar-help__trigger"
      title="快捷键操作帮助"
      aria-describedby="edit-toolbar-help-tooltip"
    >
      <UiIcon name="help-circle" size="14" />
      <span class="edit-toolbar-help__trigger-text">快捷键</span>
    </UiButton>

    <div id="edit-toolbar-help-tooltip" class="edit-toolbar-help__popover" role="tooltip">
      <div class="edit-toolbar-help__section">
        <div class="edit-toolbar-help__title">
          <UiIcon name="mouse-pointer" size="14" />
          <span>鼠标操作</span>
        </div>
        <div class="edit-toolbar-help__item">
          <span class="edit-toolbar-help__key">左键点击气泡</span>
          <span class="edit-toolbar-help__description">选择气泡</span>
        </div>
        <div class="edit-toolbar-help__item">
          <span class="edit-toolbar-help__key">Shift+左键点击</span>
          <span class="edit-toolbar-help__description">多选气泡</span>
        </div>
        <div class="edit-toolbar-help__item">
          <span class="edit-toolbar-help__key">左键拖拽四角/边</span>
          <span class="edit-toolbar-help__description">调整大小</span>
        </div>
        <div class="edit-toolbar-help__item">
          <span class="edit-toolbar-help__key">左键拖拽框内部</span>
          <span class="edit-toolbar-help__description">移动气泡框</span>
        </div>
        <div class="edit-toolbar-help__item">
          <span class="edit-toolbar-help__key">中键拖拽</span>
          <span class="edit-toolbar-help__description">绘制新气泡框</span>
        </div>
      </div>

      <div class="edit-toolbar-help__section">
        <div class="edit-toolbar-help__title">
          <UiIcon name="keyboard" size="14" />
          <span>快捷键</span>
        </div>
        <div class="edit-toolbar-help__item">
          <span class="edit-toolbar-help__key">A / D</span>
          <span class="edit-toolbar-help__description">切换上/下一张图片</span>
        </div>
        <div class="edit-toolbar-help__item">
          <span class="edit-toolbar-help__key">Ctrl+Enter</span>
          <span class="edit-toolbar-help__description">应用并跳转下一张</span>
        </div>
        <div class="edit-toolbar-help__item">
          <span class="edit-toolbar-help__key">Delete / Backspace</span>
          <span class="edit-toolbar-help__description">删除选中气泡</span>
        </div>
        <div class="edit-toolbar-help__item">
          <span class="edit-toolbar-help__key">按住R+左键拖拽</span>
          <span class="edit-toolbar-help__description">修复笔刷</span>
        </div>
        <div class="edit-toolbar-help__item">
          <span class="edit-toolbar-help__key">按住U+左键拖拽</span>
          <span class="edit-toolbar-help__description">还原笔刷</span>
        </div>
        <div class="edit-toolbar-help__item">
          <span class="edit-toolbar-help__key">笔刷模式下滚轮</span>
          <span class="edit-toolbar-help__description">调整笔刷大小</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
</script>

<style scoped>
.edit-toolbar-help {
  --edit-toolbar-help-trigger-border: var(--color-border-input);
  --edit-toolbar-help-trigger-focus-border: var(--color-border-brand-gradient);
  --edit-toolbar-help-trigger-background: color-mix(in srgb, var(--color-surface-base) 90%, transparent);
  --edit-toolbar-help-trigger-hover-text: var(--color-text-brand);
  --edit-toolbar-help-section-divider: var(--color-border-muted);
  --edit-toolbar-help-popover-shadow: var(--shadow-medium);
  --edit-toolbar-help-title-text: var(--color-text-heading);
  --edit-toolbar-help-key-text: var(--color-text-secondary);
  --edit-toolbar-help-description-text: var(--color-text-heading);

  position: relative;
  display: inline-flex;
}

.edit-toolbar-help__trigger {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  height: 28px;
  padding: 0 10px;
  border: 1px solid var(--edit-toolbar-help-trigger-border);
  border-radius: 14px;
  background: var(--edit-toolbar-help-trigger-background);
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: all 0.2s;
}

.edit-toolbar-help__trigger-text {
  font-size: 12px;
  font-weight: 500;
  white-space: nowrap;
}

.edit-toolbar-help__trigger:hover {
  background: var(--color-surface-base);
  border-color: var(--edit-toolbar-help-trigger-focus-border);
  color: var(--edit-toolbar-help-trigger-hover-text);
}

.edit-toolbar-help__popover {
  position: absolute;
  top: 100%;
  right: 0;
  z-index: var(--z-overlay);
  min-width: 260px;
  margin-top: 8px;
  padding: 12px 14px;
  visibility: hidden;
  background: var(--color-surface-base);
  border: 1px solid var(--color-border-muted);
  border-radius: 10px;
  box-shadow: 0 4px 20px var(--edit-toolbar-help-popover-shadow);
  opacity: 0;
  transform: translateY(-5px);
  transition: all 0.2s ease;
}

.edit-toolbar-help:hover .edit-toolbar-help__popover,
.edit-toolbar-help:focus-within .edit-toolbar-help__popover {
  visibility: visible;
  opacity: 1;
  transform: translateY(0);
}

.edit-toolbar-help__section {
  margin-bottom: 10px;
}

.edit-toolbar-help__section:last-child {
  margin-bottom: 0;
}

.edit-toolbar-help__title {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--edit-toolbar-help-section-divider);
  color: var(--edit-toolbar-help-title-text);
  font-size: 12px;
  font-weight: 600;
}

.edit-toolbar-help__item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 3px 0;
  font-size: 11px;
}

.edit-toolbar-help__key {
  padding: 2px 6px;
  border-radius: 4px;
  background: var(--color-surface-muted);
  color: var(--edit-toolbar-help-key-text);
  font-family: var(--font-mono);
  font-size: 10px;
}

.edit-toolbar-help__description {
  color: var(--edit-toolbar-help-description-text);
}
</style>
