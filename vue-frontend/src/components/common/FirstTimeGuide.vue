<script setup lang="ts">
/**
 * 首次使用引导组件
 * 在用户首次使用时显示设置提醒弹窗
 * 
 * 功能：
 * - 检测是否首次使用（localStorage）
 * - 显示设置提醒弹窗
 * - 支持"不再显示"选项
 * - 引导用户配置翻译服务
 */

import { ref, onMounted } from 'vue'
import BaseModal from './BaseModal.vue'

// ============================================================
// 常量定义
// ============================================================

/** localStorage 存储键（与原版 translation_validator.js 保持一致） */
const GUIDE_SHOWN_KEY = 'saber_translator_dismiss_setup_reminder'

// ============================================================
// Props 和 Emits
// ============================================================

const emit = defineEmits<{
  /** 打开设置 */
  (e: 'openSettings'): void
}>()

// ============================================================
// 状态定义
// ============================================================

/** 是否显示引导弹窗 */
const showGuide = ref(false)

/** 是否勾选"不再显示" */
const dontShowAgain = ref(false)

// ============================================================
// 生命周期
// ============================================================

onMounted(() => {
  // 检查是否已经显示过引导（与原版 translation_validator.js 保持一致）
  const dismissed = localStorage.getItem(GUIDE_SHOWN_KEY)
  if (dismissed !== 'true') {
    // 首次使用或未勾选"不再显示"，显示引导弹窗
    showGuide.value = true
  }
})

// ============================================================
// 方法
// ============================================================

/**
 * 关闭引导弹窗
 */
function closeGuide() {
  if (dontShowAgain.value) {
    // 保存"不再显示"状态
    localStorage.setItem(GUIDE_SHOWN_KEY, 'true')
  }
  showGuide.value = false
}

/**
 * 打开设置并关闭引导
 */
function openSettingsAndClose() {
  // 保存已显示状态（用户主动点击了设置按钮）
  localStorage.setItem(GUIDE_SHOWN_KEY, 'true')
  showGuide.value = false
  emit('openSettings')
}

/**
 * 重置引导状态（用于测试）
 */
function resetGuideState() {
  localStorage.removeItem(GUIDE_SHOWN_KEY)
}

// 暴露方法供外部调用
defineExpose({
  resetGuideState,
  showGuide
})
</script>

<template>
  <BaseModal
    :model-value="showGuide"
    title="欢迎使用 Saber-Translator"
    @close="closeGuide"
  >
    <div class="guide-content">
      <div class="guide-icon">🎉</div>
      
      <div class="guide-message">
        <p class="guide-title">首次使用提醒</p>
        <p class="guide-text">
          在开始翻译之前，请先配置翻译服务。
        </p>
        <p class="guide-text">
          点击右上角的 <span class="highlight">⚙️ 设置</span> 按钮，配置以下内容：
        </p>
        <ul class="guide-list">
          <li>选择 OCR 引擎（文字识别）</li>
          <li>配置翻译服务商和 API Key</li>
          <li>（可选）配置高质量翻译和 AI 校对</li>
        </ul>
      </div>
      
      <div class="guide-actions">
        <button 
          class="guide-btn primary"
          @click="openSettingsAndClose"
        >
          <span class="btn-icon">⚙️</span>
          立即配置
        </button>
        <button 
          class="guide-btn secondary"
          @click="closeGuide"
        >
          稍后配置
        </button>
      </div>
      
      <div class="guide-footer">
        <label class="dont-show-option">
          <input 
            type="checkbox" 
            v-model="dontShowAgain"
          >
          <span>不再显示此提醒</span>
        </label>
      </div>
    </div>
  </BaseModal>
</template>

<style scoped>
/* 引导内容容器 */
.guide-content {
  text-align: center;
  padding: 16px 0;
}

/* 引导图标 */
.guide-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

/* 引导消息 */
.guide-message {
  margin-bottom: 24px;
}

.guide-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary, #333);
  margin-bottom: 12px;
}

.guide-text {
  font-size: 14px;
  color: var(--text-secondary, #666);
  margin-bottom: 8px;
  line-height: 1.6;
}

.highlight {
  color: var(--color-primary, #4a90d9);
  font-weight: 500;
}

/* 引导列表 */
.guide-list {
  text-align: left;
  margin: 16px auto;
  max-width: 280px;
  padding-left: 20px;
}

.guide-list li {
  font-size: 14px;
  color: var(--text-secondary, #666);
  margin-bottom: 8px;
  line-height: 1.5;
}

/* 引导按钮 */
.guide-actions {
  display: flex;
  justify-content: center;
  gap: 12px;
  margin-bottom: 16px;
}

.guide-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 10px 20px;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.guide-btn.primary {
  background-color: var(--color-primary, #4a90d9);
  color: white;
  border: none;
}

.guide-btn.primary:hover {
  background-color: var(--primary-hover, #3a7bc8);
}

.guide-btn.secondary {
  background-color: transparent;
  color: var(--text-secondary, #666);
  border: 1px solid var(--border-color, #e0e0e0);
}

.guide-btn.secondary:hover {
  background-color: var(--hover-bg, #f5f5f5);
}

.btn-icon {
  font-size: 16px;
}

/* 引导底部 */
.guide-footer {
  padding-top: 12px;
  border-top: 1px solid var(--border-color, #e0e0e0);
}

.dont-show-option {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: var(--text-secondary, #666);
  cursor: pointer;
}

.dont-show-option input {
  cursor: pointer;
}

/* .settings-highlight 及 @keyframes settingsHighlight 已迁移到 global.css */
</style>
