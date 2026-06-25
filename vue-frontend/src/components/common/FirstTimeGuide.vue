<script setup lang="ts">
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiButton from '@/components/ui/UiButton.vue'
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

/** 设置提醒关闭状态的 localStorage 存储键 */
const DISMISS_SETUP_REMINDER_KEY = 'saber_translator_dismiss_setup_reminder'

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
  // 检查用户是否已关闭设置提醒
  const dismissed = localStorage.getItem(DISMISS_SETUP_REMINDER_KEY)
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
    localStorage.setItem(DISMISS_SETUP_REMINDER_KEY, 'true')
  }
  showGuide.value = false
}

/**
 * 打开设置并关闭引导
 */
function openSettingsAndClose() {
  // 用户主动打开设置时关闭后续设置提醒。
  localStorage.setItem(DISMISS_SETUP_REMINDER_KEY, 'true')
  showGuide.value = false
  emit('openSettings')
}

/**
 * 重置引导状态（用于测试）
 */
function resetGuideState() {
  localStorage.removeItem(DISMISS_SETUP_REMINDER_KEY)
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
        <UiButton
          variant="primary" 
          class="guide-btn"
          @click="openSettingsAndClose"
        >
          <span class="button-icon">⚙️</span>
          立即配置
        </UiButton>
        <UiButton
          variant="secondary" 
          class="guide-btn"
          @click="closeGuide"
        >
          稍后配置
        </UiButton>
      </div>
      
      <div class="guide-footer">
        <UiCheckbox
          v-model="dontShowAgain"
          label="不再显示此提醒"
        />
      </div>
    </div>
  </BaseModal>
</template>

<style scoped>
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
  color: var(--color-text-default);
  margin-bottom: 12px;
}

.guide-text {
  font-size: 14px;
  color: var(--color-text-supporting);
  margin-bottom: 8px;
  line-height: 1.6;
}

.highlight {
  color: var(--color-action-primary);
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
  color: var(--color-text-supporting);
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

.button-icon {
  font-size: 16px;
}

/* 引导底部 */
.guide-footer {
  padding-top: 12px;
  border-top: 1px solid var(--color-border-muted, var(--color-border-default));
}

</style>
