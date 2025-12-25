<!--
  加载动画覆盖层组件
  显示全屏或局部的加载动画，支持自定义消息
-->
<template>
  <Teleport :to="teleportTarget" :disabled="!fullscreen">
    <Transition name="fade">
      <div
        v-if="visible"
        class="loading-overlay"
        :class="{ 'loading-overlay-fullscreen': fullscreen }"
      >
        <div class="loading-content">
          <!-- 加载动画 -->
          <div class="loading-spinner" :class="spinnerClass">
            <div class="spinner-ring"></div>
            <div class="spinner-ring"></div>
            <div class="spinner-ring"></div>
          </div>
          
          <!-- 加载消息 -->
          <div v-if="message" class="loading-message">{{ message }}</div>
          
          <!-- 进度显示 -->
          <div v-if="showProgress && progress !== undefined" class="loading-progress">
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: `${progress}%` }"></div>
            </div>
            <span class="progress-text">{{ progress }}%</span>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
/**
 * 加载动画覆盖层组件
 * 支持全屏和局部加载动画
 */
import { computed } from 'vue'

// ============================================================
// Props
// ============================================================

const props = withDefaults(defineProps<{
  /** 是否显示 */
  visible?: boolean
  /** 加载消息 */
  message?: string
  /** 是否全屏显示 */
  fullscreen?: boolean
  /** 动画大小 */
  size?: 'small' | 'medium' | 'large'
  /** 是否显示进度 */
  showProgress?: boolean
  /** 进度值（0-100） */
  progress?: number
}>(), {
  visible: false,
  message: '',
  fullscreen: false,
  size: 'medium',
  showProgress: false,
  progress: undefined
})

// ============================================================
// 计算属性
// ============================================================

/** Teleport 目标 */
const teleportTarget = computed(() => props.fullscreen ? 'body' : undefined)

/** 动画大小类名 */
const spinnerClass = computed(() => `spinner-${props.size}`)
</script>

<style scoped>
/* 覆盖层 */
.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(13, 27, 42, 0.85);
  backdrop-filter: blur(4px);
  z-index: 100;
}

.loading-overlay-fullscreen {
  position: fixed;
  z-index: 9999;
}

/* 加载内容 */
.loading-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
}

/* 加载动画 */
.loading-spinner {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}

.spinner-small {
  width: 32px;
  height: 32px;
}

.spinner-medium {
  width: 48px;
  height: 48px;
}

.spinner-large {
  width: 64px;
  height: 64px;
}

.spinner-ring {
  position: absolute;
  width: 100%;
  height: 100%;
  border: 3px solid transparent;
  border-top-color: #667eea;
  border-radius: 50%;
  animation: spin 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
}

.spinner-ring:nth-child(1) {
  animation-delay: -0.45s;
  border-top-color: #667eea;
}

.spinner-ring:nth-child(2) {
  animation-delay: -0.3s;
  border-top-color: #764ba2;
  width: 80%;
  height: 80%;
}

.spinner-ring:nth-child(3) {
  animation-delay: -0.15s;
  border-top-color: #00ff88;
  width: 60%;
  height: 60%;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

/* 加载消息 */
.loading-message {
  color: rgba(255, 255, 255, 0.9);
  font-size: 14px;
  text-align: center;
  max-width: 300px;
}

/* 进度条 */
.loading-progress {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 200px;
}

.progress-bar {
  flex: 1;
  height: 6px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea, #764ba2);
  border-radius: 3px;
  transition: width 0.3s ease;
}

.progress-text {
  color: rgba(255, 255, 255, 0.7);
  font-size: 12px;
  min-width: 40px;
  text-align: right;
}

/* 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
