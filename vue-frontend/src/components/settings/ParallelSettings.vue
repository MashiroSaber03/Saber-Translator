<script setup lang="ts">
/**
 * 并行翻译设置组件
 * 
 * 放在设置面板的"更多"tab中
 */

import { computed } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'

const settingsStore = useSettingsStore()

const parallelEnabled = computed({
  get: () => settingsStore.settings.parallel.enabled,
  set: (value: boolean) => {
    settingsStore.updateSettings({
      parallel: {
        ...settingsStore.settings.parallel,
        enabled: value
      }
    })
  }
})

const lockSize = computed({
  get: () => settingsStore.settings.parallel.deepLearningLockSize,
  set: (value: number) => {
    settingsStore.updateSettings({
      parallel: {
        ...settingsStore.settings.parallel,
        deepLearningLockSize: Math.max(1, Math.min(4, value))
      }
    })
  }
})
</script>

<template>
  <div class="parallel-settings">
    <!-- 并行翻译设置 -->
    <div class="settings-group">
      <div class="settings-group-title">🚀 并行翻译</div>
      
      <!-- 启用开关 -->
      <div class="settings-item">
        <label>启用并行模式:</label>
        <label class="toggle-switch">
          <input type="checkbox" v-model="parallelEnabled">
          <span class="toggle-slider"></span>
        </label>
        <div class="input-hint">使用流水线并行处理，可能提升批量翻译速度</div>
      </div>

      <!-- 深度学习锁大小 -->
      <div class="settings-item" :class="{ 'item-disabled': !parallelEnabled }">
        <label>深度学习并发数:</label>
        <div class="number-control">
          <button class="btn btn-sm" @click="lockSize = Math.max(1, lockSize - 1)" :disabled="!parallelEnabled">-</button>
          <input 
            type="number" 
            v-model.number="lockSize" 
            min="1" 
            max="4"
            :disabled="!parallelEnabled"
            class="number-input"
          >
          <button class="btn btn-sm" @click="lockSize = Math.min(4, lockSize + 1)" :disabled="!parallelEnabled">+</button>
        </div>
        <div class="input-hint">控制检测/OCR/颜色/修复的最大并发数（建议1-2）</div>
      </div>

      <!-- 说明 -->
      <div class="settings-note" v-if="parallelEnabled">
        <div class="note-title">⚠️ 注意事项：</div>
        <ul>
          <li>并发数设为1时为串行执行，最稳定</li>
          <li>增大并发数可能加速处理，但会占用更多GPU/CPU资源</li>
          <li>如果遇到显存不足，请将并发数设为1</li>
        </ul>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 开关样式 */
.toggle-switch {
  position: relative;
  display: inline-block;
  width: 44px;
  height: 24px;
  margin-left: 8px;
}

.toggle-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-slider {
  position: absolute;
  cursor: pointer;
  inset: 0;
  background-color: var(--border-color);
  transition: 0.3s;
  border-radius: 24px;
}

.toggle-slider::before {
  position: absolute;
  content: "";
  height: 18px;
  width: 18px;
  left: 3px;
  bottom: 3px;
  background-color: white;
  transition: 0.3s;
  border-radius: 50%;
}

input:checked + .toggle-slider {
  background-color: var(--color-primary);
}

input:checked + .toggle-slider::before {
  transform: translateX(20px);
}

/* 数字控制 */
.number-control {
  display: flex;
  align-items: center;
  gap: 4px;
}

.number-control .btn-sm {
  width: 28px;
  height: 28px;
  padding: 0;
  font-size: 14px;
}

.number-input {
  width: 50px;
  height: 28px;
  text-align: center;
  border: 1px solid var(--border-color);
  background: var(--input-bg);
  color: var(--text-color);
  border-radius: 4px;
  font-size: 14px;
}

.number-input::-webkit-inner-spin-button,
.number-input::-webkit-outer-spin-button {
  appearance: none;
  margin: 0;
}

/* 禁用状态 */
.item-disabled {
  opacity: 0.5;
  pointer-events: none;
}

/* 说明提示 */
.settings-note {
  margin-top: 12px;
  padding: 10px 12px;
  background: rgb(255, 193, 7, 0.1);
  border: 1px solid rgb(255, 193, 7, 0.3);
  border-radius: 6px;
  font-size: 12px;
}

.note-title {
  color: #ffc107;
  font-weight: 500;
  margin-bottom: 6px;
}

.settings-note ul {
  margin: 0;
  padding-left: 18px;
  color: var(--text-secondary);
}

.settings-note li {
  margin: 3px 0;
}
</style>
