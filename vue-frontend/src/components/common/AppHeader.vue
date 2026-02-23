<template>
  <header class="app-header" :class="[`app-header--${variant}`]">
    <div class="header-content">
      <!-- 左侧：Logo 和应用名称 -->
      <div class="logo-container">
        <router-link to="/" :title="logoTitle">
          <img :src="'/pic/logo.png'" alt="Saber-Translator Logo" class="app-logo" />
          <span class="app-name">Saber-Translator</span>
        </router-link>
      </div>

      <!-- 右侧：导航链接（使用 slot 允许各 View 自定义内容） -->
      <div class="header-links">
        <slot name="header-links">
          <!-- 默认内容：通用导航链接 -->

          <!-- 返回书架按钮（仅在非书架页面显示） -->
          <router-link v-if="showBackToShelf" to="/" class="back-to-shelf">
            📚 返回书架
          </router-link>

          <!-- 保存按钮（仅在书架模式下显示） -->
          <button
            v-if="showSaveButton"
            class="save-header-btn"
            title="保存当前进度"
            @click="$emit('save')"
          >
            💾 保存
          </button>

          <!-- 设置按钮 -->
          <button
            v-if="showSettingsButton"
            class="header-btn settings-btn"
            :class="{ 'highlight-animation': highlightSettings }"
            title="设置"
            @click="$emit('openSettings')"
          >
            ⚙️
          </button>

          <!-- 使用教程链接 -->
          <a
            href="http://www.mashirosaber.top"
            target="_blank"
            class="tutorial-link"
            title="使用教程"
          >
            📖 使用教程
          </a>

          <!-- 赞助按钮 -->
          <a href="#" class="donate-link" title="请作者喝奶茶" @click.prevent="$emit('donate')">
            🍵 赞助
          </a>

          <!-- GitHub 链接 -->
          <a
            href="https://github.com/MashiroSaber03/saber-translator"
            target="_blank"
            class="github-link"
            title="GitHub 仓库"
          >
            <img :src="'/pic/github.jpg'" alt="GitHub" class="github-icon" />
            GitHub
          </a>

          <button class="theme-toggle" title="功能开发中" @click="showFeatureNotice">
            <span class="theme-icon">☀️</span>
          </button>
        </slot>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { showToast } from '@/utils/toast'

// Props 定义
interface Props {
  /** 是否显示保存按钮 */
  showSaveButton?: boolean
  /** 是否显示设置按钮 */
  showSettingsButton?: boolean
  /** 是否高亮设置按钮（引导动画） */
  highlightSettings?: boolean
  /** 样式变体：'default' 为浮动圆角头部，'bookshelf' 为紫色渐变头部，'insight' 为固定全宽头部 */
  variant?: 'default' | 'bookshelf' | 'insight'
  /** Logo 链接的 title 属性 */
  logoTitle?: string
}

withDefaults(defineProps<Props>(), {
  showSaveButton: false,
  showSettingsButton: false,
  highlightSettings: false,
  variant: 'default',
  logoTitle: '返回书架'
})

// Emits 定义
defineEmits<{
  /** 保存按钮点击 */
  save: []
  /** 设置按钮点击 */
  openSettings: []
  /** 赞助按钮点击 */
  donate: []
}>()

// 路由和状态
const route = useRoute()

// 计算属性：是否显示返回书架按钮
const showBackToShelf = computed(() => {
  return route.path !== '/'
})

// 显示功能开发中提示
function showFeatureNotice(): void {
  showToast('🌙 该功能正在开发中，敬请期待！', 'info')
}

</script>

<style scoped>
/* ============ 头部样式（默认变体） ============ */

.app-header {
  background: transparent;
  color: #2c3e50;
  padding: 10px 20px;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
  width: auto;
  margin: 0 auto;
  max-width: calc(100% - 700px);
  z-index: var(--z-dropdown);
}

.header-content {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 10px;
  background: rgb(255, 255, 255, 0.9);
  border-radius: 12px;
  box-shadow: 0 2px 10px rgb(0,0,0,0.1);
}

.logo-container {
  display: flex;
  align-items: center;
}

.logo-container a {
  display: flex;
  align-items: center;
  text-decoration: none;
  color: #2c3e50;
}

.app-logo {
  height: 40px;
  width: auto;
  margin-right: 15px;
  border-radius: 8px;
}

.app-name {
  font-size: 1.5em;
  font-weight: bold;
  letter-spacing: 0.5px;
}

.header-links {
  display: flex;
  align-items: center;
  gap: 15px;
}

.tutorial-link, .github-link {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 12px;
  background-color: rgb(0,0,0,0.05);
  border-radius: 20px;
  color: #2c3e50;
  text-decoration: none;
  transition: all 0.3s ease;
}

.tutorial-link:hover, .github-link:hover {
  background-color: rgb(0,0,0,0.1);
  transform: translateY(-2px);
}

.github-icon {
  width: 20px;
  height: 20px;
  border-radius: 50%;
}

/* 赞助按钮样式 */
.donate-link {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 12px;
  background-color: rgb(255, 105, 180, 0.15);
  border-radius: 20px;
  color: #e91e63;
  text-decoration: none;
  transition: all 0.3s ease;
}

.donate-link:hover {
  background-color: rgb(255, 105, 180, 0.25);
  transform: translateY(-2px);
}

/* 返回书架按钮样式 */
.back-to-shelf {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 14px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 20px;
  color: white;
  text-decoration: none;
  font-size: 0.9em;
  font-weight: 500;
  transition: all 0.3s ease;
}

.back-to-shelf:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgb(102, 126, 234, 0.4);
}

/* 保存按钮样式（顶部） */
.save-header-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px 14px;
  background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
  border: none;
  border-radius: 20px;
  color: white;
  font-size: 1em;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.save-header-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgb(40, 167, 69, 0.4);
}

.save-header-btn:active {
  transform: translateY(0);
}

/* 设置按钮基础样式 */
.header-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px 12px;
  background-color: rgb(0, 0, 0, 0.05);
  border: none;
  border-radius: 20px;
  cursor: pointer;
  font-size: 1.1em;
  transition: all 0.3s ease;
}

.header-btn:hover {
  background-color: rgb(0, 0, 0, 0.1);
  transform: translateY(-2px);
}

/* 设置按钮高亮引导动画 */
.highlight-animation {
  animation: pulse-highlight 1.5s ease-in-out infinite;
}

.theme-toggle {
  background-color: #f0f2f5;
  border: 1px solid #dcdfe6;
  border-radius: 20px;
  cursor: pointer;
  padding: 6px 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 42px;
  transition: background-color 0.3s;
}

.theme-toggle:hover {
  background-color: #e6e8eb;
}

.theme-icon {
  font-size: 16px;
}

/* ============ Insight 变体样式 ============ */

.app-header--insight {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 56px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-color);
  z-index: var(--z-dropdown);
  display: flex;
  align-items: center;
  padding: 0 20px;
  max-width: none;
  width: auto;
  margin: 0;
}

.app-header--insight .header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  max-width: 100%;
  padding: 0;
  background: transparent;
  border-radius: 0;
  box-shadow: none;
}

.app-header--insight .logo-container a {
  display: flex;
  align-items: center;
  gap: 10px;
  text-decoration: none;
  color: var(--text-primary);
}

.app-header--insight .app-logo {
  height: 32px;
  width: auto;
  max-height: 32px;
  margin-right: 0;
}

.app-header--insight .app-name {
  font-weight: 600;
  font-size: 18px;
}

.app-header--insight .header-links {
  display: flex;
  align-items: center;
  gap: 16px;
}

/* ============ Bookshelf 变体样式 ============ */

.app-header--bookshelf {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 0 24px;
  height: 64px;
  box-shadow: 0 2px 20px rgba(102, 126, 234, 0.3);
  position: sticky;
  top: 0;
  z-index: var(--z-sticky, 200);
  display: flex;
  align-items: center;
  max-width: none;
  width: 100%;
  margin: 0;
}

.app-header--bookshelf .header-content {
  max-width: 1400px;
  width: 100%;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: transparent;
  box-shadow: none;
  padding: 0;
  border-radius: 0;
}

.app-header--bookshelf .logo-container a {
  display: flex;
  align-items: center;
  gap: 12px;
  text-decoration: none;
  color: white;
}

.app-header--bookshelf .app-logo {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  margin-right: 0;
}

.app-header--bookshelf .app-name {
  font-size: 1.3rem;
  font-weight: 700;
  color: white;
  letter-spacing: -0.5px;
}

.app-header--bookshelf .header-links {
  display: flex;
  align-items: center;
  gap: 16px;
}
</style>

