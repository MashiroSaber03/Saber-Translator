<template>
  <header class="app-header" :class="[`app-header--${variant}`]">
    <div class="app-header__content">
      <!-- 左侧：Logo 和应用名称 -->
      <div class="app-header__logo-container">
        <router-link to="/" :title="logoTitle" class="app-header__logo-link">
          <img :src="'/pic/logo.png'" alt="Saber-Translator Logo" class="app-header__logo" />
          <span class="app-header__name">Saber-Translator</span>
        </router-link>
      </div>

      <!-- 右侧：导航链接（使用 slot 允许各 View 自定义内容） -->
      <div class="app-header__links">
        <slot name="header-links">
          <!-- 默认内容：通用导航链接 -->

          <!-- 返回书架按钮（仅在非书架页面显示） -->
          <router-link v-if="showBackToShelf" to="/" class="app-header__back-link">
            📚 返回书架
          </router-link>

          <!-- 保存按钮（仅在书架模式下显示） -->
          <UiButton
            variant="toolbar"
            v-if="showSaveButton"
            class="app-header__save-button"
            title="保存当前进度"
            @click="$emit('save')"
          >
            💾 保存
          </UiButton>

          <!-- 设置按钮 -->
          <UiButton
            variant="toolbar"
            v-if="showSettingsButton"
            class="app-header__settings-button"
            :class="{ 'app-header__settings-button--highlight': highlightSettings }"
            title="设置"
            aria-label="打开设置"
            @click="$emit('openSettings')"
          >
            ⚙️
          </UiButton>

          <!-- 使用教程链接 -->
          <a
            href="http://www.mashirosaber.top"
            target="_blank"
            rel="noopener noreferrer"
            class="app-header__link app-header__link--tutorial"
            title="使用教程"
          >
            📖 使用教程
          </a>

          <!-- 赞助按钮 -->
          <a href="#" class="app-header__link app-header__link--donate" title="请作者喝奶茶" @click.prevent="$emit('donate')">
            🍵 赞助
          </a>

          <!-- GitHub 链接 -->
          <a
            href="https://github.com/MashiroSaber03/saber-translator"
            target="_blank"
            rel="noopener noreferrer"
            class="app-header__link app-header__link--github"
            title="GitHub 仓库"
          >
            <img :src="'/pic/github.jpg'" alt="GitHub" class="app-header__github-icon" />
            GitHub
          </a>

          <UiButton
            variant="toolbar"
            class="app-header__theme-toggle"
            title="功能开发中"
            aria-label="功能开发中"
            @click="showFeatureNotice"
          >
            <span class="app-header__theme-icon">☀️</span>
          </UiButton>
        </slot>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">

import UiButton from '@/components/ui/UiButton.vue'
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

<style scoped>/* ============ 头部样式（默认变体） ============ */

.app-header {
  background: transparent;
  color: var(--color-text-heading);
  padding: 10px 20px;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
  width: auto;
  margin: 0 auto;
  max-width: calc(100% - 740px);
  z-index: var(--z-dropdown);
}

.app-header__content {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 10px;
  background: var(--app-header-background);
  border-radius: 12px;
  box-shadow: 0 2px 10px var(--app-header-shadow);
}

.app-header__logo-container {
  display: flex;
  align-items: center;
}

.app-header__logo-link {
  display: flex;
  align-items: center;
  text-decoration: none;
  color: var(--color-text-heading);
}

.app-header__logo {
  height: 40px;
  width: auto;
  margin-right: 15px;
  border-radius: 8px;
}

.app-header__name {
  font-size: 1.5em;
  font-weight: bold;
  letter-spacing: 0;
}

.app-header__links {
  display: flex;
  align-items: center;
  gap: 15px;
}

.app-header__link {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 12px;
  background-color: var(--app-header-link-background);
  border-radius: 20px;
  color: var(--color-text-heading);
  text-decoration: none;
  transition: all 0.3s ease;
}

.app-header__link:hover {
  background-color: var(--app-header-link-hover-background);
  transform: translateY(-2px);
}

.app-header__github-icon {
  width: 20px;
  height: 20px;
  border-radius: 50%;
}

/* 赞助按钮样式 */
.app-header__link--donate {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 12px;
  background-color: var(--app-header-donate-background);
  border-radius: 20px;
  color: var(--app-header-donate-text);
  text-decoration: none;
  transition: all 0.3s ease;
}

.app-header__link--donate:hover {
  background-color: var(--app-header-donate-hover-background);
  transform: translateY(-2px);
}

/* 返回书架按钮样式 */
.app-header__back-link {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 14px;
  background: linear-gradient(135deg, var(--color-action-brand) 0%, var(--color-action-brand-strong) 100%);
  border-radius: 20px;
  color: white;
  text-decoration: none;
  font-size: 0.9em;
  font-weight: 500;
  transition: all 0.3s ease;
}

.app-header__back-link:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px var(--shadow-action-brand);
}

/* 保存按钮样式（顶部） */
.app-header__save-button {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px 14px;
  background: linear-gradient(135deg, var(--color-action-success) 0%, var(--color-action-success-strong) 100%);
  border: none;
  border-radius: 20px;
  color: white;
  font-size: 1em;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.app-header__save-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px var(--shadow-action-success);
}

.app-header__save-button:active {
  transform: translateY(0);
}

/* 设置按钮基础样式 */
.app-header__settings-button {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px 12px;
  background-color: var(--app-header-settings-background);
  border: none;
  border-radius: 20px;
  cursor: pointer;
  font-size: 1.1em;
  transition: all 0.3s ease;
}

.app-header__settings-button:hover {
  background-color: var(--app-header-settings-hover-background);
  transform: translateY(-2px);
}

/* 设置按钮高亮引导动画 */
.app-header__settings-button--highlight {
  animation: pulse-highlight 1.5s ease-in-out infinite;
}

.app-header__theme-toggle {
  background-color: var(--app-header-theme-background);
  border: 1px solid var(--app-header-panel-border);
  border-radius: 20px;
  cursor: pointer;
  padding: 6px 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 42px;
  transition: background-color 0.3s;
}

.app-header__theme-toggle:hover {
  background-color: var(--app-header-theme-hover-background);
}

.app-header__theme-icon {
  font-size: 16px;
}

/* ============ Insight 变体样式 ============ */

.app-header--insight {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 56px;
  background: var(--color-surface-subtle);
  border-bottom: 1px solid var(--color-border-muted);
  z-index: var(--z-dropdown);
  display: flex;
  align-items: center;
  padding: 0 20px;
  max-width: none;
  width: auto;
  margin: 0;
}

.app-header--insight .app-header__content {
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

.app-header--insight .app-header__logo-link {
  display: flex;
  align-items: center;
  gap: 10px;
  text-decoration: none;
  color: var(--color-text-default);
}

.app-header--insight .app-header__logo {
  height: 32px;
  width: auto;
  max-height: 32px;
  margin-right: 0;
}

.app-header--insight .app-header__name {
  font-weight: 600;
  font-size: 18px;
}

.app-header--insight .app-header__links {
  display: flex;
  align-items: center;
  gap: 16px;
}

/* ============ Bookshelf 变体样式 ============ */

.app-header--bookshelf {
  background: linear-gradient(135deg, var(--color-action-brand) 0%, var(--color-action-brand-strong) 100%);
  padding: 0 24px;
  height: 64px;
  box-shadow: 0 2px 20px var(--app-header-bookshelf-shadow);
  position: sticky;
  top: 0;
  z-index: var(--z-sticky);
  display: flex;
  align-items: center;
  max-width: none;
  width: 100%;
  margin: 0;
}

.app-header--bookshelf .app-header__content {
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

.app-header--bookshelf .app-header__logo-link {
  display: flex;
  align-items: center;
  gap: 12px;
  text-decoration: none;
  color: white;
}

.app-header--bookshelf .app-header__logo {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  box-shadow: 0 2px 8px var(--app-header-logo-shadow);
  margin-right: 0;
}

.app-header--bookshelf .app-header__name {
  font-size: 1.3rem;
  font-weight: 700;
  color: white;
  letter-spacing: 0;
}

.app-header--bookshelf .app-header__links {
  display: flex;
  align-items: center;
  gap: 16px;
}

@media (--breakpoint-md-down) {
  .app-header--default {
    width: 100%;
    max-width: none;
    padding: 8px 10px;
  }

  .app-header--default .app-header__content {
    flex-direction: column;
    flex-wrap: wrap;
    justify-content: center;
    gap: 12px;
    width: auto;
    background: transparent;
    box-shadow: none;
    padding: 9px 0 1px;
  }

  .app-header--default .app-header__logo-container {
    display: flex;
    flex: 1 1 100%;
    justify-content: center;
    min-width: 0;
  }

  .app-header--default .app-header__logo-link {
    justify-content: center;
    min-width: 0;
  }

  .app-header--default .app-header__logo {
    height: 30px;
    margin-right: 0;
  }

  .app-header--default .app-header__name {
    display: none;
  }

  .app-header--default .app-header__links {
    flex: 0 0 auto;
    flex-wrap: wrap;
    justify-content: center;
    width: 160px;
    gap: 10px 8px;
  }
}
</style>
