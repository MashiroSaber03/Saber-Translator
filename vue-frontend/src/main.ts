/**
 * Vue 应用入口文件
 * 初始化 Vue 应用、路由和状态管理
 */
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'

// 引入全局样式 - token 层按依赖顺序显式加载
import './styles/tokens/palette.css'
import './styles/tokens/semantic.css'
import './styles/tokens/component.css'
import './styles/tokens/domain.css'
import './styles/reset.css'
import './styles/animations.css'
import './styles/base.css'

// 创建 Vue 应用实例
const app = createApp(App)

// 安装 Pinia 状态管理
const pinia = createPinia()
app.use(pinia)

// 安装 Vue Router
app.use(router)

// 全局错误处理
app.config.errorHandler = (err, _instance, info) => {
  console.error('Vue 错误:', err, info)
}

// 挂载应用
app.mount('#app')
