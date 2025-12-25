/**
 * Vue 应用入口文件
 * 初始化 Vue 应用、路由和状态管理
 */
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'

// 引入全局样式 - 仅CSS变量和基础reset
import './styles/global.css'

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
  // TODO: 集成全局错误提示组件
}

// 挂载应用
app.mount('#app')
