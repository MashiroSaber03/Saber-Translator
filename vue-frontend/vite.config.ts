import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  
  // 路径别名配置
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  
  // 构建配置 - 输出到 Flask 静态目录
  build: {
    outDir: '../src/app/static/vue',
    emptyOutDir: true,
    // 生成 sourcemap 便于调试
    sourcemap: false,
    // 代码分割配置
    rollupOptions: {
      output: {
        entryFileNames: 'js/[name].[hash].js',
        chunkFileNames: 'js/[name].[hash].js',
        assetFileNames: 'assets/[name].[hash].[ext]',
        // 手动分割代码块，优化加载性能
        manualChunks: {
          // Vue 核心库
          'vue-vendor': ['vue', 'vue-router', 'pinia'],
          // 工具库
          'utils-vendor': ['axios'],
        },
      },
    },
    // 设置 chunk 大小警告阈值
    chunkSizeWarningLimit: 1000,
  },
  
  // 基础路径 - 用于 Flask 静态文件服务
  base: '/static/vue/',
  
  // 开发服务器配置
  server: {
    port: 5173,
    // 允许局域网访问
    host: true,
    // API 请求代理到 Flask 后端
    proxy: {
      // 所有 API 请求代理到 Flask
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      // 注意：CSS文件从public目录加载，不需要代理
      // '/css' 已移除，使用 public/css 目录
      '/fonts': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      // 图片资源代理
      '/pic': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
    },
  },
  
  // 预览服务器配置（用于预览构建结果）
  preview: {
    port: 4173,
    host: true,
  },
})
