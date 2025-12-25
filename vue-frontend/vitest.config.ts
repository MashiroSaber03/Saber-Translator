import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  test: {
    // 使用 jsdom 作为测试环境
    environment: 'jsdom',
    // 全局导入测试函数
    globals: true,
    // 测试文件匹配模式（包含属性测试）
    include: ['src/**/*.{test,spec}.{js,ts}', 'tests/**/*.{test,spec,property}.{js,ts}'],
    // 属性测试配置 - 每个属性测试运行100次迭代
    testTimeout: 30000,
    // 覆盖率配置
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: ['node_modules/', 'tests/'],
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
})
