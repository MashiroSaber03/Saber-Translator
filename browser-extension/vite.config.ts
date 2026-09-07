import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'node:path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: { '@': resolve(import.meta.dirname, '../vue-frontend/src') },
    dedupe: ['vue'],
  },
  css: { postcss: resolve(import.meta.dirname, '../vue-frontend') },
  build: { target: 'chrome120' },
})
