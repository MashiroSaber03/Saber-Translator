/// <reference types="vite/client" />

/**
 * Vue 单文件组件类型声明
 * 让 TypeScript 能够识别 .vue 文件的导入
 */
declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<object, object, unknown>
  export default component
}
