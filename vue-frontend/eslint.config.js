import js from '@eslint/js'
import pluginVue from 'eslint-plugin-vue'
import tseslint from 'typescript-eslint'
import globals from 'globals'

export default [
  // 基础 JavaScript 规则
  js.configs.recommended,
  
  // TypeScript 规则
  ...tseslint.configs.recommended,
  
  // Vue 规则
  ...pluginVue.configs['flat/recommended'],
  
  // 自定义配置
  {
    files: ['**/*.{ts,vue}'],
    languageOptions: {
      parserOptions: {
        parser: tseslint.parser,
        ecmaVersion: 'latest',
        sourceType: 'module',
      },
      // 添加浏览器全局变量
      globals: {
        ...globals.browser,
        ...globals.es2021,
      },
    },
    rules: {
      // Vue 规则
      'vue/multi-word-component-names': 'off', // 允许单词组件名
      'vue/no-v-html': 'off', // 允许 v-html
      'vue/max-attributes-per-line': 'off', // 关闭属性换行限制
      'vue/singleline-html-element-content-newline': 'off', // 关闭单行元素内容换行
      'vue/html-self-closing': 'off', // 关闭自闭合标签检查
      'vue/attributes-order': 'off', // 关闭属性顺序检查
      'vue/require-default-prop': 'off', // 关闭 prop 默认值检查
      
      // TypeScript 规则
      '@typescript-eslint/no-unused-vars': ['warn', { 
        argsIgnorePattern: '^_',
        varsIgnorePattern: '^_',
        caughtErrorsIgnorePattern: '^_|^error$'  // 允许 catch 块中的 error 变量
      }],
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/no-explicit-any': 'off',  // 迁移项目中允许 any
      '@typescript-eslint/ban-ts-comment': 'off', // 允许 @ts-ignore
      
      // 通用规则 - 开发阶段允许 console，生产构建时由 Vite 移除
      'no-console': 'off',
      'no-debugger': 'warn',
    },
  },
  
  // 测试文件配置
  {
    files: ['tests/**/*.{ts,js}'],
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.es2021,
        ...globals.node,
      },
    },
  },
  
  // 忽略文件
  {
    ignores: ['dist/**', 'node_modules/**', '*.config.js', '*.config.ts'],
  },
]
