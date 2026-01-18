# 🗑️ 删除网页导入功能指南

> **注意**: 本文档指导如何移除"从网页导入漫画"功能的用户入口。

## 📋 目录
- [为什么删除](#为什么删除)
- [简化版删除方案（推荐）](#简化版删除方案推荐)
- [完整删除方案](#完整删除方案)
- [验证方法](#验证方法)

---

## 🎯 为什么删除

网页导入功能存在**高法律风险**：
- 涉及网页内容爬取，可能违反目标网站的服务条款
- 部分网站有反爬虫机制，爬取行为可能触犯法律
- 公开发布此功能可能带来法律责任

因此，在公开发布版本前，建议移除此功能的用户入口。

---

## ⚡ 简化版删除方案（推荐）

**只需删除前端入口**，后端代码保留不影响使用。

### 原理
- 用户无法通过界面看到或触发此功能
- 后端 API 虽然存在，但没有入口调用
- 代码保留方便日后恢复

### 操作方法

**文件**: `vue-frontend/src/components/translate/ImageUpload.vue`

#### 步骤 1: 删除导入语句

```typescript
// 删除这一行
import { useWebImportStore } from '@/stores/webImportStore'
```

#### 步骤 2: 删除 Store 声明

```typescript
// 删除这一行
const webImportStore = useWebImportStore()
```

#### 步骤 3: 删除函数定义

```typescript
// 删除这个函数
/**
 * 触发网页导入模态框
 */
function triggerWebImport() {
  webImportStore.openModal()
}
```

#### 步骤 4: 删除模板入口（在 `<template>` 中的上传区域）

```vue
<!-- 删除这 4 行 -->
<span class="separator"> | </span>
<span class="select-link web-import-link" @click="triggerWebImport">
  🌐 从网页导入
</span>
```

修改后的上传区域代码为：
```vue
<p class="drop-text">
  拖拽图片、PDF或MOBI文件到这里，或 
  <span class="select-link" @click="triggerFileSelect">
    选择文件
  </span>
  <span class="separator"> | </span>
  <span class="select-link folder-link" @click="triggerFolderSelect">
    📁 选择文件夹
  </span>
</p>
```

### 优点
| 项目 | 说明 |
|------|------|
| ✅ 工作量小 | 只修改 1 个文件，删除约 10 行代码 |
| ✅ 无编译错误 | 删除干净后无 lint 警告 |
| ✅ 易于恢复 | 恢复只需加回这几行 |
| ✅ 用户不可见 | 功能入口完全消失 |

### 预计耗时
**1 分钟**


---

## 🔧 完整删除方案

如果你想彻底清除所有相关代码，可以参考以下清单：

### 前端文件

```
vue-frontend/src/
├── components/translate/
│   ├── WebImportModal.vue          ❌ 删除
│   ├── WebImportButton.vue         ❌ 删除
│   └── WebImportDisclaimer.vue     ❌ 删除
├── api/webImport.ts                ❌ 删除
├── stores/webImportStore.ts        ❌ 删除
├── stores/settings/modules/webImport.ts  ❌ 删除
└── types/webImport.ts              ❌ 删除
```

**代码修改**：
- `components/translate/ImageUpload.vue` - 删除导入、store使用、函数和模板
- `views/TranslateView.vue` - 删除 WebImportDisclaimer 导入和使用

### 后端文件

```
src/
├── core/web_import/                ❌ 删除整个目录
└── app/api/
    ├── web_import_api.py           ❌ 删除
    └── __init__.py                 🔧 删除相关导入和蓝图
```

### 预计耗时
**5-7 分钟**

---

## ✅ 验证方法

### 简化版验证

1. **启动前端开发服务器**：
   ```powershell
   cd vue-frontend
   npm run dev
   ```

2. **检查页面**：
   - 访问 http://localhost:5173
   - 进入翻译页面
   - 确认"从网页导入"链接已消失
   - 其他功能正常（上传图片、选择文件夹等）

### 完整版验证

1. **前端编译检查**：
   ```powershell
   cd vue-frontend
   npm run build
   ```

2. **后端启动检查**：
   ```powershell
   .\venv\Scripts\activate
   python app.py
   ```

3. **全局搜索检查**：
   ```powershell
   # 前端
   Get-ChildItem -Path "vue-frontend\src" -Recurse -Include *.vue,*.ts | Select-String -Pattern "webImport|WebImport"
   
   # 后端
   Get-ChildItem -Path "src" -Recurse -Include *.py | Select-String -Pattern "web_import"
   ```

---

## 🎉 完成确认

删除完成后，你的项目应该：
- ✅ 翻译页面没有"从网页导入"选项
- ✅ 所有其他功能正常工作
- ✅ 可以安全发布

**恭喜！网页导入功能入口已移除，可以安全发布！** 🚀

---

**最后更新**: 2026-01-19
**文档版本**: 2.0 (简化版)
