# 🗑️ 删除网页导入功能指南

> **警告**: 本文档指导如何完全删除"从网页导入漫画"功能。此操作不可逆，请在删除前确认。

## 📋 目录
- [为什么删除](#为什么删除)
- [删除清单](#删除清单)
- [详细步骤](#详细步骤)
- [验证方法](#验证方法)
- [预计耗时](#预计耗时)

---

## 🎯 为什么删除

网页导入功能存在**高法律风险**：
- 涉及网页内容爬取，可能违反目标网站的服务条款
- 部分网站有反爬虫机制，爬取行为可能触犯法律
- 公开发布此功能可能带来法律责任

因此，在公开发布版本前，建议完全移除此功能。

---

## 📦 删除清单

### 前端文件 (5个文件 + 1处代码修改)

```
vue-frontend/src/
├── components/
│   └── translate/
│       ├── WebImportModal.vue          ❌ 删除 (1565行)
│       └── WebImportButton.vue         ❌ 删除 (57行)
├── api/
│   └── webImport.ts                    ❌ 删除 (177行)
├── stores/
│   ├── webImportStore.ts               ❌ 删除 (278行)
│   └── settings/modules/
│       └── webImport.ts                ❌ 删除 (300行)
└── types/
    └── webImport.ts                    ❌ 删除 (143行)
```

**代码修改**：
- `components/translate/ImageUpload.vue` - 删除4行

### 后端文件 (1个目录 + 1个文件 + 1处代码修改)

```
src/
├── core/
│   └── web_import/                     ❌ 删除整个目录 (7个文件)
│       ├── __init__.py
│       ├── agent.py
│       ├── firecrawl_tools.py
│       ├── gallery_dl_runner.py
│       ├── image_downloader.py
│       ├── image_processor.py
│       └── prompts.py
└── app/
    └── api/
        ├── web_import_api.py           ❌ 删除 (593行)
        └── __init__.py                 🔧 修改 (删除2行)
```

---

## 📝 详细步骤

### 步骤 1: 删除前端文件

#### 1.1 删除核心组件文件

在项目根目录打开 PowerShell，执行：

```powershell
# 删除网页导入模态框
Remove-Item -Path "vue-frontend\src\components\translate\WebImportModal.vue" -Force

# 删除网页导入按钮组件
Remove-Item -Path "vue-frontend\src\components\translate\WebImportButton.vue" -Force

# 删除网页导入API
Remove-Item -Path "vue-frontend\src\api\webImport.ts" -Force

# 删除网页导入Store
Remove-Item -Path "vue-frontend\src\stores\webImportStore.ts" -Force

# 删除网页导入设置模块
Remove-Item -Path "vue-frontend\src\stores\settings\modules\webImport.ts" -Force

# 删除网页导入类型定义
Remove-Item -Path "vue-frontend\src\types\webImport.ts" -Force
```

#### 1.2 修改 ImageUpload.vue

**文件**: `vue-frontend/src/components/translate/ImageUpload.vue`

**删除第19行**：
```typescript
- import { useWebImportStore } from '@/stores/webImportStore'
```

**删除第45行**：
```typescript
- const webImportStore = useWebImportStore()
```

**删除第93-95行**：
```typescript
- function triggerWebImport() {
-   webImportStore.openModal()
- }
```

**删除第529-531行**（在模板中）：
```vue
- <span class="select-link web-import-link" @click="triggerWebImport">
-   🌐 从网页导入
- </span>
```

**并删除第528行的分隔符**：
```vue
- <span class="separator">| </span>
```

修改后，第524-527行应该是：
```vue
<p class="drop-text">
  拖拽图片、PDF或MOBI文件到这里，或 
  <span class="select-link" @click="triggerFileSelect">
    点击选择文件
  </span>
</p>
```

---

### 步骤 2: 删除后端文件

#### 2.1 删除核心模块目录

在项目根目录打开 PowerShell，执行：

```powershell
# 删除整个 web_import 核心模块目录
Remove-Item -Path "src\core\web_import" -Recurse -Force

# 删除 web_import API 路由文件
Remove-Item -Path "src\app\api\web_import_api.py" -Force
```

#### 2.2 修改 API __init__.py

**文件**: `src/app/api/__init__.py`

**删除第18行**：
```python
- from .web_import_api import web_import_bp  # ✨ 网页漫画导入 API
```

**修改第21行**，从：
```python
all_blueprints = [translate_bp, config_bp, system_bp, session_bp, bookshelf_bp, manga_insight_bp, web_import_bp]
```

改为：
```python
all_blueprints = [translate_bp, config_bp, system_bp, session_bp, bookshelf_bp, manga_insight_bp]
```

---

### 步骤 3: 清理常量定义（可选）

虽然不会影响功能，但为了代码整洁，可以删除相关常量定义。

**文件**: `vue-frontend/src/constants/index.ts`

**删除第328-373行**（网页导入部分）：
```typescript
// ============================================================
// 网页导入常量
// ============================================================

/** 网页导入设置存储键 */
export const STORAGE_KEY_WEB_IMPORT_SETTINGS = 'webImportSettings'

/** 网页导入默认提取提示词 */
export const DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT = `...`

/**
 * 网页导入 AI Agent 服务商列表
 */
export const WEB_IMPORT_AGENT_PROVIDERS = [...]
```

---

## ✅ 验证方法

### 验证 1: 检查编译错误

删除后，前端和后端服务应该能正常启动：

```powershell
# 前端
cd vue-frontend
npm run dev

# 后端
cd ..
.\.venv\Scripts\activate
python app.py
```

如果有错误，检查是否还有遗漏的引用。

### 验证 2: 检查残留引用

使用全局搜索检查是否还有引用：

```powershell
# 搜索 webImport 相关引用
cd vue-frontend\src
Get-ChildItem -Recurse -Include *.vue,*.ts | Select-String -Pattern "webImport|WebImport"

# 搜索 web_import 相关引用（后端）
cd ..\..\src
Get-ChildItem -Recurse -Include *.py | Select-String -Pattern "web_import"
```

应该没有任何结果（除了 `constants/index.ts` 如果你未清理常量）。

### 验证 3: 功能测试

1. **前端测试**：
   - 访问 http://localhost:5173
   - 进入翻译页面
   - 确认"从网页导入"链接已消失
   - 其他功能正常（上传图片、翻译等）

2. **后端测试**：
   - 访问 http://localhost:5000/api/
   - 确认 `/api/web-import/*` 路由不存在
   - 其他 API 正常工作

### 验证 4: 构建测试

测试生产构建是否成功：

```powershell
# 前端构建
cd vue-frontend
npm run build

# 后端打包（如果使用 PyInstaller）
cd ..
pyinstaller app.spec
```

构建应该成功且无警告。

---

## ⏱️ 预计耗时

| 步骤 | 耗时 | 难度 |
|------|------|------|
| 删除前端文件 | 1分钟 | ⭐ 简单 |
| 修改 ImageUpload.vue | 2分钟 | ⭐⭐ 中等 |
| 删除后端文件 | 1分钟 | ⭐ 简单 |
| 修改 API __init__.py | 1分钟 | ⭐ 简单 |
| 验证与测试 | 2分钟 | ⭐⭐ 中等 |
| **总计** | **约5-7分钟** | ⭐⭐ 中等 |

---

## 🔄 快速删除脚本

如果你想一键删除（需谨慎），可以使用以下 PowerShell 脚本：

**创建文件**: `remove_web_import.ps1`

```powershell
# 网页导入功能快速删除脚本
# 警告：此操作不可逆！

Write-Host "⚠️  警告：即将删除网页导入功能" -ForegroundColor Yellow
Write-Host "此操作不可逆，确定继续吗？(y/n)" -ForegroundColor Yellow
$confirm = Read-Host

if ($confirm -ne 'y') {
    Write-Host "操作已取消" -ForegroundColor Green
    exit
}

Write-Host "`n🗑️  开始删除..." -ForegroundColor Cyan

# 删除前端文件
Write-Host "`n[1/3] 删除前端文件..." -ForegroundColor Yellow
Remove-Item -Path "vue-frontend\src\components\translate\WebImportModal.vue" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "vue-frontend\src\components\translate\WebImportButton.vue" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "vue-frontend\src\api\webImport.ts" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "vue-frontend\src\stores\webImportStore.ts" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "vue-frontend\src\stores\settings\modules\webImport.ts" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "vue-frontend\src\types\webImport.ts" -Force -ErrorAction SilentlyContinue

Write-Host "✅ 前端文件已删除" -ForegroundColor Green

# 删除后端文件
Write-Host "`n[2/3] 删除后端文件..." -ForegroundColor Yellow
Remove-Item -Path "src\core\web_import" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "src\app\api\web_import_api.py" -Force -ErrorAction SilentlyContinue

Write-Host "✅ 后端文件已删除" -ForegroundColor Green

# 提示手动修改
Write-Host "`n[3/3] 需要手动修改的文件：" -ForegroundColor Yellow
Write-Host "  1. vue-frontend\src\components\translate\ImageUpload.vue" -ForegroundColor Cyan
Write-Host "     - 删除第19行: import { useWebImportStore }..." -ForegroundColor Gray
Write-Host "     - 删除第45行: const webImportStore = ..." -ForegroundColor Gray
Write-Host "     - 删除第93-95行: function triggerWebImport() {...}" -ForegroundColor Gray
Write-Host "     - 删除第528-531行: 网页导入链接" -ForegroundColor Gray
Write-Host "`n  2. src\app\api\__init__.py" -ForegroundColor Cyan
Write-Host "     - 删除第18行: from .web_import_api import web_import_bp" -ForegroundColor Gray
Write-Host "     - 修改第21行: 从 all_blueprints 中删除 web_import_bp" -ForegroundColor Gray

Write-Host "`n✅ 自动删除完成！请手动完成上述代码修改。" -ForegroundColor Green
Write-Host "📝 建议执行验证步骤确保删除完整。" -ForegroundColor Yellow
```

**使用方法**：
```powershell
# 在项目根目录执行
.\remove_web_import.ps1
```

---

## 📌 注意事项

### ⚠️ 删除前的准备

1. **备份代码**：建议在 Git 中创建分支
   ```bash
   git checkout -b remove-web-import
   git commit -am "Backup before removing web import feature"
   ```

2. **确认依赖**：确保没有其他功能依赖网页导入

3. **通知团队**：如果是团队项目，提前通知成员

### ⚠️ 删除后的影响

1. **用户数据**：localStorage 中可能仍有 `webImportSettings`，但不影响使用
2. **后端依赖**：如果安装了 `gallery-dl`，可以选择性卸载：
   ```bash
   pip uninstall gallery-dl
   ```
3. **API Key**：Firecrawl 和 AI Agent 的 API Key 会从设置中消失（数据在 localStorage）

### 🔧 可选清理

如果想彻底清理，可以：
1. 清理 localStorage：
   ```javascript
   // 在浏览器控制台执行
   localStorage.removeItem('webImportSettings')
   ```

2. 删除可能残留的临时文件：
   ```powershell
   Remove-Item -Path "data\temp\gallery_dl" -Recurse -Force -ErrorAction SilentlyContinue
   Remove-Item -Path "data\temp\gallery_dl_download" -Recurse -Force -ErrorAction SilentlyContinue
   ```

---

## 🎉 完成确认

删除完成后，你的项目应该：
- ✅ 前端正常编译，无错误
- ✅ 后端正常启动，无导入错误
- ✅ 翻译页面没有"从网页导入"选项
- ✅ 所有其他功能正常工作
- ✅ 生产构建成功

**恭喜！网页导入功能已完全移除，可以安全发布！** 🚀

---

## 📞 问题排查

### 问题1: 前端编译错误 "Cannot find module '@/api/webImport'"

**原因**: 仍有文件引用了已删除的模块

**解决**: 全局搜索 `webImport` 或 `WebImport`，删除所有引用

### 问题2: 后端启动错误 "No module named 'web_import'"

**原因**: API __init__.py 仍在导入 web_import_bp

**解决**: 检查 `src/app/api/__init__.py`，确保已删除相关导入

### 问题3: 页面显示空白或布局错乱

**原因**: ImageUpload.vue 删除代码时影响了布局

**解决**: 确保删除后的代码闭合标签完整，检查第524-527行

---

## 📚 相关文档

- [项目架构文档](./docs/ARCHITECTURE.md)（如果有）
- [网页导入功能分析](./docs/WEB_IMPORT_ANALYSIS.md)（如果需要保存分析）

---

**最后更新**: 2026-01-03
**文档版本**: 1.0
**适用版本**: Saber-Translator (整合后版本)
