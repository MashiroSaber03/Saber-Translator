# 页面、组件与前端生产文件映射

> 最终状态已按 `current-files.tsv` 同步；审查中删除的冗余文件保留为“删除已验证”。行数按最终工作区重新计算。

本文件由当前源码导入关系生成，用于证明页面审查覆盖闭包。路径均相对 `vue-frontend/src`；测试文件由 `current-files.tsv` 单独覆盖。

## 路由与根入口

| 所有者 | 根文件 | 直接组件 |
|---|---|---|
| 启动与路由 | `main.ts + router/index.ts` | - |
| FIXED | `App.vue` | `components/common/ToastNotification.vue`<br>`components/product/ProductConfirmProvider.vue`<br>`components/product/ProductStatusBanner.vue`<br>`components/product/ProductTextInputProvider.vue`<br>`components/task-center/TaskCenterDrawer.vue`<br>`components/task-center/TaskCenterLauncher.vue`<br>`components/ui/OverlayLayer.vue`<br>`components/ui/UiButton.vue` |
| FIXED | `views/BookshelfView.vue` | `components/bookshelf/BookCard.vue`<br>`components/bookshelf/BookDetailModal.vue`<br>`components/bookshelf/BookModal.vue`<br>`components/bookshelf/BookSearch.vue`<br>`components/bookshelf/TagManageModal.vue`<br>`components/common/BaseModal.vue`<br>`components/product/ProductActionRow.vue`<br>`components/product/ProductCardGrid.vue`<br>`components/product/ProductEmptyState.vue`<br>`components/product/ProductHeaderAction.vue`<br>`components/product/ProductHeaderMetaPill.vue`<br>`components/product/ProductPageHeader.vue`<br>`components/product/ProductThemeToggle.vue`<br>`components/ui/AppShell.vue`<br>`components/ui/selectTypes.ts`<br>`components/ui/UiButton.vue`<br>`components/ui/UiCheckbox.vue`<br>`components/ui/UiIcon.vue`<br>`components/ui/UiSelect.vue` |
| FIXED | `views/TranslateView.vue` | `components/bookshelf/SponsorModal.vue`<br>`components/edit/EditWorkspace.vue`<br>`components/product/ProductHeaderAction.vue`<br>`components/product/ProductPageHeader.vue`<br>`components/product/ProductThemeToggle.vue`<br>`components/settings/SettingsModal.vue`<br>`components/translate/BookGlossaryModal.vue`<br>`components/translate/BookNonTranslateModal.vue`<br>`components/translate/FirstTimeGuide.vue`<br>`components/translate/ImageResultDisplay.vue`<br>`components/translate/ImageUpload.vue`<br>`components/translate/QuickWorkspacePromoteModal.vue`<br>`components/translate/SettingsSidebar.vue`<br>`components/translate/ThumbnailSidebar.vue`<br>`components/translate/TranslationProgress.vue`<br>`components/translate/WebImportDisclaimer.vue`<br>`components/translate/WebImportModal.vue`<br>`components/ui/AppShell.vue`<br>`components/ui/SidebarLayout.vue` |
| FIXED | `views/ReaderView.vue` | `components/product/ProductHeaderAction.vue`<br>`components/product/ProductPageHeader.vue`<br>`components/reader/ReaderCanvas.vue`<br>`components/reader/ReaderControls.vue`<br>`components/ui/AppShell.vue` |
| FIXED | `views/InsightView.vue` | `components/insight/AnalysisProgress.vue`<br>`components/insight/BookSelector.vue`<br>`components/insight/ChapterSelectModal.vue`<br>`components/insight/CharacterStudioEntryPanel.vue`<br>`components/insight/ContinuationPanel.vue`<br>`components/insight/InsightSettingsModal.vue`<br>`components/insight/NotesPanel.vue`<br>`components/insight/OverviewPanel.vue`<br>`components/insight/PageDetail.vue`<br>`components/insight/PagesTree.vue`<br>`components/insight/QAPanel.vue`<br>`components/insight/TimelinePanel.vue`<br>`components/product/ProductHeaderAction.vue`<br>`components/product/ProductPageHeader.vue`<br>`components/product/ProductTabbedWorkspace.vue`<br>`components/product/ProductThemeToggle.vue`<br>`components/product/ProductThreePaneWorkspace.vue`<br>`components/ui/AppShell.vue`<br>`components/ui/iconRegistry.ts`<br>`components/ui/UiIcon.vue`<br>`components/ui/UiIconButton.vue` |
| FIXED | `views/CharacterStudioView.vue` | `components/insight/studio/CharacterStudioEditor.vue`<br>`components/insight/studio/CharacterStudioPreview.vue`<br>`components/insight/studio/CharacterStudioSidebar.vue`<br>`components/insight/studio/StudioTopbar.vue`<br>`components/product/ProductEmptyState.vue`<br>`components/product/ProductSplitWorkspace.vue`<br>`components/product/ProductStatusBanner.vue`<br>`components/ui/AppShell.vue`<br>`components/ui/UiButton.vue` |

## 文件归属闭包

| 状态 | 文件 | 行数 | 类型 | 可达所有者 |
|---|---|---:|---|---|
| FIXED | `adapters/v2ContentAdapter.ts` | 324 | support | 翻译 |
| FIXED | `api/bookshelf.ts` | 348 | api | 书架、全局应用壳、漫画分析、角色工坊、翻译 |
| FIXED | `api/characterStudio.ts` | 1095 | api | 角色工坊 |
| FIXED | `api/client.ts` | 135 | api | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `api/continuation.ts` | 709 | api | 漫画分析 |
| PASS | `api/download.ts` | 77 | api | 漫画分析、角色工坊、翻译 |
| GENERATED-VERIFIED | `api/generated/v2.ts` | 9795 | api | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `api/insight.ts` | 1993 | api | 漫画分析 |
| FIXED | `api/plugin.ts` | 193 | api | 翻译 |
| FIXED | `api/pluginAgent.ts` | 309 | api | 翻译 |
| FIXED | `api/sse.ts` | 82 | api | 漫画分析、角色工坊、翻译；严格 SSE 行解析、尾字节与 EOF 事件闭合 |
| PASS | `api/v2/content.ts` | 384 | api | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `api/v2/continuation.ts` | 242 | api | 漫画分析 |
| PASS | `api/v2/diagnostics.ts` | 104 | api | 漫画分析、翻译 |
| FIXED | `api/v2/insight.ts` | 221 | api | 全局应用壳、漫画分析 |
| FIXED | `api/v2/jobs.ts` | 183 | api | 书架、全局应用壳、漫画分析、翻译 |
| PASS | `api/v2/operations.ts` | 424 | api | 角色工坊、翻译 |
| FIXED | `api/v2/settings.ts` | 152 | api | 书架、全局应用壳、漫画分析、翻译 |
| PASS | `api/v2/studio.ts` | 297 | api | 角色工坊 |
| PASS | `api/v2/system.ts` | 15 | api | 书架、全局应用壳 |
| PASS | `api/v2/translation.ts` | 133 | api | 书架、翻译 |
| FIXED | `api/v2/webImport.ts` | 88 | api | 翻译 |
| FIXED | `App.vue` | 96 | support | 全局应用壳；设置、Provider、任务中心与阅读器路由生命周期闭合 |
| PASS | `components/bookshelf/book-detail/BookDeleteConfirmContent.vue` | 15 | component | 书架 |
| FIXED | `components/bookshelf/book-detail/BookDetailSummary.vue` | 212 | component | 书架 |
| PASS | `components/bookshelf/book-detail/ChapterFormContent.vue` | 41 | component | 书架 |
| FIXED | `components/bookshelf/book-detail/ChapterList.vue` | 149 | component | 书架 |
| FIXED | `components/bookshelf/book-detail/ChapterRow.vue` | 223 | component | 书架 |
| PASS | `components/bookshelf/book-detail/QuickTagPicker.vue` | 161 | component | 书架 |
| PASS | `components/bookshelf/BookCard.vue` | 261 | component | 书架 |
| FIXED | `components/bookshelf/BookDetailModal.vue` | 537 | component | 书架 |
| FIXED | `components/bookshelf/BookModal.vue` | 351 | component | 书架 |
| FIXED | `components/bookshelf/BookSearch.vue` | 157 | component | 书架 |
| FIXED | `components/bookshelf/SponsorModal.vue` | 110 | component | 翻译 |
| FIXED | `components/bookshelf/TagManageModal.vue` | 424 | component | 书架 |
| FIXED | `components/common/BaseModal.vue` | 482 | component | 书架、全局应用壳、漫画分析、角色工坊、翻译 |
| PASS | `components/common/ConfirmModal.vue` | 72 | component | 全局应用壳 |
| FIXED | `components/common/OpenAIExtraBodyEditor.vue` | 161 | component | 漫画分析、翻译 |
| FIXED | `components/common/ToastNotification.vue` | 145 | component | 全局应用壳 |
| PASS | `components/edit/BubbleEditor.vue` | 865 | component | 翻译 |
| PASS | `components/edit/BubbleOverlay.vue` | 542 | component | 翻译 |
| PASS | `components/edit/bubbleOverlayGeometry.ts` | 66 | component | 翻译 |
| PASS | `components/edit/EditImageComparison.vue` | 595 | component | 翻译 |
| PASS | `components/edit/EditThumbnailPanel.vue` | 160 | component | 翻译 |
| PASS | `components/edit/EditToolbar.vue` | 626 | component | 翻译 |
| PASS | `components/edit/EditToolbarHelp.vue` | 193 | component | 翻译 |
| PASS | `components/edit/EditWorkspace.vue` | 208 | component | 翻译 |
| PASS | `components/edit/JapaneseKeyboard.vue` | 562 | component | 翻译 |
| FIXED | `components/edit/useBubbleEditor.ts` | 485 | component | 翻译 |
| PASS | `components/edit/useBubbleOverlayInteractionState.ts` | 78 | component | 翻译 |
| PASS | `components/edit/useEditWorkspace.ts` | 966 | component | 翻译 |
| FIXED | `components/insight/AnalysisProgress.vue` | 764 | component | 漫画分析 |
| PASS | `components/insight/BookSelector.vue` | 97 | component | 漫画分析 |
| PASS | `components/insight/ChapterSelectModal.vue` | 161 | component | 漫画分析 |
| PASS | `components/insight/CharacterStudioEntryPanel.vue` | 109 | component | 漫画分析 |
| FIXED | `components/insight/continuation/AddCharacterDialog.vue` | 104 | component | 漫画分析 |
| FIXED | `components/insight/continuation/AddFormDialog.vue` | 88 | component | 漫画分析 |
| PASS | `components/insight/continuation/CharacterDetailPanel.vue` | 188 | component | 漫画分析 |
| FIXED | `components/insight/continuation/CharacterManagementPanel.vue` | 613 | component | 漫画分析 |
| PASS | `components/insight/continuation/ContinuationDialogActions.vue` | 9 | component | 漫画分析 |
| PASS | `components/insight/continuation/ContinuationDialogField.vue` | 37 | component | 漫画分析 |
| PASS | `components/insight/continuation/ContinuationDialogForm.vue` | 13 | component | 漫画分析 |
| FIXED | `components/insight/continuation/ContinuationDialogShell.vue` | 53 | component | 漫画分析 |
| FIXED | `components/insight/continuation/EditCharacterDialog.vue` | 95 | component | 漫画分析 |
| FIXED | `components/insight/continuation/EditFormDialog.vue` | 91 | component | 漫画分析 |
| FIXED | `components/insight/continuation/ExportPanel.vue` | 184 | component | 漫画分析 |
| PASS | `components/insight/continuation/FormTile.vue` | 246 | component | 漫画分析 |
| FIXED | `components/insight/continuation/ImageGenerationPanel.vue` | 718 | component | 漫画分析 |
| FIXED | `components/insight/continuation/OrthographicDialog.vue` | 315 | component | 漫画分析 |
| FIXED | `components/insight/continuation/PageDetailsPanel.vue` | 249 | component | 漫画分析…12414 tokens truncated…壳、漫画分析、角色工坊、翻译 |
| FIXED | `stores/bookshelfStore.ts` | 496 | store | 书架、全局应用壳、漫画分析、角色工坊、翻译 |
| PASS | `components/insight/continuation/pageStoryTypes.ts` | 3 | component | 漫画分析 |
| FIXED | `components/insight/continuation/ReferenceImageSelector.vue` | 457 | component | 漫画分析 |
| FIXED | `components/insight/continuation/ScriptGenerationPanel.vue` | 323 | component | 漫画分析 |
| FIXED | `components/insight/ContinuationPanel.vue` | 855 | component | 漫画分析 |
| FIXED | `components/insight/InsightSettingsModal.vue` | 373 | component | 漫画分析 |
| PASS | `components/insight/notes/NoteCard.vue` | 217 | component | 漫画分析 |
| PASS | `components/insight/notes/NoteEditorModal.vue` | 230 | component | 漫画分析 |
| PASS | `components/insight/notes/NotesList.vue` | 73 | component | 漫画分析 |
| PASS | `components/insight/notes/NotesToolbar.vue` | 28 | component | 漫画分析 |
| PASS | `components/insight/NotesPanel.vue` | 306 | component | 漫画分析 |
| PASS | `components/insight/OverviewPanel.vue` | 725 | component | 漫画分析 |
| PASS | `components/insight/PageDetail.vue` | 892 | component | 漫画分析 |
| PASS | `components/insight/PagesTree.vue` | 492 | component | 漫画分析 |
| PASS | `components/insight/qa/EmbeddingRebuildControl.vue` | 33 | component | 漫画分析 |
| PASS | `components/insight/qa/QAComposer.vue` | 39 | component | 漫画分析 |
| PASS | `components/insight/qa/QAMessageItem.vue` | 154 | component | 漫画分析 |
| PASS | `components/insight/qa/QAMessageList.vue` | 94 | component | 漫画分析 |
| PASS | `components/insight/qa/QAOptionsBar.vue` | 251 | component | 漫画分析 |
| PASS | `components/insight/qa/QASaveNoteModal.vue` | 144 | component | 漫画分析 |
| PASS | `components/insight/QAPanel.vue` | 718 | component | 漫画分析 |
| FIXED | `components/insight/settings/BatchSettingsTab.vue` | 367 | component | 漫画分析 |
| FIXED | `components/insight/settings/EmbeddingSettingsTab.vue` | 234 | component | 漫画分析 |
| FIXED | `components/insight/settings/ImageGenSettingsTab.vue` | 132 | component | 漫画分析 |
| PASS | `components/insight/settings/InsightModelProviderSection.vue` | 182 | component | 漫画分析 |
| PASS | `components/insight/settings/InsightSettingsPanel.vue` | 40 | component | 漫画分析 |
| FIXED | `components/insight/settings/LlmSettingsTab.vue` | 237 | component | 漫画分析 |
| FIXED | `components/insight/settings/PromptsSettingsTab.vue` | 568 | component | 漫画分析 |
| FIXED | `components/insight/settings/RerankerSettingsTab.vue` | 216 | component | 漫画分析 |
| FIXED | `components/insight/settings/types.ts` | 74 | component | 漫画分析 |
| FIXED | `components/insight/settings/useInsightModelFetch.ts` | 95 | component | 漫画分析 |
| PASS | `components/insight/settings/useInsightSettingsDraft.ts` | 36 | component | 漫画分析 |
| FIXED | `components/insight/settings/VlmSettingsTab.vue` | 285 | component | 漫画分析 |
| PASS | `components/insight/studio/CandidateListPane.vue` | 138 | component | 角色工坊 |
| FIXED | `components/insight/studio/CharacterStudioEditor.vue` | 763 | component | 角色工坊 |
| PASS | `components/insight/studio/characterStudioEditorConfig.ts` | 36 | component | 角色工坊 |
| FIXED | `components/insight/studio/CharacterStudioPreview.vue` | 299 | component | 角色工坊 |
| PASS | `components/insight/studio/characterStudioPreviewHelpers.ts` | 34 | component | 角色工坊 |
| FIXED | `components/insight/studio/CharacterStudioPreviewModals.vue` | 233 | component | 角色工坊 |
| FIXED | `components/insight/studio/CharacterStudioSidebar.vue` | 239 | component | 角色工坊 |
| FIXED | `components/insight/studio/DiagnosticsPanel.vue` | 160 | component | 角色工坊 |
| PASS | `components/insight/studio/DocumentListPane.vue` | 189 | component | 角色工坊 |
| PASS | `components/insight/studio/editor/StudioEditorSectionPanel.vue` | 56 | component | 角色工坊 |
| PASS | `components/insight/studio/editor/StudioHeroSection.vue` | 150 | component | 角色工坊 |
| FIXED | `components/insight/studio/editor/StudioOverviewTab.vue` | 386 | component | 角色工坊 |
| PASS | `components/insight/studio/GreetingWorkbench.vue` | 186 | component | 角色工坊 |
| FIXED | `components/insight/studio/LorebookTreeBranch.vue` | 397 | component | 角色工坊 |
| FIXED | `components/insight/studio/LorebookTreeEditor.vue` | 172 | component | 角色工坊 |
| FIXED | `components/insight/studio/preview/AgentWorkspace.vue` | 309 | component | 角色工坊 |
| FIXED | `components/insight/studio/preview/ChatComposer.vue` | 268 | component | 角色工坊 |
| FIXED | `components/insight/studio/preview/ChatWorkspace.vue` | 171 | component | 角色工坊 |
| PASS | `components/insight/studio/preview/MessageList.vue` | 303 | component | 角色工坊 |
| PASS | `components/insight/studio/preview/RuntimeWorkspace.vue` | 142 | component | 角色工坊 |
| FIXED | `components/insight/studio/preview/SessionToolbar.vue` | 484 | component | 角色工坊 |
| PASS | `components/insight/studio/preview/StudioPreviewWorkspaceHeader.vue` | 48 | component | 角色工坊 |
| PASS | `components/insight/studio/preview/StudioPreviewWorkspacePanel.vue` | 24 | component | 角色工坊 |
| PASS | `components/insight/studio/RegexWorkbench.vue` | 195 | component | 角色工坊 |
| PASS | `components/insight/studio/StudioTopbar.vue` | 219 | component | 角色工坊 |
| PASS | `components/insight/studio/TaskWorkbench.vue` | 203 | component | 角色工坊 |
| PASS | `components/insight/timeline/PlotThreadsList.vue` | 101 | component | 漫画分析 |
| PASS | `components/insight/timeline/TimelineArcCard.vue` | 61 | component | 漫画分析 |
| PASS | `components/insight/timeline/TimelineCharacterGrid.vue` | 101 | component | 漫画分析 |
| PASS | `components/insight/timeline/TimelineEventCardShell.vue` | 209 | component | 漫画分析 |
| PASS | `components/insight/timeline/TimelineGroupCard.vue` | 32 | component | 漫画分析 |
| PASS | `components/insight/timeline/TimelineHeader.vue` | 56 | component | 漫画分析 |
| PASS | `components/insight/timeline/TimelineStats.vue` | 74 | component | 漫画分析 |
| PASS | `components/insight/timeline/TimelineSummaryCard.vue` | 46 | component | 漫画分析 |
| PASS | `components/insight/timeline/TimelineTrack.vue` | 136 | component | 漫画分析 |
| FIXED | `components/insight/timeline/useTimelinePanel.ts` | 243 | component | 漫画分析 |
| PASS | `components/insight/TimelinePanel.vue` | 240 | component | 漫画分析 |
| FIXED | `components/insight/useQANoteModal.ts` | 114 | component | 漫画分析 |
| PASS | `components/product/ProductActionRow.vue` | 119 | component | 书架、全局应用壳、漫画分析、角色工坊、翻译 |
| PASS | `components/product/ProductAvatar.vue` | 101 | component | 漫画分析、角色工坊 |
| PASS | `components/product/ProductBookSelector.vue` | 73 | component | 漫画分析 |
| PASS | `components/product/ProductBreadcrumbTrail.vue` | 102 | component | 翻译 |
| PASS | `components/product/ProductCardGrid.vue` | 38 | component | 书架 |
| PASS | `components/product/ProductChipList.vue` | 201 | component | 书架、漫画分析、角色工坊、翻译 |
| PASS | `components/product/ProductChoiceCardGrid.vue` | 194 | component | 漫画分析、角色工坊 |
| PASS | `components/product/productClassTypes.ts` | 1 | component | 漫画分析 |
| PASS | `components/product/ProductCollapsibleSection.vue` | 147 | component | 翻译 |
| PASS | `components/product/ProductComposer.vue` | 110 | component | 漫画分析 |
| PASS | `components/product/ProductConfirmProvider.vue` | 23 | component | 全局应用壳 |
| PASS | `components/product/ProductDetailPanel.vue` | 25 | component | 漫画分析 |
| PASS | `components/product/ProductDetailSection.vue` | 84 | component | 漫画分析 |
| PASS | `components/product/ProductEmptyState.vue` | 164 | component | 书架、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `components/product/ProductFileDropzone.vue` | 138 | component | 书架、漫画分析、翻译 |
| PASS | `components/product/ProductFolderCard.vue` | 83 | component | 翻译 |
| PASS | `components/product/ProductFormSection.vue` | 54 | component | 翻译 |
| PASS | `components/product/ProductHeaderAction.vue` | 260 | component | 书架、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `components/product/ProductHeaderMetaPill.vue` | 56 | component | 书架 |
| PASS | `components/product/ProductLogPanel.vue` | 153 | component | 翻译 |
| PASS | `components/product/ProductMessageBubble.vue` | 234 | component | 漫画分析、角色工坊、翻译 |
| PASS | `components/product/ProductPageHeader.vue` | 366 | component | 书架、漫画分析、阅读器、翻译 |
| PASS | `components/product/ProductRecordCard.vue` | 166 | component | 书架、漫画分析、角色工坊、翻译 |
| PASS | `components/product/ProductScrollStack.vue` | 84 | component | 书架、漫画分析、翻译 |
| PASS | `components/product/ProductSearchField.vue` | 132 | component | 书架、角色工坊、翻译 |
| PASS | `components/product/ProductSearchToolbar.vue` | 53 | component | 书架 |
| PASS | `components/product/ProductSectionHeader.vue` | 117 | component | 书架、漫画分析、翻译 |
| PASS | `components/product/ProductSegmentedTabs.vue` | 212 | component | 漫画分析、角色工坊、翻译 |
| PASS | `components/product/ProductSelectableImageGrid.vue` | 132 | component | 翻译 |
| PASS | `components/product/ProductSplitWorkspace.vue` | 202 | component | 角色工坊 |
| PASS | `components/product/ProductStatusBanner.vue` | 151 | component | 书架、全局应用壳、漫画分析、角色工坊、翻译 |
| FIXED | `components/product/ProductTabbedWorkspace.vue` | 217 | component | 漫画分析；活动标签在响应式变窄时保持完整可见 |
| FIXED | `components/product/ProductTextInputProvider.vue` | 75 | component | 全局应用壳 |
| PASS | `components/product/ProductThemeToggle.vue` | 57 | component | 书架、漫画分析、翻译 |
| FIXED | `components/product/ProductThreePaneWorkspace.vue` | 133 | component | 漫画分析 |
| FIXED | `components/product/ProductThumbnailGrid.vue` | 296 | component | 漫画分析、翻译 |
| PASS | `components/product/ProductWizardSteps.vue` | 164 | component | 漫画分析 |
| PASS | `components/product/ProductWorkspacePanel.vue` | 102 | component | 漫画分析 |
| FIXED | `components/reader/ReaderCanvas.vue` | 151 | component | 阅读器 |
| FIXED | `components/reader/ReaderControls.vue` | 445 | component | 阅读器 |
| PASS | `components/reader/readerSettings.ts` | 84 | component | 阅读器 |
| PASS | `components/settings/AiProviderCredentialFields.vue` | 103 | component | 漫画分析、翻译；受控凭据字段，无状态镜像 |
| PASS | `components/settings/AiProviderSelectField.vue` | 46 | component | 漫画分析、翻译；受控 Provider 选择 |
| FIXED | `components/settings/DetectionSettings.vue` | 209 | component | 翻译；唯一 Store 草稿与响应式长表单已验收 |
| FIXED | `components/settings/HqTranslationSettings.vue` | 333 | component | 翻译；唯一 Store 草稿、模型请求与当前高质量选项已验收 |
| FIXED | `components/settings/MoreSettings.vue` | 276 | component | 翻译；并行、字体、维护和调试资源边界已验收 |
| FIXED | `components/settings/OcrSettings.vue` | 628 | component | 翻译；当前 OCR、混合 OCR 条件态与提示词边界已验收 |
| PASS | `components/settings/ocrSettingsOptions.ts` | 108 | component | 翻译；当前 OCR 选项纯映射 |
| FIXED | `components/settings/ParallelSettings.vue` | 111 | component | 翻译；并行设置唯一草稿与正整数边界已验收 |
| PASS | `components/settings/PluginAgentModal.vue` | 893 | component | 翻译；唯一 Store 草稿、会话命令锁与执行状态 |
| PASS | `components/settings/pluginAgentTimeline.ts` | 274 | component | 翻译；事件单向折叠为展示模型 |
| FIXED | `components/settings/PluginManager.vue` | 684 | component | 翻译；权威插件资源、加载三态、命令与刷新竞态 |
| PASS | `components/settings/PromptLibrary.vue` | 449 | component | 翻译；失败重试与后端权威写结果 |
| FIXED | `components/settings/ProofreadingSettings.vue` | 538 | component | 翻译；稳定轮次、显式 action 与请求生命周期已验收 |
| PASS | `components/settings/SavedPromptsPicker.vue` | 126 | component | 翻译；请求竞态与失败重试 |
| FIXED | `components/settings/SettingsModal.vue` | 366 | component | 翻译；设置事务、关闭锁、九标签桌面与 390px 窄屏已验收 |
| PASS | `components/settings/shared/TranslationConstraintTable.vue` | 383 | component | 翻译；筛选排序保持原记录身份 |
| FIXED | `components/settings/TextStyleDefaultsSettings.vue` | 380 | component | 翻译；唯一事务草稿与当前文字样式边界已验收 |
| FIXED | `components/settings/TranslationSettings.vue` | 604 | component | 翻译；Provider 草稿、凭据、模型请求与高级选项已验收 |
| PASS | `components/settings/translationSettingsLabels.ts` | 51 | component | 翻译；当前 UI 文案纯投影 |
| PASS | `components/settings/usePluginAgentDisplayAnimation.ts` | 126 | component | 翻译；可清理的展示动画临时状态 |
| PASS | `components/settings/usePluginAgentModal.ts` | 869 | component | 翻译；会话恢复、命令串行、完整事件补拉与失败释放 |
| FIXED | `components/task-center/TaskBatchAnalysisModal.vue` | 258 | component | 全局应用壳 |
| FIXED | `components/task-center/TaskCenterDrawer.vue` | 1201 | component | 全局应用壳 |
| PASS | `components/task-center/TaskCenterLauncher.vue` | 96 | component | 全局应用壳 |
| FIXED | `components/task-center/TaskStatusBadge.vue` | 92 | component | 书架 |
| FIXED | `components/translate/BookGlossaryModal.vue` | 253 | component | 翻译 |
| FIXED | `components/translate/BookNonTranslateModal.vue` | 168 | component | 翻译 |
| FIXED | `components/translate/DetectedTextPanel.vue` | 119 | component | 翻译 |
| PASS | `components/translate/FirstTimeGuide.vue` | 155 | component | 翻译 |
| PASS | `components/translate/firstTimeGuideState.ts` | 29 | component | 翻译 |
| FIXED | `components/translate/ImageResultDisplay.vue` | 205 | component | 翻译 |
| FIXED | `components/translate/ImageUpload.vue` | 394 | component | 翻译 |
| FIXED | `components/translate/PageSelectionModal.vue` | 397 | component | 翻译 |
| FIXED | `components/translate/QuickWorkspacePromoteModal.vue` | 222 | component | 翻译 |
| PASS | `components/translate/result/ExportActions.vue` | 173 | component | 翻译 |
| PASS | `components/translate/result/ResultImageCanvas.vue` | 164 | component | 翻译 |
| PASS | `components/translate/result/ResultToolbar.vue` | 129 | component | 翻译 |
| PASS | `components/translate/settings-sidebar/ApplyOptionsSection.vue` | 199 | component | 翻译 |
| PASS | `components/translate/settings-sidebar/BookConstraintSection.vue` | 109 | component | 翻译 |
| PASS | `components/translate/settings-sidebar/NavigationButtons.vue` | 66 | component | 翻译 |
| PASS | `components/translate/settings-sidebar/PageSelectionSection.vue` | 192 | component | 翻译 |
| PASS | `components/translate/settings-sidebar/TextStyleSection.vue` | 529 | component | 翻译 |
| PASS | `components/translate/settings-sidebar/WorkflowSection.vue` | 166 | component | 翻译 |
| PASS | `components/translate/SettingsSidebar.vue` | 255 | component | 翻译 |
| PASS | `components/translate/ThumbnailSidebar.vue` | 307 | component | 翻译 |
| PASS | `components/translate/TranslationProgress.vue` | 209 | component | 翻译；AUDIT-110/AUDIT-126 |
| FIXED | `components/translate/useSettingsSidebar.ts` | 560 | component | 翻译；AUDIT-111/AUDIT-126 |
| PASS | `components/translate/useWebImportModal.ts` | 979 | component | 翻译 |
| PASS | `components/translate/web-import/WebImportAdvancedSettingsPanel.vue` | 61 | component | 翻译 |
| PASS | `components/translate/web-import/WebImportBasicSettingsPanel.vue` | 290 | component | 翻译 |
| PASS | `components/translate/web-import/WebImportExtractBar.vue` | 231 | component | 翻译 |
| FIXED | `components/translate/web-import/WebImportFooterActions.vue` | 46 | component | 翻译 |
| FIXED | `components/translate/web-import/WebImportLogsPanel.vue` | 46 | component | 翻译 |
| PASS | `components/translate/web-import/WebImportResultsGrid.vue` | 161 | component | 翻译 |
| PASS | `components/translate/web-import/webImportSettingsActions.ts` | 35 | component | 翻译 |
| PASS | `components/translate/web-import/WebImportSettingsPanel.vue` | 219 | component | 翻译 |
| PASS | `components/translate/WebImportDisclaimer.vue` | 370 | component | 翻译 |
| PASS | `components/translate/WebImportModal.vue` | 190 | component | 翻译 |
| PASS | `components/translate/WebImportPreprocessSettings.vue` | 152 | component | 翻译 |
| PASS | `components/translate/workflowModeConfig.ts` | 85 | component | 翻译 |
| PASS | `components/ui/AppShell.vue` | 101 | component | 书架、漫画分析、阅读器、角色工坊、翻译；五页面使用矩阵闭合并删除死公共 API |
| PASS | `components/ui/iconRegistry.ts` | 166 | component | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `components/ui/OverlayLayer.vue` | 53 | component | 全局应用壳、漫画分析、阅读器、翻译 |
| PASS | `components/ui/selectTypes.ts` | 12 | component | 书架、全局应用壳、漫画分析、角色工坊、翻译 |
| PASS | `components/ui/SidebarLayout.vue` | 232 | component | 翻译 |
| PASS | `components/ui/UiButton.vue` | 292 | component | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `components/ui/UiCheckbox.vue` | 97 | component | 书架、全局应用壳、漫画分析、角色工坊、翻译 |
| PASS | `components/ui/UiColorInput.vue` | 81 | component | 书架、翻译 |
| PASS | `components/ui/UiColorSwatchGroup.vue` | 77 | component | 阅读器 |
| FIXED | `components/ui/UiCombobox.vue` | 520 | component | 漫画分析、翻译 |
| PASS | `components/ui/UiField.vue` | 252 | component | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `components/ui/UiFileInput.vue` | 63 | component | 书架、漫画分析、角色工坊、翻译 |
| PASS | `components/ui/UiFormGrid.vue` | 18 | component | 书架、漫画分析、角色工坊、翻译 |
| PASS | `components/ui/UiIcon.vue` | 53 | component | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `components/ui/UiIconButton.vue` | 150 | component | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `components/ui/UiInput.vue` | 194 | component | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `components/ui/UiModelPicker.vue` | 191 | component | 漫画分析、翻译 |
| PASS | `components/ui/UiNumberField.vue` | 202 | component | 漫画分析、角色工坊、翻译 |
| PASS | `components/ui/UiPasswordField.vue` | 91 | component | 漫画分析、翻译 |
| PASS | `components/ui/UiProgressBar.vue` | 141 | component | 漫画分析、翻译 |
| FIXED | `components/ui/UiSelect.vue` | 477 | component | 书架、全局应用壳、漫画分析、角色工坊、翻译 |
| PASS | `components/ui/UiSpinner.vue` | 45 | component | 漫画分析、阅读器、翻译 |
| PASS | `components/ui/UiSwitch.vue` | 110 | component | 漫画分析、翻译 |
| PASS | `components/ui/UiTextarea.vue` | 190 | component | 漫画分析、角色工坊、翻译 |
| FIXED | `components/virtual/VirtualPageStream.vue` | 249 | component | 阅读器 |
| FIXED | `components/virtual/VirtualThumbnailGrid.vue` | 171 | component | 漫画分析、翻译 |
| FIXED | `components/virtual/VirtualThumbnailList.vue` | 135 | component | 翻译 |
| FIXED | `components/virtual/virtualWindow.ts` | 90 | component | 漫画分析、阅读器、翻译 |
| FIXED | `composables/continuation/continuationActionRunner.ts` | 44 | composable | 漫画分析 |
| FIXED | `composables/continuation/promptValidation.ts` | 34 | composable | 漫画分析 |
| FIXED | `composables/continuation/useCharacterManagement.ts` | 237 | composable | 漫画分析 |
| FIXED | `composables/continuation/useContinuationState.ts` | 368 | composable | 漫画分析 |
| PASS | `composables/continuation/useImageGeneration.ts` | 166 | composable | 漫画分析 |
| PASS | `composables/edit/useEditWorkspaceKeyboardShortcuts.ts` | 109 | composable | 翻译 |
| PASS | `composables/edit/useEditWorkspaceProcessingActions.ts` | 236 | composable | 翻译 |
| PASS | `composables/edit/useEditWorkspaceResizeActions.ts` | 134 | composable | 翻译 |
| PASS | `composables/useAiModelDiscovery.ts` | 156 | composable | 漫画分析、翻译 |
| PASS | `composables/useBrush.ts` | 370 | composable | 翻译 |
| PASS | `composables/useBubbleActions.ts` | 262 | composable | 翻译 |
| PASS | `composables/useBodyScrollLock.ts` | 40 | composable | 全局应用壳 |
| PASS | `composables/useDialogLifecycle.ts` | 128 | composable | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `composables/useEditRender.ts` | 145 | composable | 翻译 |
| PASS | `composables/useExportImport.ts` | 242 | composable | 翻译 |
| FIXED | `composables/useFolderTree.ts` | 136 | composable | 翻译 |
| PASS | `composables/useImageViewer.ts` | 149 | composable | 翻译 |
| PASS | `composables/useLatestRequestGuard.ts` | 28 | composable | 漫画分析、翻译 |
| PASS | `composables/useOverlayDismiss.ts` | 89 | composable | 书架、全局应用壳、漫画分析、角色工坊、翻译 |
| FIXED | `composables/useProductConfirm.ts` | 70 | composable | 书架、全局应用壳、漫画分析、角色工坊、翻译 |
| FIXED | `composables/useProductTextInput.ts` | 69 | composable | 全局应用壳、漫画分析 |
| FIXED | `composables/useTextStyleSync.ts` | 385 | composable | 翻译 |
| FIXED | `composables/useThumbnailSelection.ts` | 67 | composable | 翻译 |
| FIXED | `composables/useTranslateInit.ts` | 530 | composable | 翻译 |
| FIXED | `composables/useTranslationPipeline.ts` | 792 | composable | 翻译；AUDIT-126 |
| FIXED | `composables/useValidation.ts` | 343 | composable | 翻译 |
| PASS | `config/aiProviders.ts` | 97 | support | 书架、全局应用壳、漫画分析、翻译 |
| PASS | `constants/bookshelf.ts` | 1 | support | 书架、全局应用壳、漫画分析、翻译 |
| PASS | `constants/edit.ts` | 10 | support | 书架、全局应用壳、漫画分析、翻译 |
| PASS | `constants/index.ts` | 15 | support | 书架、全局应用壳、漫画分析、翻译 |
| PASS | `constants/ocr.ts` | 1 | support | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `constants/prompts.ts` | 51 | support | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `constants/rateLimits.ts` | 5 | support | 书架、全局应用壳、漫画分析、翻译 |
| PASS | `constants/routes.ts` | 7 | support | 书架、启动与路由、全局应用壳、漫画分析、翻译 |
| PASS | `constants/storage.ts` | 1 | support | 书架、全局应用壳、漫画分析、翻译 |
| PASS | `constants/webImport.ts` | 6 | support | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `defaults/textStyleDefaults.ts` | 190 | support | 书架、全局应用壳、漫画分析、翻译 |
| 删除已验证 | `defaults/textStyleFactoryDefaults.ts` | 11 | support | 翻译 |
| PASS | `env.d.ts` | 7 | support | ORPHAN |
| FIXED | `main.ts` | 25 | support | 启动与路由 |
| FIXED | `router/index.ts` | 67 | support | 启动与路由 |
| PASS | `services/backendAccessGate.ts` | 28 | support | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `services/pageDocumentPersistence.ts` | 451 | support | 翻译 |
| FIXED | `stores/bookshelfStore.ts` | 496 | store | 书架、全局应用壳、漫画分析、角色工坊、翻译 |
| FIXED | `stores/bookTranslationConstraintsStore.ts` | 72 | store | 翻译 |
| FIXED | `stores/bubbleStore.ts` | 256 | store | 翻译 |
| FIXED | `stores/characterStudio/useCharacterStudioChat.ts` | 316 | store | 角色工坊 |
| PASS | `stores/characterStudioActivity.ts` | 93 | store | 角色工坊 |
| PASS | `stores/characterStudioAgentOutput.ts` | 35 | store | 角色工坊 |
| PASS | `stores/characterStudioChatSession.ts` | 32 | store | 角色工坊 |
| PASS | `stores/characterStudioExports.ts` | 23 | store | 角色工坊 |
| FIXED | `stores/characterStudioPatch.ts` | 730 | store | 角色工坊 |
| PASS | `stores/characterStudioPatchSummary.ts` | 190 | store | 角色工坊 |
| FIXED | `stores/characterStudioStore.ts` | 1127 | store | 角色工坊 |
| FIXED | `stores/imageStore.ts` | 188 | store | 翻译；AUDIT-126 |
| 删除已验证 | `stores/insight/insightConfigApiHydration.ts` | 100 | store | 漫画分析 |
| 删除已验证 | `stores/insight/insightConfigApiPayload.ts` | 176 | store | 漫画分析 |
| PASS | `stores/insight/insightConfigDefaults.ts` | 52 | store | 漫画分析 |
| 删除已验证 | `stores/insight/insightNotesModels.ts` | 53 | store | 漫画分析 |
| 删除已验证 | `stores/insight/insightProviderSettingsHydration.ts` | 101 | store | 漫画分析 |
| FIXED | `stores/insight/useInsightConfigManager.ts` | 163 | store | 漫画分析 |
| FIXED | `stores/insight/useInsightNotes.ts` | 241 | store | 漫画分析 |
| FIXED | `stores/insight/useInsightQA.ts` | 22 | store | 漫画分析 |
| FIXED | `stores/insightStore.ts` | 407 | store | 漫画分析 |
| FIXED | `stores/settings/defaults.ts` | 183 | store | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `stores/settings/index.ts` | 922 | store | 书架、全局应用壳、漫画分析、翻译 |
| PASS | `stores/settings/modules/detection.ts` | 59 | store | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `stores/settings/modules/hqTranslation.ts` | 104 | store | 书架、全局应用壳、漫画分析、翻译 |
| PASS | `stores/settings/modules/index.ts` | 9 | store | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `stores/settings/modules/misc.ts` | 53 | store | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `stores/settings/modules/ocr.ts` | 162 | store | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `stores/settings/modules/pluginAgent.ts` | 88 | store | 书架、全局应用壳、漫画分析、翻译 |
| PASS | `stores/settings/modules/prompts.ts` | 19 | store | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `stores/settings/modules/proofreading.ts` | 58 | store | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `stores/settings/modules/translation.ts` | 129 | store | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `stores/settings/modules/webImport.ts` | 274 | store | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `stores/settings/providerConfigCache.ts` | 99 | store | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `stores/settings/schema.ts` | 326 | store | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `stores/settings/types.ts` | 42 | store | 书架、全局应用壳、漫画分析、翻译 |
| PASS | `stores/settings/useThemePreference.ts` | 88 | store | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `stores/taskCenterProjection.ts` | 142 | store | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `stores/taskCenterStore.ts` | 646 | store | 书架、全局应用壳、漫画分析、翻译；权威快照、SSE 事件全集、命令和详情竞态闭合 |
| FIXED | `stores/webImportSettingsPayload.ts` | 279 | store | 翻译 |
| FIXED | `stores/webImportStore.ts` | 541 | store | 翻译 |
| 删除已验证 | `styles/animations.css` | 75 | style | 启动与路由 |
| 删除已验证 | `styles/base.css` | 1 | style | 启动与路由 |
| FIXED | `styles/reset.css` | 24 | style | 启动与路由 |
| FIXED | `styles/tokens/component.css` | 53 | style | 启动与路由 |
| PASS | `styles/tokens/domain.css` | 16 | style | 启动与路由 |
| PASS | `styles/tokens/foundation.css` | 38 | style | 启动与路由 |
| PASS | `styles/tokens/semantic.css` | 176 | style | 启动与路由 |
| PASS | `types/api.ts` | 3 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/apiCore.ts` | 6 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `types/bookshelf.ts` | 28 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `types/bookTranslationConstraints.ts` | 3 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `types/bubble.ts` | 65 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/characterStudio.ts` | 9 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `types/characterStudioApi.ts` | 53 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `types/characterStudioChat.ts` | 56 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `types/characterStudioDocument.ts` | 118 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/characterStudioEditor.ts` | 8 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/characterStudioPatch.ts` | 70 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/diagnostics.ts` | 4 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/folder.ts` | 8 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `types/image.ts` | 74 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/index.ts` | 23 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/insight.ts` | 7 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `types/insightNotesQaTypes.ts` | 23 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/insightResponseTypes.ts` | 16 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `types/insightStoreTypes.ts` | 130 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `types/insightTimelineTypes.ts` | 76 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/ocr.ts` | 3 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/ocrSettings.ts` | 30 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/openaiSettings.ts` | 17 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/settings.ts` | 37 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/settingsProviders.ts` | 33 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/textStyleSettings.ts` | 29 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/translationConstraints.ts` | 12 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `types/translationSettings.ts` | 118 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| FIXED | `types/webImport.ts` | 116 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `types/workflow.ts` | 12 | type | 书架、全局应用壳、漫画分析、阅读器、角色工坊、翻译 |
| PASS | `utils/binaryMaskPng.ts` | 119 | util | 翻译 |
| FIXED | `utils/bookTranslationConstraints.ts` | 17 | util | 翻译 |
| PASS | `utils/browserDownload.ts` | 21 | util | 全局应用壳、漫画分析、角色工坊、翻译 |
| PASS | `utils/bubbleDrag.ts` | 19 | util | 翻译 |
| FIXED | `utils/bubbleFactory.ts` | 101 | util | 翻译 |
| PASS | `utils/bubbleResize.ts` | 163 | util | 翻译 |
| PASS | `utils/characterStudioGreetings.ts` | 34 | util | 角色工坊 |
| PASS | `utils/clipboard.ts` | 34 | util | 书架、漫画分析、角色工坊、翻译 |
| PASS | `utils/deepClone.ts` | 14 | util | 书架、全局应用壳、漫画分析、角色工坊、翻译 |
| PASS | `utils/fontFiles.ts` | 15 | util | 翻译 |
| FIXED | `utils/hybridOcr.ts` | 80 | util | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `utils/insightJobProgress.ts` | 36 | util | 书架、全局应用壳、漫画分析、翻译 |
| PASS | `utils/insightStatus.ts` | 20 | util | 漫画分析 |
| FIXED | `utils/openaiOptions.ts` | 140 | util | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `utils/pageSelection.ts` | 28 | util | 翻译 |
| FIXED | `utils/sanitizeHtml.ts` | 89 | util | 书架、启动与路由、全局应用壳、漫画分析、阅读器、翻译 |
| FIXED | `utils/taskDisplay.ts` | 140 | util | 书架、全局应用壳、漫画分析、翻译 |
| FIXED | `utils/textStyleForm.ts` | 17 | util | 翻译 |
| FIXED | `utils/toast.ts` | 104 | util | 书架、启动与路由、全局应用壳、漫画分析、阅读器、翻译 |
| FIXED | `utils/translationConstraintTable.ts` | 119 | util | 翻译 |
| FIXED | `views/BookshelfView.vue` | 643 | view | 书架 |
| FIXED | `views/CharacterStudioView.vue` | 599 | view | 角色工坊 |
| FIXED | `views/InsightView.vue` | 786 | view | 漫画分析 |
| FIXED | `views/ReaderView.vue` | 407 | view | 阅读器 |
| FIXED | `views/TranslateView.vue` | 665 | view | 翻译 |
| FIXED | `views/useTranslateViewActions.ts` | 326 | view | 翻译 |

## 初始孤立文件复核

- [x] `env.d.ts`：它是 TypeScript/Vite 的环境声明入口，由 `tsconfig.app.json` 的 `src/**/*.ts` include 覆盖，并由 `typeSourceContracts.spec.ts` 校验；不是运行时 import，也不是死文件。

## 使用规则

- 页面完成前，其所有专属文件和共享依赖均必须在 `current-files.tsv` 有结论。
- 共享文件要分别核对所有可达页面的使用契约，不能只在一个页面验证。
- 动态字符串、后端返回资源和构建资产另由 API/资源清单验证。
