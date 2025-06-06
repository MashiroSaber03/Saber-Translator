import * as state from './state.js';
import * as ui from './ui.js';
import * as api from './api.js';
import * as editMode from './edit_mode.js'; // 引入编辑模式逻辑
import * as labelingMode from './labeling_mode.js'; // <--- 新增导入
import * as main from './main.js';
import * as session from './session.js'; // <--- 新增导入
import * as constants from './constants.js'; // 确保导入前端常量
import * as hqTranslation from './high_quality_translation.js'; // 导入高质量翻译模块

/**
 * 绑定所有事件监听器
 */
export function bindEventListeners() {
    console.log("开始绑定事件监听器...");

    // --- 文件上传与拖拽 ---
    const dropArea = $("#drop-area");
    const imageUploadInput = $("#imageUpload");
    const selectFileLink = $("#select-file-link");

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.on(eventName, preventDefaults);
    });
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.on(eventName, () => dropArea.addClass('highlight'));
    });
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.on(eventName, () => dropArea.removeClass('highlight'));
    });
    dropArea.on('drop', handleDrop);
    selectFileLink.on('click', (e) => {
        e.preventDefault();
        imageUploadInput.click();
    });
    imageUploadInput.on('change', handleFileSelect);

    // --- 主要操作按钮 ---
    $("#translateButton").on('click', handleTranslateCurrent);
    $("#translateAllButton").on('click', handleTranslateAll);
    $("#proofreadButton").on('click', handleProofread);
    $("#proofreadSettingsButton").on('click', handleProofreadSettings);
    $("#removeTextOnlyButton").on('click', handleRemoveTextOnly); // 仅消除文字
    $("#removeAllTextButton").on('click', handleRemoveAllText); // 消除所有图片文字
    $("#deleteCurrentImageButton").on('click', handleDeleteCurrent);
    $("#clearAllImagesButton").on('click', handleClearAll);
    $("#applyFontSettingsToAllButton").on('click', handleApplySettingsToAll); // 应用设置到全部

    // --- 导航与显示 ---
    $("#prevImageButton").on('click', handlePrevImage);
    $("#nextImageButton").on('click', handleNextImage);
    $("#toggleImageButton").on('click', handleToggleImageDisplay);
    $("#imageSize").on('input', handleImageSizeChange);
    $("#thumbnail-sidebar #thumbnailList").on('click', '.thumbnail-item', handleThumbnailClick); // 事件委托

    // --- 下载 ---
    $("#downloadButton").on('click', handleDownloadCurrent);
    $("#downloadAllImagesButton").on('click', handleDownloadAll);
    
    // --- 导出和导入文本 ---
    $("#exportTextButton").on('click', handleExportText);
    $("#importTextButton").on('click', handleImportTextClick);
    $("#importTextFileInput").on('change', handleImportTextFile);

    // --- 设置项变更 ---
    $("#fontSize").on('change', handleGlobalSettingChange);
    $("#autoFontSize").on('change', handleGlobalSettingChange); // 自动字号也触发
    $("#layoutDirection").on('change', handleGlobalSettingChange);
    $("#textColor").on('input', handleGlobalSettingChange); // 颜色实时变化
    $("#useInpainting").on('change', handleInpaintingMethodChange);
    $("#inpaintingStrength").on('input', handleInpaintingStrengthChange); // 滑块实时变化
    $("#blendEdges").on('change', handleGlobalSettingChange); // 边缘融合也触发重渲染
    $("#fillColor").on('input', handleGlobalSettingChange); // 填充颜色实时变化
    $("#ocrEngine").on('change', handleOcrEngineChange); // OCR引擎变化
    
    // 百度OCR相关事件
    $("#baiduApiKey, #baiduSecretKey, #baiduVersion").on('change', handleGlobalSettingChange);
    $("#testBaiduOcrButton").on('click', api.testBaiduOcrConnectionApi); // 百度OCR测试按钮
    
    // --- AI视觉OCR相关事件 ---
    // (如果之前有AI视觉OCR的事件绑定，这是修改；如果是全新的，这是新增)

    // AI视觉OCR服务商选择变更
    $("#aiVisionProvider").on('change', async function(e) {
        const provider = $(this).val();
        state.setAiVisionProvider(provider); // 更新 state 中的服务商

        // VVVVVV 新增逻辑：根据选择显示/隐藏自定义 Base URL 输入框 VVVVVV
        if (provider === constants.CUSTOM_AI_VISION_PROVIDER_ID_FRONTEND) { // 使用前端常量比较
            ui.toggleCustomAiVisionBaseUrlUI(true);
            // 可选: 如果用户切换到自定义，但 Base URL 为空，可以聚焦输入框
            // if (!$("#customAiVisionBaseUrl").val().trim()) {
            //     $("#customAiVisionBaseUrl").focus();
            // }
        } else {
            ui.toggleCustomAiVisionBaseUrlUI(false);
        }
        // ^^^^^^ 结束新增逻辑 ^^^^^^

        // --- 已有的火山引擎模型建议逻辑 (保持或调整) ---
        if (provider === 'volcano') {
            // const { getUsedModelsApi } = await import('./api.js'); // 这些 import 最好在文件顶部
            // const { updateModelSuggestions } = await import('./ui.js');
            try {
                const response = await api.getUsedModelsApi('volcano'); // 使用 api.js 中的函数
                if (response && response.models) {
                    // ... (更新 datalist 的逻辑保持不变) ...
                    // 示例性地假设 updateModelSuggestions 能够处理 aiVisionModelName 的 datalist
                    ui.updateModelSuggestions(response.models, 'aiVisionModelsList', 'aiVisionModelName');
                }
            } catch (error) {
                console.error("获取火山引擎模型建议失败:", error);
            }
        } else {
            // 清除非火山引擎时的模型建议 (如果适用)
             ui.updateModelSuggestions([], 'aiVisionModelsList', 'aiVisionModelName');
        }
        // --- 结束火山引擎逻辑 ---

        // 确保这个变更也被视为全局设置变更，以便在必要时触发重渲染或状态保存
        handleGlobalSettingChange({ target: this }); // 模拟事件对象，this 指向 #aiVisionProvider
    });

    // AI视觉OCR API Key, 模型名称, 提示词的 change 事件 (保持不变，它们调用 handleGlobalSettingChange)
    $("#aiVisionApiKey, #aiVisionModelName, #aiVisionOcrPrompt").on('change', handleGlobalSettingChange);
    // 移除旧的按钮点击事件绑定，改为在上面使用选择器change事件
    $("#rpmAiVisionOcr").on('change input', function() { // rpm 设置
        const value = $(this).val();
        state.setrpmLimitAiVisionOcr(value);
    });


    // VVVVVV 新增：自定义AI视觉Base URL输入框的 input 事件监听 VVVVVV
    $("#customAiVisionBaseUrl").on('input', function() {
        const url = $(this).val().trim();
        state.setCustomAiVisionBaseUrl(url); // 实时更新 state.js 中的值
        // 如果需要，也可以将此视为一个会触发自动存档或影响"已修改"状态的全局设置变更
        // handleGlobalSettingChange({ target: this }); // 但通常 Base URL 更改不直接触发重渲染
    });
    // ^^^^^^ 结束新增 ^^^^^^

    // 测试AI视觉OCR按钮点击 (保持不变，内部逻辑在 handleTestAiVisionOcr 中修改)
    $("#testAiVisionOcrButton").on('click', handleTestAiVisionOcr);

    // 模型与提示词 ---
    $("#modelProvider").on('change', handleModelProviderChange);
    $("#modelName").on('focus', handleModelNameFocus);
    $("#modelName").on('blur', handleModelNameBlur);
    // Ollama/Sakura 测试按钮 (如果存在)
    // 注意：按钮是动态添加的，需要使用事件委托或在按钮创建后绑定
    $(document).on('click', '#testOllamaButton', function() {
        // 使用函数调用以确保 this 上下文正确
        api.testOllamaConnectionApi().then(response => {
            if (response && response.success) {
                ui.showGeneralMessage("Ollama服务连接成功!", "success");
                // 刷新模型列表
                import('./main.js').then(main => {
                    main.fetchOllamaModels();
                });
            } else {
                ui.showGeneralMessage("Ollama服务连接失败: " + (response?.message || "未知错误"), "error");
            }
        }).catch(error => {
            ui.showGeneralMessage("Ollama服务连接失败: " + error.message, "error");
        });
    });
    
    $(document).on('click', '#testSakuraButton', function() {
        // 使用函数调用以确保 this 上下文正确
        api.testSakuraConnectionApi(true).then(response => {  // 传入true强制刷新模型列表
            if (response && response.success) {
                ui.showGeneralMessage("Sakura服务连接成功!", "success");
                // 刷新模型列表
                import('./main.js').then(main => {
                    main.fetchSakuraModels();
                });
            } else {
                ui.showGeneralMessage("Sakura服务连接失败: " + (response?.message || "未知错误"), "error");
            }
        }).catch(error => {
            ui.showGeneralMessage("Sakura服务连接失败: " + error.message, "error");
        });
    });
    
    // 模型按钮点击 (事件委托)
    $(document).on('click', '#ollamaModelsList .model-button, #sakuraModelsList .model-button', handleModelButtonClick);

    $("#savePromptButton").on('click', handleSavePrompt);
    $("#promptDropdownButton").on('click', (e) => { e.stopPropagation(); $("#promptDropdown").toggle(); });
    // 下拉列表项和删除按钮的点击事件在 populate 函数中绑定
    
    // --- 新增 JSON 提示词切换按钮事件 ---
    // 移除旧的按钮点击事件绑定，改为使用选择器change事件
    // ------------------------------------
    
    // 记住提示词复选框变更事件
    $("#rememberPrompt").on('change', function() {
        $("#promptName").toggle($(this).is(':checked'));
    });

    $("#enableTextboxPrompt").on('change', handleEnableTextboxPromptChange);
    $("#saveTextboxPromptButton").on('click', handleSaveTextboxPrompt);
    $("#textboxPromptDropdownButton").on('click', (e) => { e.stopPropagation(); $("#textboxPromptDropdown").toggle(); });
    // 下拉列表项和删除按钮的点击事件在 populate 函数中绑定
    
    // 记住文本框提示词复选框变更事件
    $("#rememberTextboxPrompt").on('change', function() {
        $("#textboxPromptName").toggle($(this).is(':checked'));
    });

    // --- 系统与其他 ---
    $("#cleanDebugFilesButton").on('click', handleCleanDebugFiles);
    $(document).on('keydown', handleGlobalKeyDown); // 全局快捷键
    $("#themeToggle").on('click', handleThemeToggle); // 主题切换
    $(document).on('click', '#donateButton', handleDonateClick); // 赞助按钮
    $(document).on('click', '.donate-close', handleDonateClose);
    $(window).on('click', handleWindowClickForModal); // 点击模态框外部关闭

    // === 修改/新增：会话管理按钮事件 ===
    $("#saveCurrentSessionButton").on('click', handleSaveCurrentSession); // <--- 新增"保存"绑定
    $("#saveAsSessionButton").on('click', handleSaveAsSession);       // <--- 修改为"另存为"绑定 (原 handleSaveSession)
    $("#loadSessionButton").on('click', handleLoadSessionClick);
    // === 结束修改/新增 ===

    // --- 编辑模式 ---
    $("#toggleEditModeButton").on('click', editMode.toggleEditMode);
    // 气泡列表项点击 (事件委托)
    $("#bubbleList").on('click', '.bubble-item', handleBubbleItemClick);
    // 编辑区域输入/变更事件
    $("#bubbleTextEditor").on('input', handleBubbleEditorChange);
    $("#bubbleFontSize").on('change', handleBubbleSettingChange);
    $("#autoBubbleFontSize").on('change', handleBubbleSettingChange);
    $("#bubbleFontFamily").on('change', handleBubbleSettingChange);
    $("#bubbleTextDirection").on('change', handleBubbleSettingChange);
    $("#bubbleTextColor").on('input', handleBubbleSettingChange);
    $("#bubbleRotationAngle").on('input', handleBubbleRotationChange);
    $("#bubbleFillColor").on('input', handleBubbleSettingChange);
    
    // === 新增：气泡描边控件事件绑定 START ===
    $("#bubbleEnableStroke").on('change', handleBubbleEnableStrokeChange);
    $("#bubbleStrokeColor").on('input', handleBubbleStrokeSettingChange);
    $("#bubbleStrokeWidth").on('input', handleBubbleStrokeSettingChange);
    // === 新增：气泡描边控件事件绑定 END ===
    
    $("#positionOffsetX").on('input', handleBubblePositionChange); // 位置实时变化 (带延迟)
    $("#positionOffsetY").on('input', handleBubblePositionChange); // 位置实时变化 (带延迟)
    
    // 编辑操作按钮
    $("#applyBubbleEdit").on('click', handleApplyBubbleEdit);
    $("#applyToAllBubbles").on('click', handleApplyToAllBubbles);
    $("#resetBubbleEdit").on('click', handleResetBubbleEdit);

    // 位置调整按钮 (mousedown 用于连续调整)
    $("#moveUp, #moveDown, #moveLeft, #moveRight").on("mousedown", handlePositionButtonMouseDown);
    $(document).on("mouseup", handlePositionButtonMouseUp); // 监听全局 mouseup
    $("#moveUp, #moveDown, #moveLeft, #moveRight").on("mouseleave", handlePositionButtonMouseUp); // 离开按钮也停止
    $("#resetPosition").on("click", handleResetPosition);

    // 修改全局点击事件，避免点击输入框时隐藏推荐列表
    $(document).on('click', (e) => {
        // 如果点击的是模型推荐列表或模型输入框，不隐藏模型推荐列表
        const isModelInput = $(e.target).is('#modelName');
        const isModelSuggestion = $(e.target).closest('#model-suggestions').length > 0;
        
        if (!isModelInput && !isModelSuggestion) {
            $("#model-suggestions").hide();
        }
        
        // 其他下拉菜单的隐藏逻辑
        const isPromptDropdown = $(e.target).closest('#promptDropdown, #prompt-dropdown-container').length > 0;
        const isTextboxDropdown = $(e.target).closest('#textboxPromptDropdown, #textbox-prompt-dropdown-container').length > 0;
        
        if (!isPromptDropdown) {
            $("#promptDropdown").hide();
        }
        
        if (!isTextboxDropdown) {
            $("#textboxPromptDropdown").hide();
        }
    }); // 点击页面上的不相关区域时隐藏下拉列表

    // --- 插件管理 ---
    $("#managePluginsButton").on('click', handleManagePluginsClick);
    $(document).on('click', '.plugin-modal-close', handlePluginModalClose);
    $(window).on('click', handleWindowClickForPluginModal);
    // 实时启用/禁用开关 (事件委托) - target 指向 input
    $("#pluginListContainer").on('change', 'input.plugin-enable-toggle', handlePluginToggleChange);
    // 删除按钮 (事件委托) - target 指向 button
    $("#pluginListContainer").on('click', 'button.plugin-delete-button', handlePluginDeleteClick);
    // 设置按钮 (事件委托) - target 指向 button
    $("#pluginListContainer").on('click', 'button.plugin-settings-button', handlePluginSettingsClick);
    // 配置模态框关闭 (事件委托) - target 指向 span
    $(document).on('click', '#pluginConfigModal .plugin-modal-close', handlePluginConfigModalClose);
    // 配置保存 (事件委托) - target 指向 form
    $(document).on('submit', '#pluginConfigForm', handlePluginConfigSave);
    // --- 新增: 默认启用状态开关 (事件委托) ---
    $("#pluginListContainer").on('change', 'input.plugin-default-toggle', handlePluginDefaultStateChange);
    // ----------------------------------------

    // --- 新增：标注模式 ---
    $("#toggleLabelingModeButton").on('click', handleToggleLabelingMode);
    $("#autoDetectBoxesButton").on('click', labelingMode.handleAutoDetectClick);
    $("#detectAllImagesButton").on('click', labelingMode.handleDetectAllImagesClick);
    $("#clearManualBoxesButton").on('click', labelingMode.handleClearManualBoxesClick);
    $("#deleteSelectedBoxButton").on('click', handleDeleteSelectedBoxClick);
    $("#useManualBoxesButton").on('click', labelingMode.handleUseManualBoxesClick);

    const imageContainer = $('.image-container');
    imageContainer.on('mousedown', labelingMode.handleMouseDownOnImage); // 绘制新框
    $(document).on('mousemove', labelingMode.handleMouseMove);
    $(document).on('mouseup', labelingMode.handleMouseUp);

    // 修改：只在框本身（非手柄）上按下鼠标才触发拖动
    imageContainer.on('mousedown', '.manual-bounding-box.draggable-box:not(:has(.resize-handle:hover))', labelingMode.handleBoxMouseDown);

    // 新增：为调整大小手柄绑定 mousedown 事件
    imageContainer.on('mousedown', '.resize-handle', labelingMode.handleResizeHandleMouseDown);

    // 标注框点击事件 (用于选择，可以保留 click，但 mousedown 优先处理拖动/缩放)
    imageContainer.on('click', '.manual-bounding-box', labelingMode.handleBoxClick);

    // === 新增：会话管理模态框内的按钮事件 (使用事件委托) ===
    // 点击"加载"按钮
    $(document).on('click', '#sessionListContainer .session-load-button', function() {
        const itemDiv = $(this).closest('.session-item'); // 找到包含按钮的会话项目 div
        const sessionNameToLoad = itemDiv.data('session-name');
        if (sessionNameToLoad) {
            session.handleLoadSession(sessionNameToLoad);
        } else {
            console.error("无法获取要加载的会话名称！");
            ui.showGeneralMessage("无法加载会话，未能识别会话名称。", "error");
        }
    });

    // 点击"删除"按钮
    $(document).on('click', '#sessionListContainer .session-delete-button', function() {
        const itemDiv = $(this).closest('.session-item');
        const sessionNameToDelete = itemDiv.attr('data-session-name');
        if (sessionNameToDelete) {
            // === 确认这里调用的是正确的函数 ===
            session.handleDeleteSession(sessionNameToDelete); // 调用 session.js 中的删除处理函数
            // === 结束确认 ===
        } else {
            console.error("无法获取要删除的会话名称！ data-session-name 属性可能丢失。");
            ui.showGeneralMessage("无法删除会话，未能识别会话名称。", "error");
        }
    });

    // === 新增：点击"重命名"按钮 ===
    $(document).on('click', '#sessionListContainer .session-rename-button', function() {
        const itemDiv = $(this).closest('.session-item');
        const sessionNameToRename = itemDiv.data('session-name');
        if (sessionNameToRename) {
            // 调用 session.js 中的重命名处理函数
            session.handleRenameSession(sessionNameToRename);
        } else {
            console.error("无法获取要重命名的会话名称！");
            ui.showGeneralMessage("无法重命名会话，未能识别会话名称。", "error");
        }
    });
    // === 结束新增 ===

    // --- 模态框关闭事件 (已有) ---
    $(document).on('click', '#sessionManagerModal .session-modal-close', function() {
        ui.hideSessionManagerModal();
    });
    $(window).on('click', function(event) {
        const modal = $("#sessionManagerModal");
        if (modal.length > 0 && event.target == modal[0]) {
            ui.hideSessionManagerModal();
        }
    });
    // ------------------------
    
    // --- 校对设置模态框事件 ---
    $(document).on('click', '#proofreadingSettingsModal .plugin-modal-close', function() {
        ui.hideProofreadingSettingsModal();
    });
    $(window).on('click', function(event) {
        const modal = $("#proofreadingSettingsModal");
        if (modal.length > 0 && event.target == modal[0]) {
            ui.hideProofreadingSettingsModal();
        }
    });
    // 保存校对设置按钮
    $(document).on('click', '#saveProofreadingSettingsButton', function() {
        import('./ai_proofreading.js').then(proofreading => {
            proofreading.saveProofreadingSettings();
        }).catch(error => {
            console.error("保存校对设置失败:", error);
            ui.showGeneralMessage("保存校对设置失败: " + error.message, "error");
        });
    });
    // ------------------------

    // --- 新增：rpm 设置变更事件 ---
    $("#rpmTranslation").on('change input', function() { // 'input' 事件可实现更实时的更新（可选）
        const value = $(this).val();
        state.setrpmLimitTranslation(value);
        // 可选：如果需要，可以在这里触发一个保存用户偏好的操作，例如到 localStorage
        // localStorage.setItem('rpmLimitTranslation', state.rpmLimitTranslation);
    });

    $("#rpmAiVisionOcr").on('change input', function() {
        const value = $(this).val();
        state.setrpmLimitAiVisionOcr(value);
        // localStorage.setItem('rpmLimitAiVisionOcr', state.rpmLimitAiVisionOcr);
    });
    // ---------------------------

    // 绑定事件监听器部分中添加选择器change事件
    $("#translatePromptModeSelect").on('change', handleTranslatePromptModeChange);
    $("#aiVisionPromptModeSelect").on('change', handleAiVisionPromptModeChange);

    // 字体家族下拉框变更事件 - 主界面和编辑模式字体选择
    $(document).on('change', "#fontFamily, #bubbleFontFamily", function() {
        const selectedValue = $(this).val();
        const isEditMode = this.id === 'bubbleFontFamily';
        
        if (selectedValue === 'custom-font') {
            // 触发文件选择对话框
            $('#fontUpload').click();
            
            // 重新选择之前的值，因为"自定义字体..."不是真正的字体选项
            const previousValue = $(this).data('previous-value') || 'fonts/msyh.ttc'; // 默认微软雅黑
            $(this).val(previousValue);
        } else {
            // 保存当前选择的值，以便"自定义字体"选项后可以恢复
            $(this).data('previous-value', selectedValue);
            
            // 如果是编辑模式，调用编辑模式的处理函数，否则调用全局设置处理函数
            if (isEditMode) {
                handleBubbleSettingChange({ target: this });
            } else {
                handleGlobalSettingChange({ target: this });
            }
        }
    });

    // 字体文件上传事件
    $("#fontUpload").on('change', function(event) {
        if (this.files && this.files.length > 0) {
            const fontFile = this.files[0];
            ui.handleFontUpload(fontFile);
            // 重置文件输入，以便可以再次选择同一文件
            this.value = '';
        }
    });

    console.log("事件监听器绑定完成。");

    // 添加自动存档开关的事件处理
    $(document).on('change', '#autoSaveToggle', function() {
        const isEnabled = $(this).is(':checked');
        state.setAutoSaveEnabled(isEnabled);
        console.log(`自动存档功能已${isEnabled ? '启用' : '禁用'}`);
        ui.showGeneralMessage(`自动存档功能已${isEnabled ? '启用' : '禁用'}`, isEnabled ? 'success' : 'info', false, 2000);
    });

    // 初始化高质量翻译模块UI
    hqTranslation.initHqTranslationUI();

    // 绑定高质量翻译模式的事件
    $("#hqTranslateProvider").on('change', handleHqTranslateProviderChange);
    $("#hqApiKey").on('change', handleHqApiKeyChange);
    $("#hqModelName").on('change', handleHqModelNameChange);
    $("#hqCustomBaseUrl").on('input', handleHqCustomBaseUrlChange);
    $("#hqBatchSize").on('change', handleHqBatchSizeChange);
    $("#hqSessionReset").on('change', handleHqSessionResetChange);
    $("#hqRpmLimit").on('change', handleHqRpmLimitChange);
    $("#hqLowReasoning").on('change', handleHqLowReasoningChange);
    $("#hqForceJsonOutput").on('change', handleHqForceJsonOutputChange); // 新增：绑定强制JSON输出选项的事件
    $("#hqPrompt").on('input', handleHqPromptChange);
    $("#startHqTranslation").on('click', function() {
        hqTranslation.startHqTranslation();
    });

    // 源语言和目标语言
    $('#sourceLanguage').on('change', handleSourceLanguageChange);
    $('#targetLanguage').on('change', handleTargetLanguageChange);

    // --- 描边设置 ---  (可以放在 "#textColor" 事件绑定之后)
    $("#enableTextStroke").on('change', handleEnableTextStrokeChange);
    $("#textStrokeColor").on('input', handleTextStrokeSettingChange);
    $("#textStrokeWidth").on('input', handleTextStrokeSettingChange); // 'input' 事件用于实时响应数字输入框
}

// --- 事件处理函数 ---

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    preventDefaults(e);
    const dt = e.originalEvent.dataTransfer;
    const files = dt.files;
    handleFiles(files); // handleFiles 需要在 main.js 或 state.js 中定义并导出
}

function handleFileSelect(e) {
    handleFiles(e.target.files); // handleFiles 需要在 main.js 或 state.js 中定义并导出
}

// handleFiles 函数需要从原始 script.js 迁移到 main.js 或 state.js
// 这里假设它在 main.js 中定义并导出
import { handleFiles } from './main.js';

function handleTranslateCurrent() {
    if (state.currentImageIndex === -1) {
        ui.showGeneralMessage("请先选择要翻译的图片", "warning");
        return;
    }
    // 使用showGeneralMessage替代showLoading
    ui.showGeneralMessage("翻译中...", "info", false, 0);
    // translateCurrentImage 函数需要在 main.js 或 api.js 中定义并导出
    import('./main.js').then(main => {
        main.translateCurrentImage()
            .then(() => {
                session.triggerAutoSave(); // <--- 翻译成功后触发自动存档
            })
            .catch(error => {
                // 错误处理
            });
    });
}

function handleRemoveTextOnly() {
    if (state.currentImageIndex === -1) {
        ui.showGeneralMessage("请先选择图片", "warning");
        return;
    }
    // 使用showGeneralMessage替代showLoading
    ui.showGeneralMessage("消除文字中...", "info", false, 0);
    // removeBubbleTextOnly 函数需要在 main.js 或 api.js 中定义并导出
    import('./main.js').then(main => {
        main.removeBubbleTextOnly()
            .then(() => {
                session.triggerAutoSave(); // <--- 消除文字成功后触发
            })
            .catch(error => {
                // 错误处理
            });
    });
}

function handleRemoveAllText() {
    if (state.images.length === 0) {
        ui.showGeneralMessage("请先添加图片", "warning");
        return;
    }
    import('./main.js').then(main => main.removeAllBubblesText());
}

function handleTranslateAll() {
    if (state.images.length === 0) {
        ui.showGeneralMessage("请先添加图片", "warning");
        return;
    }
    // translateAllImages 函数需要在 main.js 或 api.js 中定义并导出
    import('./main.js').then(main => main.translateAllImages());
}

function handleClearAll() {
    if (confirm('确定要清除所有图片吗？这将丢失所有未保存的进度。')) {
        state.clearImages();
        ui.renderThumbnails();
        ui.showResultSection(false);
        ui.updateButtonStates();
        ui.updateProgressBar(0, ''); // 重置进度条
        editMode.exitEditMode(); // 如果在编辑模式则退出
        session.triggerAutoSave(); // <--- 清除所有图片后触发 (会保存一个空状态)
    }
}

function handleDeleteCurrent() {
    if (state.currentImageIndex !== -1) {
        if (confirm(`确定要删除当前图片 (${state.images[state.currentImageIndex].fileName}) 吗？`)) {
            const deletedIndex = state.currentImageIndex;
            state.deleteImage(deletedIndex); // state.deleteImage 会更新 currentImageIndex
            ui.renderThumbnails();
            if (state.images.length > 0) {
                // state.deleteImage 已经处理了索引调整，直接用更新后的索引切换
                import('./main.js').then(main => main.switchImage(state.currentImageIndex));
            } else {
                // 没有图片了
                state.setCurrentImageIndex(-1);
                ui.showResultSection(false);
                ui.updateButtonStates();
                editMode.exitEditMode();
            }
            session.triggerAutoSave(); // <--- 删除图片后触发
        }
    }
}

function handleApplySettingsToAll() {
     if (state.images.length === 0) {
        ui.showGeneralMessage("请先添加图片", "warning");
        return;
    }
    if (!state.images.some(img => img.translatedDataURL)) {
         ui.showGeneralMessage("没有已翻译的图片可应用设置", "warning");
         return;
    }
    // applySettingsToAll 函数需要在 main.js 或 api.js 中定义并导出
    import('./main.js').then(main => main.applySettingsToAll());
}


function handlePrevImage() {
    if (state.currentImageIndex > 0) {
        // switchImage 函数需要在 main.js 中定义并导出
        import('./main.js').then(main => main.switchImage(state.currentImageIndex - 1));
    }
}

function handleNextImage() {
    if (state.currentImageIndex < state.images.length - 1) {
        // switchImage 函数需要在 main.js 中定义并导出
        import('./main.js').then(main => main.switchImage(state.currentImageIndex + 1));
    }
}

function handleToggleImageDisplay() {
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.translatedDataURL) return;
    state.updateCurrentImageProperty('showOriginal', !currentImage.showOriginal);
    if (currentImage.showOriginal) {
        ui.updateTranslatedImage(currentImage.originalDataURL);
        $('#toggleImageButton').text('显示翻译图');
    } else {
        ui.updateTranslatedImage(currentImage.translatedDataURL);
        $('#toggleImageButton').text('显示原图');
    }
}

function handleImageSizeChange(e) {
    ui.updateImageSizeDisplay(e.target.value);
}

function handleThumbnailClick(e) {
    const index = $(e.currentTarget).data('index');
    if (index !== undefined && index !== state.currentImageIndex) {
        // switchImage 函数需要在 main.js 中定义并导出
        import('./main.js').then(main => main.switchImage(index));
    }
}

function handleDownloadCurrent() {
    main.downloadCurrentImage();
}

function handleDownloadAll() {
    main.downloadAllImages();
}

function handleGlobalSettingChange(event) {
    const currentImage = state.getCurrentImage();
    const changedElement = event.target; // 获取触发事件的元素
    const settingId = changedElement.id; // 获取元素 ID (如 'fontSize', 'fontFamily', etc.)
    let newValue;

    console.log(`全局设置变更: ${settingId}`);

    // 获取新值，并根据类型处理
    if (changedElement.type === 'checkbox') {
        newValue = changedElement.checked;
    } else if (changedElement.type === 'number' || settingId === 'fontSize' || settingId === 'textStrokeWidth') { // 添加 textStrokeWidth
        newValue = parseInt(changedElement.value);
        if (settingId === 'fontSize' && isNaN(newValue)) newValue = state.defaultFontSize;
        if (settingId === 'textStrokeWidth' && (isNaN(newValue) || newValue < 0)) newValue = 0; // 描边宽度不能为负
    } else if (changedElement.type === 'range') {
        newValue = parseFloat(changedElement.value); // 处理滑块
    } else {
        newValue = changedElement.value; // 其他 (select, color, text)
    }

    let settingsUpdated = false; // 用于决定是否触发通用重渲染

    // 更新全局默认值状态（用于新图片）
    switch (settingId) {
        case 'fontSize': state.setDefaultFontSize(newValue); settingsUpdated = true; break;
        case 'fontFamily': state.setDefaultFontFamily(newValue); settingsUpdated = true; break;
        case 'layoutDirection': state.setDefaultLayoutDirection(newValue); settingsUpdated = true; break;
        case 'textColor': state.setDefaultTextColor(newValue); settingsUpdated = true; break;
        case 'fillColor': 
            state.setDefaultFillColor(newValue);
            if (currentImage && !state.isLabelingModeActive) { // 移除描边相关的条件检查
                console.log("全局填充色变更，触发带新填充色的重渲染...");
                main.reRenderWithNewFillColor(newValue); // 调用新的函数
            }
            settingsUpdated = true; // 标记设置已更新
            break;
        // AI视觉OCR设置
        case 'aiVisionProvider': state.setAiVisionProvider(newValue); break;
        case 'aiVisionModelName': state.setAiVisionModelName(newValue); break;
        case 'aiVisionOcrPrompt': state.setAiVisionOcrPrompt(newValue); break;
        // case 'rotationAngle': state.setDefaultRotationAngle(newValue); break; // 如果需要

        // === 新增描边相关的 case START ===
        case 'enableTextStroke':
            // 状态已在 handleEnableTextStrokeChange 中通过 state.setEnableTextStroke 更新
            settingsUpdated = true; // 标记需要重渲染
            break;
        case 'textStrokeColor':
            // 状态已在 handleTextStrokeSettingChange 中通过 state.setTextStrokeColor 更新
            if ($("#enableTextStroke").is(':checked')) settingsUpdated = true;
            break;
        case 'textStrokeWidth':
            // 状态已在 handleTextStrokeSettingChange 中通过 state.setTextStrokeWidth 更新
            if ($("#enableTextStroke").is(':checked')) settingsUpdated = true;
            break;
        // === 新增描边相关的 case END ===
    }

    // --- 确认更新 state.bubbleSettings 的逻辑 ---
    // **只在编辑模式下，全局设置的更改才应该直接修改 state.bubbleSettings**
    // **非编辑模式下，全局设置更改应该只更新图片的基础属性，并在下次 reRender 时生效**
    if (state.editModeActive && state.bubbleSettings && state.bubbleSettings.length > 0) {
        // 编辑模式：将全局更改应用到 state.bubbleSettings (如果适用)
        state.bubbleSettings.forEach(setting => {
            switch (settingId) {
                case 'fontSize':
                    if (!setting.autoFontSize) { setting.fontSize = newValue; settingsUpdated = true; } break;
                case 'autoFontSize':
                     setting.autoFontSize = newValue;
                     setting.fontSize = newValue ? 'auto' : (setting.lastManualFontSize || state.defaultFontSize);
                     if (!newValue) setting.lastManualFontSize = setting.fontSize;
                     settingsUpdated = true;
                     $('#fontSize').prop('disabled', newValue).val(newValue ? '-' : setting.fontSize);
                     break;
                // ... (其他 case 不变: fontFamily, layoutDirection, textColor) ...
                 case 'fontFamily': setting.fontFamily = newValue; settingsUpdated = true; break;
                 case 'layoutDirection': setting.textDirection = newValue; settingsUpdated = true; break;
                 case 'textColor': setting.textColor = newValue; settingsUpdated = true; break;
                 // === 新增：处理描边相关全局参数 START ===
                 case 'enableTextStroke':
                     // 更新所有气泡的描边启用状态
                     setting.enableStroke = newValue;
                     settingsUpdated = true;
                     break;
                 case 'textStrokeColor':
                     // 如果描边已启用，更新所有气泡的描边颜色
                     if ($("#enableTextStroke").is(':checked')) {
                         setting.strokeColor = newValue;
                         settingsUpdated = true;
                     }
                     break;
                 case 'textStrokeWidth':
                     // 如果描边已启用，更新所有气泡的描边宽度
                     if ($("#enableTextStroke").is(':checked')) {
                         setting.strokeWidth = newValue;
                         settingsUpdated = true;
                     }
                     break;
                 // === 新增：处理描边相关全局参数 END ===
            }
        });
        if (settingsUpdated) {
            state.setBubbleSettings([...state.bubbleSettings]); // 更新引用
            // 同时更新当前图片的 bubbleSettings 备份
             if (currentImage) {
                 currentImage.bubbleSettings = JSON.parse(JSON.stringify(state.bubbleSettings));
             }
        }
    } else if (currentImage && !state.editModeActive) {
        // 非编辑模式：更新当前图片的基础属性
        settingsUpdated = true; // 标记需要重渲染
        
        // 首先总是更新图片的全局属性，这些是基础设置
        switch (settingId) {
             case 'fontSize': state.updateCurrentImageProperty('fontSize', newValue); break;
             case 'autoFontSize':
                 state.updateCurrentImageProperty('autoFontSize', newValue);
                 state.updateCurrentImageProperty('fontSize', newValue ? 'auto' : (currentImage.fontSize !== 'auto' ? currentImage.fontSize : state.defaultFontSize));
                 $('#fontSize').prop('disabled', newValue).val(newValue ? '-' : state.getCurrentImage().fontSize);
                 break;
             case 'fontFamily': state.updateCurrentImageProperty('fontFamily', newValue); break;
             case 'layoutDirection': state.updateCurrentImageProperty('layoutDirection', newValue); break; // 或 textDirection
             case 'textColor': state.updateCurrentImageProperty('textColor', newValue); break;
             case 'rotationAngle': state.updateCurrentImageProperty('rotationAngle', newValue); break;
             case 'fillColor': state.updateCurrentImageProperty('fillColor', newValue); break;
             case 'blendEdges': state.updateCurrentImageProperty('blendEdges', newValue); break;
            // 修复设置通常是全局的，不直接保存在图片上，除非你有意设计如此
        }
        
        // 检查是否存在个性化气泡设置，如果存在，只更新对应的参数，而不是清空
        if (currentImage.bubbleSettings && Array.isArray(currentImage.bubbleSettings) && 
            currentImage.bubbleSettings.length === (currentImage.bubbleCoords ? currentImage.bubbleCoords.length : 0)) {
            
            console.log(`检测到个性化气泡设置，只更新 ${settingId} 属性，保留其他个性化设置`);
            
            // 深拷贝当前设置
            const updatedBubbleSettings = JSON.parse(JSON.stringify(currentImage.bubbleSettings));
            
            // 根据变更的设置类型更新对应属性
            updatedBubbleSettings.forEach(setting => {
                switch (settingId) {
                    case 'fontSize':
                        if (!setting.autoFontSize) { setting.fontSize = newValue; }
                        break;
                    case 'autoFontSize':
                        setting.autoFontSize = newValue;
                        if (newValue) {
                            setting.fontSize = 'auto';
                        } else if (setting.lastManualFontSize) {
                            setting.fontSize = setting.lastManualFontSize;
                        }
                        break;
                    case 'fontFamily': 
                        setting.fontFamily = newValue; 
                        break;
                    case 'layoutDirection': 
                        setting.textDirection = newValue; 
                        break;
                    case 'textColor': 
                        setting.textColor = newValue; 
                        break;
                    case 'rotationAngle': 
                        setting.rotationAngle = newValue; 
                        break;
                    // === 新增：处理描边相关全局参数 START ===
                    case 'enableTextStroke':
                        // 更新所有气泡的描边启用状态
                        setting.enableStroke = newValue;
                        break;
                    case 'textStrokeColor':
                        // 如果描边已启用，更新所有气泡的描边颜色
                        if ($("#enableTextStroke").is(':checked')) {
                            setting.strokeColor = newValue;
                        }
                        break;
                    case 'textStrokeWidth':
                        // 如果描边已启用，更新所有气泡的描边宽度
                        if ($("#enableTextStroke").is(':checked')) {
                            setting.strokeWidth = newValue;
                        }
                        break;
                    // === 新增：处理描边相关全局参数 END ===
                    // 其他属性保持不变
                }
            });
            
            // 更新气泡设置
            state.updateCurrentImageProperty('bubbleSettings', updatedBubbleSettings);
        } else {
            // 没有个性化设置，清空bubbleSettings表示使用全局设置
            state.updateCurrentImageProperty('bubbleSettings', null);
        }
    }
    // --- 结束确认 ---

    // --- 触发重渲染 ---
    // 只有在当前有已翻译图片并且确实有设置被更新时才重渲染
    // 并且 *不是* fillColor 的改动（因为它已经通过 reRenderWithNewFillColor 处理了）
    if (currentImage && currentImage.translatedDataURL && settingsUpdated && settingId !== 'fillColor') {
        console.log(`全局设置变更 (${settingId}) 后，准备重新渲染...`);
        // 确保 editModeActive 为 false 时，reRenderFullImage 能正确处理全局设置
        // 或者，如果 reRenderFullImage 总是从 state.js 读取描边设置，则不需要额外操作
        if (state.editModeActive) {
            editMode.reRenderFullImage();
        } else {
            // 非编辑模式下的重渲染，可能需要一个通用的重渲染函数或修改 reRenderFullImage
            // 暂时假设 reRenderFullImage 能处理非编辑模式下的全局参数
            editMode.reRenderFullImage();
        }
    } else if (!settingsUpdated && settingId !== 'fillColor') {
         console.log("全局设置变更未导致需要重渲染的更新。");
    }
}

function handleInpaintingMethodChange() {
    const repairMethod = $(this).val();
    // 根据新的函数签名调用，只在选择 'lama' 时传递 true 给第一个参数
    ui.toggleInpaintingOptions(repairMethod === 'lama', repairMethod === 'false');
    // 这个变化不直接触发重渲染，下次翻译或手动重渲染时生效
}

function handleInpaintingStrengthChange(e) {
    $('#inpaintingStrengthValue').text(e.target.value);
    // 强度变化不实时重渲染，在下次翻译或手动重渲染时生效
}

function handleModelProviderChange() {
    const selectedProvider = $(this).val().toLowerCase(); // 转小写以便比较
    const isLocalDeployment = selectedProvider === 'ollama' || selectedProvider === 'sakura';
    
    // 更新API Key输入框状态 (ui.js中的函数已更新以处理custom_openai)
    ui.updateApiKeyInputState(
        isLocalDeployment,
        isLocalDeployment ? '本地部署无需API Key' : '请输入API Key'
    );

    // 切换特定服务商的UI元素
    ui.toggleOllamaUI(selectedProvider === 'ollama');
    ui.toggleSakuraUI(selectedProvider === 'sakura');
    ui.toggleCaiyunUI(selectedProvider === 'caiyun');
    ui.toggleBaiduTranslateUI(selectedProvider === 'baidu_translate');
    ui.toggleYoudaoTranslateUI(selectedProvider === 'youdao_translate');
    // --- 新增：切换自定义 Base URL 输入框 ---
    ui.toggleCustomOpenAiUI(selectedProvider === 'custom_openai'); // 使用常量会更好
    // ------------------------------------

    // 清理/加载模型列表
    if (selectedProvider === 'ollama') {
        fetchOllamaModels();
        $('#sakuraModelsList').empty().hide(); // 隐藏Sakura列表
    } else if (selectedProvider === 'sakura') {
        fetchSakuraModels();
        $('#ollamaModelsList').empty().hide(); // 隐藏Ollama列表
    } else {
        // 对于其他云服务商（包括新增的Gemini），隐藏本地模型列表
        $('#ollamaModelsList').empty().hide();
        $('#sakuraModelsList').empty().hide();
    }
    
    // 更新模型建议
    if (selectedProvider && 
        selectedProvider !== 'baidu_translate' && 
        selectedProvider !== 'youdao_translate' &&
        selectedProvider !== 'ollama' &&
        selectedProvider !== 'sakura' &&
        selectedProvider !== 'custom_openai' // 自定义服务商不显示历史建议
        ) {
        api.getUsedModelsApi($(this).val())
            .then(response => ui.updateModelSuggestions(response.models))
            .catch(error => console.error("获取模型建议失败:", error));
    } else {
        ui.updateModelSuggestions([]);
    }

    // 调整标签
    if (selectedProvider === 'baidu_translate') {
        $('label[for="apiKey"]').text('App ID:');
        $('#apiKey').attr('placeholder', '请输入百度翻译App ID');
        $('label[for="modelName"]').text('App Key:');
        $('#modelName').attr('placeholder', '请输入百度翻译App Key');
    } else if (selectedProvider === 'youdao_translate') {
        $('label[for="apiKey"]').text('App Key:');
        $('#apiKey').attr('placeholder', '请输入有道翻译应用ID');
        $('label[for="modelName"]').text('App Secret:');
        $('#modelName').attr('placeholder', '请输入有道翻译应用密钥');
    } else if (selectedProvider === 'custom_openai') { // --- 新增对自定义服务商的标签处理 ---
        $('label[for="apiKey"]').text('API Key:');
        $('#apiKey').attr('placeholder', '请输入 API Key');
        $('label[for="modelName"]').text('模型名称:'); // 模型名称对于自定义服务也是必须的
        $('#modelName').attr('placeholder', '例如: gpt-3.5-turbo');
    } else {
        // 恢复默认标签
        $('label[for="apiKey"]').text('API Key:');
        $('#apiKey').attr('placeholder', '请输入API Key');
        $('label[for="modelName"]').text('大模型型号:');
        $('#modelName').attr('placeholder', '请输入模型型号');
    }
}

function handleModelNameFocus() {
    const modelProvider = $('#modelProvider').val();
    if (modelProvider && 
        modelProvider !== 'ollama' && 
        modelProvider !== 'sakura' && 
        modelProvider !== 'baidu_translate' && 
        modelProvider !== 'youdao_translate') { // 本地模型和敏感API不显示历史建议
        api.getUsedModelsApi(modelProvider)
            .then(response => ui.updateModelSuggestions(response.models))
            .catch(error => console.error("获取模型建议失败:", error));
    } else {
        ui.updateModelSuggestions([]); // 清空建议
    }
}

function handleModelNameBlur() {
    // 移除延迟隐藏，让点击事件处理器来决定何时隐藏
    // 不再使用: setTimeout(() => $("#model-suggestions").hide(), 200);
}

function handleModelButtonClick(e) {
    const modelName = $(e.target).text();
    $('#modelName').val(modelName);
    $('.model-button').removeClass('selected');
    $(e.target).addClass('selected');
    
    // 获取当前选择的模型提供商
    const modelProvider = $('#modelProvider').val();
    
    // 保存模型信息到历史记录
    if (modelProvider && modelName) {
        api.saveModelInfoApi(modelProvider, modelName)
            .then(() => {
                console.log(`模型信息已保存: ${modelProvider}/${modelName}`);
            })
            .catch(error => {
                console.error(`保存模型信息失败: ${error.message}`);
            });
    }
}


function handleSavePrompt() {
    const promptName = $("#promptName").val();
    const promptContent = $("#promptContent").val();
    const remember = $("#rememberPrompt").is(':checked');

    if (remember) {
        if (!promptName) {
            ui.showGeneralMessage("请为要保存的提示词输入名称。", "warning");
            return;
        }
        ui.showLoading("保存提示词...");
        api.savePromptApi(promptName, promptContent)
            .then(response => {
                ui.hideLoading();
                ui.showGeneralMessage("提示词保存成功！", "success");
                // 重新加载提示词列表
                import('./main.js').then(main => main.initializePromptSettings());
            })
            .catch(error => {
                ui.hideLoading();
                ui.showGeneralMessage(`保存提示词失败: ${error.message}`, "error");
            });
    } else {
        // 仅应用，不保存
        state.setPromptState(promptContent, state.defaultPromptContent, state.savedPromptNames);
        ui.showGeneralMessage("提示词已应用（未保存）。", "info");
    }
}

function handleEnableTextboxPromptChange(e) {
    const use = e.target.checked;
    state.setUseTextboxPrompt(use);
    $("#textboxPromptContent").toggle(use);
    $("#textboxPromptManagement").toggle(use);
    $("#saveTextboxPromptButton").toggle(use);
    $("#textbox-prompt-dropdown-container").toggle(use);
    if (use && !$("#textboxPromptContent").val()) {
        $("#textboxPromptContent").val(state.defaultTextboxPromptContent);
        state.currentTextboxPromptContent = state.defaultTextboxPromptContent;
    }
}

function handleSaveTextboxPrompt() {
    const promptName = $("#textboxPromptName").val();
    const promptContent = $("#textboxPromptContent").val();
    const remember = $("#rememberTextboxPrompt").is(':checked');

    if (remember) {
        if (!promptName) {
            ui.showGeneralMessage("请为要保存的文本框提示词输入名称。", "warning");
            return;
        }
        ui.showLoading("保存文本框提示词...");
        api.saveTextboxPromptApi(promptName, promptContent)
            .then(response => {
                ui.hideLoading();
                ui.showGeneralMessage("文本框提示词保存成功！", "success");
                // 重新加载提示词列表
                import('./main.js').then(main => main.initializeTextboxPromptSettings());
            })
            .catch(error => {
                ui.hideLoading();
                ui.showGeneralMessage(`保存文本框提示词失败: ${error.message}`, "error");
            });
    } else {
        state.setTextboxPromptState(promptContent, state.defaultTextboxPromptContent, state.savedTextboxPromptNames);
        ui.showGeneralMessage("文本框提示词已应用（未保存）。", "info");
    }
}

function handleCleanDebugFiles() {
    if (confirm('确定要清理所有调试文件和下载临时文件吗？这将释放磁盘空间，但不会影响您的翻译图片。')) {
        ui.showLoading("正在清理文件...");
        api.cleanDebugFilesApi()
            .then(response => {
                ui.hideLoading();
                // 消息可能包含多个结果，用换行显示更清晰
                const formattedMessage = response.message.replace(/\s*\|\s*/g, '<br>');
                ui.showGeneralMessage(formattedMessage, response.success ? "success" : "error");
            })
            .catch(error => {
                ui.hideLoading();
                ui.showGeneralMessage(`清理文件失败: ${error.message}`, "error");
            });
    }
}

function handleGlobalKeyDown(e) {
    // 检查是否在文本输入框中
    const isInTextInput = $(e.target).is('input[type="text"], textarea, [contenteditable="true"]') || 
                          $(e.target).attr('id') === 'bubbleTextEditor';
    
    // 编辑模式下禁用部分快捷键，或赋予不同功能
    if (state.editModeActive) {
        // 只有当不在文本输入框中时，左右箭头才切换气泡
        if (!isInTextInput) {
            if (e.keyCode == 37) { // Left Arrow
                e.preventDefault();
                editMode.selectPrevBubble();
                return;
            } else if (e.keyCode == 39) { // Right Arrow
                e.preventDefault();
                editMode.selectNextBubble();
                return;
            }
        }
        // 可以添加其他编辑模式快捷键，如 Alt+Up/Down 调整字号
    }

    // 如果在文本输入框中，不拦截键盘事件，让浏览器处理默认行为
    if (isInTextInput) return;

    // 非编辑模式下的快捷键
    if (e.altKey) {
        const fontSizeInput = $("#fontSize"); // 获取全局字号输入框
        if (e.keyCode == 38) { // Up Arrow
            e.preventDefault();
            if (!fontSizeInput.prop('disabled')) { // 仅在非自动字号时调整
                let currentFontSize = parseInt(fontSizeInput.val()) || state.defaultFontSize;
                fontSizeInput.val(currentFontSize + 1).trigger('change');
            }
        } else if (e.keyCode == 40) { // Down Arrow
            e.preventDefault();
             if (!fontSizeInput.prop('disabled')) {
                let currentFontSize = parseInt(fontSizeInput.val()) || state.defaultFontSize;
                fontSizeInput.val(Math.max(10, currentFontSize - 1)).trigger('change');
             }
        } else if (e.keyCode == 37) { // Left Arrow
            e.preventDefault();
            handlePrevImage();
        } else if (e.keyCode == 39) { // Right Arrow
            e.preventDefault();
            handleNextImage();
        }
    }
}

function handleThemeToggle() {
    const body = document.body;
    if (body.classList.contains('dark-mode')) {
        body.classList.remove('dark-mode');
        body.classList.add('light-mode');
        localStorage.setItem('themeMode', 'light');
    } else {
        body.classList.remove('light-mode');
        body.classList.add('dark-mode');
        localStorage.setItem('themeMode', 'dark');
    }
}

function handleDonateClick() {
    $("#donateModal").css("display", "block");
}

function handleDonateClose() {
    $("#donateModal").css("display", "none");
}

function handleWindowClickForModal(event) {
    // 检查点击的目标是否是模态框本身 (即背景遮罩)
    const modal = $(".modal:visible");
    if (modal.length > 0 && event.target == modal[0]) {
        modal.hide();
    }
    
    // 检查是否点击了赞助模态窗口的背景
    if (event.target === $("#donateModal")[0]) {
        $("#donateModal").css("display", "none");
    }
}

// === 新增："加载/管理会话"按钮的处理函数 ===
function handleLoadSessionClick() {
    session.showSessionManager(); // 调用 session.js 中的函数
}
// === 结束新增 ===

// --- 编辑模式事件处理 ---

function handleBubbleItemClick(e) {
    const index = parseInt($(e.currentTarget).data('index'));
    if (!isNaN(index)) {
        editMode.selectBubble(index);
    }
}

// 文本编辑器输入事件 (实时更新状态，但不触发渲染)
function handleBubbleEditorChange(e) {
    const index = state.selectedBubbleIndex;
    if (index >= 0) {
        state.updateSingleBubbleSetting(index, { text: e.target.value });
        ui.updateBubbleListUI(); // 更新预览文本
        // 添加延迟渲染预览，避免频繁渲染影响性能
        clearTimeout(window.textEditTimer);
        window.textEditTimer = setTimeout(() => {
            editMode.renderBubblePreview(index); // 触发预览渲染
        }, 500); // 500ms延迟，在用户停止输入后再渲染
    }
}

// 气泡设置变更事件 (字体、大小、方向、颜色) - 触发预览渲染
function handleBubbleSettingChange(event) {
    const index = state.selectedBubbleIndex;
    if (index >= 0) {
        const currentSetting = state.bubbleSettings[index];
        const isAuto = $('#autoBubbleFontSize').is(':checked');
        let fromAutoSwitch = false;

        // 处理自动字号切换
        if (event && event.target.id === 'autoBubbleFontSize') {
            const isPrevAuto = currentSetting.autoFontSize;
            if (isPrevAuto && !isAuto) {
                fromAutoSwitch = true;
                console.log("单气泡从自动字号切换到手动字号，使用上次手动字号或默认");
                const lastManualSize = currentSetting.lastManualFontSize || state.defaultFontSize;
                $('#bubbleFontSize').prop('disabled', false).val(lastManualSize);
            } else if (!isPrevAuto && isAuto) {
                // 从手动切换到自动，记录当前手动字号
                currentSetting.lastManualFontSize = parseInt($('#bubbleFontSize').val()) || state.defaultFontSize;
            }
        }
        
        const newFontSize = isAuto ? 'auto' : (fromAutoSwitch ? (currentSetting.lastManualFontSize || state.defaultFontSize) : parseInt($('#bubbleFontSize').val()));

        const settingUpdate = {
            fontSize: newFontSize,
            autoFontSize: isAuto,
            fontFamily: $('#bubbleFontFamily').val(),
            textDirection: $('#bubbleTextDirection').val(),
            textColor: $('#bubbleTextColor').val(),
            fillColor: $('#bubbleFillColor').val(), // <--- 获取独立填充色
            // === 新增：获取描边设置 START ===
            enableStroke: $('#bubbleEnableStroke').is(':checked'),
            strokeColor: $('#bubbleStrokeColor').val(),
            strokeWidth: parseInt($('#bubbleStrokeWidth').val()) || 0
            // === 新增：获取描边设置 END ===
        };

        state.updateSingleBubbleSetting(index, settingUpdate);
        // 触发预览 (renderBubblePreview 内部调用 reRenderFullImage)
        // reRenderFullImage 内部现在会处理独立填充色
        import('./edit_mode.js').then(editMode => {
            editMode.renderBubblePreview(index);
        });
    }
}

function handleBubbleRotationChange(e) {
    const index = state.selectedBubbleIndex;
    if (index >= 0) {
        const angle = parseInt(e.target.value);
        $('#bubbleRotationAngleValue').text(angle + '°'); // 更新显示
        state.updateSingleBubbleSetting(index, { rotationAngle: angle });
        // 使用延迟避免过于频繁的渲染
        clearTimeout(rotationTimer);
        rotationTimer = setTimeout(() => {
            editMode.renderBubblePreview(index);
        }, 300); // 300ms 延迟
    }
}

function handleBubblePositionChange(e) {
    const index = state.selectedBubbleIndex;
    if (index >= 0) {
        const x = parseInt($('#positionOffsetX').val());
        const y = parseInt($('#positionOffsetY').val());
        $('#positionOffsetXValue').text(x);
        $('#positionOffsetYValue').text(y);
        state.updateSingleBubbleSetting(index, { position: { x: x, y: y } });
        // 使用延迟
        clearTimeout(positionTimer);
        positionTimer = setTimeout(() => {
            editMode.renderBubblePreview(index);
        }, 300);
    }
}

function handleApplyBubbleEdit() {
    if (state.selectedBubbleIndex >= 0) {
        // 最终确认应用更改，实际上预览已经完成了渲染
        // 这里可以加一个确认提示或保存状态的逻辑（如果需要）
        ui.showGeneralMessage(`气泡 ${state.selectedBubbleIndex + 1} 的更改已应用`, "success", false, 2000);
        // 确保状态已保存到当前图片的 bubbleSettings
        const currentImage = state.getCurrentImage();
        if(currentImage) {
            currentImage.bubbleSettings = JSON.parse(JSON.stringify(state.bubbleSettings));
        }
    }
}

function handleApplyToAllBubbles() {
    if (state.selectedBubbleIndex >= 0) {
        editMode.applySettingsToAllBubbles(); // 调用编辑模式的函数
    }
}

function handleResetBubbleEdit() {
    if (state.selectedBubbleIndex >= 0) {
        editMode.resetCurrentBubble(); // 调用编辑模式的函数
    }
}

// 位置按钮按下/松开/离开的处理
let positionInterval = null;
function handlePositionButtonMouseDown(e) {
    const direction = e.target.id; // 'moveUp', 'moveDown', etc.
    const adjust = () => editMode.adjustPosition(direction);
    adjust(); // 立即执行一次
    positionInterval = setInterval(adjust, 150); // 每 150ms 重复
}
function handlePositionButtonMouseUp() {
    if (positionInterval) {
        clearInterval(positionInterval);
        positionInterval = null;
    }
}
function handleResetPosition() {
    editMode.resetPosition();
}

// --- 插件管理 ---
function handleManagePluginsClick() {
    console.log("打开插件管理窗口");
    const modal = $("#pluginManagerModal");
    const container = $("#pluginListContainer");
    container.html("<p>正在加载插件列表...</p>"); // 显示加载提示
    modal.css("display", "block"); // 先显示模态框

    // --- 修改: 先获取默认状态，再获取插件列表 ---
    let fetchedDefaultStates = {};
    api.getPluginDefaultStatesApi()
        .then(stateResponse => {
            if (stateResponse.success) {
                fetchedDefaultStates = stateResponse.states || {};
            } else {
                console.warn("获取插件默认状态失败:", stateResponse.error);
                // 即使失败也继续加载列表，只是复选框状态可能不正确
            }
            // 然后获取插件列表
            return api.getPluginsApi();
        })
        .then(response => {
            if (response.success) {
                // 将获取到的默认状态传递给渲染函数
                ui.renderPluginList(response.plugins, fetchedDefaultStates);
            } else {
                throw new Error(response.error || "无法加载插件列表");
            }
        })
        .catch(error => {
            container.html(`<p class="error">加载插件列表失败: ${error.message}</p>`);
        });
    // --------------------------------------------
}

function handlePluginModalClose() {
    $("#pluginManagerModal").css("display", "none");
}

function handleWindowClickForPluginModal(event) {
    if (event.target === $("#pluginManagerModal")[0]) {
        $("#pluginManagerModal").css("display", "none");
    }
}

function handlePluginToggleChange(e) {
    const checkbox = $(e.target);
    const pluginItem = checkbox.closest('.plugin-item');
    const pluginName = pluginItem.data('plugin-name');
    const isEnabled = checkbox.prop('checked');
    const label = checkbox.parent();

    label.find('input').prop('disabled', true); // 禁用开关防止重复点击
    label.contents().last().replaceWith(isEnabled ? ' 启用中...' : ' 禁用中...'); // 更新文本

    const action = isEnabled ? api.enablePluginApi(pluginName) : api.disablePluginApi(pluginName);

    action.then(response => {
        if (response.success) {
            label.contents().last().replaceWith(isEnabled ? ' 已启用' : ' 已禁用');
            ui.showGeneralMessage(response.message, "success", false, 2000);
        } else {
            throw new Error(response.error || "操作失败");
        }
    }).catch(error => {
        ui.showGeneralMessage(`操作插件 '${pluginName}' 失败: ${error.message}`, "error");
        // 恢复复选框状态
        checkbox.prop('checked', !isEnabled);
        label.contents().last().replaceWith(!isEnabled ? ' 已启用' : ' 已禁用');
    }).finally(() => {
        label.find('input').prop('disabled', false); // 重新启用开关
    });
}

function handlePluginDeleteClick(e) {
    const button = $(e.target);
    const pluginItem = button.closest('.plugin-item');
    const pluginName = pluginItem.data('plugin-name');

    if (confirm(`确定要删除插件 "${pluginName}" 吗？\n这个操作会物理删除插件文件，并且需要重启应用才能完全生效。`)) {
        button.prop('disabled', true).text('删除中...');
        api.deletePluginApi(pluginName)
            .then(response => {
                if (response.success) {
                    pluginItem.fadeOut(300, function() { $(this).remove(); });
                    ui.showGeneralMessage(response.message, "success");
                } else {
                    throw new Error(response.error || "删除失败");
                }
            })
            .catch(error => {
                ui.showGeneralMessage(`删除插件 '${pluginName}' 失败: ${error.message}`, "error");
                button.prop('disabled', false).text('删除');
            });
    }
}

// --- 插件配置事件处理 ---

function handlePluginSettingsClick(e) {
    const button = $(e.target);
    const pluginItem = button.closest('.plugin-item');
    const pluginName = pluginItem.data('plugin-name');

    if (!pluginName) return;

    ui.showLoading("加载配置...");
    // 1. 获取配置规范
    api.getPluginConfigSchemaApi(pluginName)
        .then(schemaResponse => {
            if (!schemaResponse.success) throw new Error(schemaResponse.error || "无法获取配置规范");
            const schema = schemaResponse.schema;
            // 2. 获取当前配置值
            return api.getPluginConfigApi(pluginName).then(configResponse => {
                if (!configResponse.success) throw new Error(configResponse.error || "无法获取当前配置");
                return { schema, config: configResponse.config };
            });
        })
        .then(({ schema, config }) => {
            ui.hideLoading();
            // 3. 显示配置模态框
            ui.showPluginConfigModal(pluginName, schema, config);
        })
        .catch(error => {
            ui.hideLoading();
            ui.showGeneralMessage(`加载插件 '${pluginName}' 配置失败: ${error.message}`, "error");
        });
}

function handlePluginConfigModalClose() {
    $('#pluginConfigModal').remove(); // 关闭时移除模态框
}

function handlePluginConfigSave(e) {
    e.preventDefault(); // 阻止表单默认提交
    const form = $(e.target);
    const pluginName = form.closest('.plugin-modal-content').find('h3').text().replace('插件设置: ', '');
    const configData = {};

    // 从表单收集数据
    const formData = new FormData(form[0]);
    const schema = []; // 需要从某处获取 schema 来正确处理类型，或者后端处理类型转换
    // 简单收集：
    formData.forEach((value, key) => {
         // 处理 checkbox (未选中时 FormData 不会包含它)
         const inputElement = form.find(`[name="${key}"]`);
         if (inputElement.is(':checkbox')) {
             configData[key] = inputElement.is(':checked');
         } else {
             configData[key] = value;
         }
    });
    // 确保所有 boolean 字段都被包含 (即使未选中)
    form.find('input[type="checkbox"]').each(function() {
        const name = $(this).attr('name');
        if (!configData.hasOwnProperty(name)) {
            configData[name] = false;
        }
    });

    console.log(`保存插件 '${pluginName}' 配置:`, configData);
    ui.showLoading("保存配置...");

    api.savePluginConfigApi(pluginName, configData)
        .then(response => {
            ui.hideLoading();
            if (response.success) {
                ui.showGeneralMessage(response.message, "success");
                $('#pluginConfigModal').remove(); // 关闭模态框
                // 注意：配置更改可能需要重启应用或特定操作才能完全生效
            } else {
                throw new Error(response.error || "保存失败");
            }
        })
        .catch(error => {
            ui.hideLoading();
            ui.showGeneralMessage(`保存插件 '${pluginName}' 配置失败: ${error.message}`, "error");
        });
}

// --- 新增: 处理默认启用状态变化的事件处理器 ---
function handlePluginDefaultStateChange(e) {
    const checkbox = $(e.target);
    const pluginName = checkbox.data('plugin-name');
    const isEnabled = checkbox.prop('checked');
    const label = checkbox.parent(); // 获取父 label 元素

    if (!pluginName) {
        console.error("无法获取插件名称从 data-plugin-name 属性");
        return;
    }

    console.log(`用户更改插件 '${pluginName}' 默认启用状态为: ${isEnabled}`);

    // 临时禁用复选框，防止重复点击
    checkbox.prop('disabled', true);
    // 可选：添加视觉反馈，比如给 label 添加一个 loading class
    label.css('opacity', 0.6);

    api.setPluginDefaultStateApi(pluginName, isEnabled)
        .then(response => {
            if (response.success) {
                ui.showGeneralMessage(response.message, "success", false, 2000);
            } else {
                throw new Error(response.error || "设置默认状态失败");
            }
        })
        .catch(error => {
            ui.showGeneralMessage(`设置插件 '${pluginName}' 默认状态失败: ${error.message}`, "error");
            // 操作失败，恢复复选框到之前的状态
            checkbox.prop('checked', !isEnabled);
        })
        .finally(() => {
            // 无论成功或失败，都重新启用复选框并移除视觉反馈
            checkbox.prop('disabled', false);
            label.css('opacity', 1);
        });
}
// -------------------------------------------

// --- 新增：标注模式 ---
function handleToggleLabelingMode() {
    // 根据当前状态切换
    if (state.isLabelingModeActive) {
        labelingMode.exitLabelingMode();
    } else {
        labelingMode.enterLabelingMode();
    }
}

// === 新增：保存会话按钮的处理函数 ===
function handleSaveCurrentSession() {
    session.triggerSaveCurrentSession(); // 调用新的保存函数
}

/**
 * 处理"另存为"按钮点击
 */
function handleSaveAsSession() {
    session.triggerSaveAsSession(); // 调用重命名后的另存为函数
}

/**
 * 处理OCR引擎变更
 */
function handleOcrEngineChange() {
    const ocr_engine = $("#ocrEngine").val();
    // 先清除之前的样式
    $("#baiduOcrOptions, #aiVisionOcrOptions").css({
        'margin-bottom': '15px',
        'padding': '10px',
        'border-radius': '8px',
        'background-color': 'rgba(0,0,0,0.02)'
    });
    
    // 隐藏所有OCR选项
    $("#baiduOcrOptions").hide();
    $("#aiVisionOcrOptions").hide();
    
    // 根据选择显示对应的OCR设置选项
    if (ocr_engine === 'baidu_ocr') {
        // 显示百度OCR设置
        $("#baiduOcrOptions").show();
    } else if (ocr_engine === 'ai_vision') {
        // 显示AI视觉OCR设置
        $("#aiVisionOcrOptions").show();
    }
    
    // 保存OCR引擎状态
    state.setOcrEngine(ocr_engine);
    
    // 强制父容器重新计算高度
    setTimeout(() => {
        $("#font-settings .collapsible-content").css('height', 'auto');
        if ($("#font-settings .collapsible-content").is(":visible")) {
            $("#font-settings .collapsible-content").scrollTop(0);
        }
    }, 10);
    
    handleGlobalSettingChange();
}

/**
 * 处理 "删除选中框" 按钮点击 (修改后 - 修补干净背景并强制重渲染)
 */
function handleDeleteSelectedBoxClick() {
    import('./labeling_mode.js').then(module => {
        module.handleDeleteSelectedBoxClick();
    });
}

// 添加OCR引擎切换处理
document.getElementById('ocrEngine').addEventListener('change', function(e) {
    const ocrEngine = e.target.value;
    const baiduOcrOptions = document.getElementById('baiduOcrOptions');
    const aiVisionOcrOptions = document.getElementById('aiVisionOcrOptions');
    
    // 隐藏所有OCR引擎特定选项
    baiduOcrOptions.style.display = 'none';
    aiVisionOcrOptions.style.display = 'none';
    
    // 根据选择的OCR引擎显示对应选项
    if (ocrEngine === 'baidu_ocr') {
        baiduOcrOptions.style.display = 'block';
    } else if (ocrEngine === 'ai_vision') {
        aiVisionOcrOptions.style.display = 'block';
    }
    
    // 更新状态
    import('./state.js').then(state => {
        state.setOcrEngine(ocrEngine);
    });
});

// 添加百度OCR测试按钮事件
document.getElementById('testBaiduOcrButton').addEventListener('click', async function() {
    const apiKey = document.getElementById('baiduApiKey').value.trim();
    const secretKey = document.getElementById('baiduSecretKey').value.trim();
    
    if (!apiKey || !secretKey) {
        showMessage('请输入百度OCR的API Key和Secret Key', 'error');
        return;
    }
    
    showMessage('正在测试百度OCR连接...', 'info');
    try {
        const { testBaiduOcrConnectionApi } = await import('./api.js');
        const result = await testBaiduOcrConnectionApi(apiKey, secretKey);
        
        if (result.success) {
            showMessage('百度OCR连接测试成功: ' + result.message, 'success');
        } else {
            showMessage('百度OCR连接测试失败: ' + result.message, 'error');
        }
    } catch (error) {
        showMessage('百度OCR连接测试出错: ' + (error.message || '未知错误'), 'error');
    }
});

// 添加AI视觉OCR测试按钮事件
document.getElementById('testAiVisionOcrButton').addEventListener('click', async function() {
    const provider = document.getElementById('aiVisionProvider').value.trim();
    const apiKey = document.getElementById('aiVisionApiKey').value.trim();
    const modelName = document.getElementById('aiVisionModelName').value.trim();
    const prompt = document.getElementById('aiVisionOcrPrompt').value.trim();
    
    if (!provider || !apiKey || !modelName) {
        showMessage('请输入AI视觉OCR的服务商、API Key和模型名称', 'error');
        return;
    }
    
    showMessage('正在测试AI视觉OCR连接...', 'info');
    try {
        const { testAiVisionOcrApi } = await import('./api.js');
        const result = await testAiVisionOcrApi(provider, apiKey, modelName, prompt);
        
        if (result.success) {
            showMessage('AI视觉OCR测试成功: ' + result.message, 'success');
        } else {
            showMessage('AI视觉OCR测试失败: ' + result.message, 'error');
        }
    } catch (error) {
        showMessage('AI视觉OCR测试出错: ' + (error.message || '未知错误'), 'error');
    }
});

// 处理AI视觉OCR服务商变更
document.getElementById('aiVisionProvider').addEventListener('change', async function(e) {
    const provider = e.target.value;
    
    // 保存到state
    const { setAiVisionProvider } = await import('./state.js');
    setAiVisionProvider(provider);
    
    // 新增：处理自定义Base URL输入框的显示/隐藏
    const { toggleCustomAiVisionBaseUrlUI } = await import('./ui.js');
    const { CUSTOM_AI_VISION_PROVIDER_ID_FRONTEND } = await import('./constants.js');
    
    // 显示/隐藏自定义Base URL输入框
    toggleCustomAiVisionBaseUrlUI(provider === CUSTOM_AI_VISION_PROVIDER_ID_FRONTEND);
    
    // 如果是火山引擎，获取历史模型建议
    if (provider === 'volcano') {
        const { getUsedModelsApi } = await import('./api.js');
        const { updateModelSuggestions } = await import('./ui.js');
        
        try {
            const response = await getUsedModelsApi('volcano');
            if (response && response.models) {
                // 更新模型建议列表，用于AI视觉OCR的火山引擎
                const modelInput = document.getElementById('aiVisionModelName');
                // 使用数据列表提供建议
                let datalistId = 'aiVisionModelsList';
                let datalist = document.getElementById(datalistId);
                
                if (!datalist) {
                    datalist = document.createElement('datalist');
                    datalist.id = datalistId;
                    document.body.appendChild(datalist);
                    modelInput.setAttribute('list', datalistId);
                }
                
                // 清空现有选项
                datalist.innerHTML = '';
                
                // 添加新选项
                response.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    datalist.appendChild(option);
                });
            }
        } catch (error) {
            console.error("获取火山引擎模型建议失败:", error);
        }
    }
    
    handleGlobalSettingChange();
});

// 处理AI视觉OCR模型名称变更
document.getElementById('aiVisionModelName').addEventListener('input', function(e) {
    const modelName = e.target.value.trim();
    import('./state.js').then(state => {
        state.setAiVisionModelName(modelName);
    });
});

// 处理AI视觉OCR提示词变更
document.getElementById('aiVisionOcrPrompt').addEventListener('input', function(e) {
    const prompt = e.target.value.trim();
    import('./state.js').then(state => {
        state.setAiVisionOcrPrompt(prompt);
    });
});

// 处理自定义AI视觉Base URL变更
document.getElementById('customAiVisionBaseUrl').addEventListener('input', function(e) {
    const baseUrl = e.target.value.trim();
    import('./state.js').then(state => {
        state.setCustomAiVisionBaseUrl(baseUrl);
    });
});

// 修改 handleTestAiVisionOcr 函数
async function handleTestAiVisionOcr() {
    const provider = $("#aiVisionProvider").val();
    const apiKey = $("#aiVisionApiKey").val();
    const modelName = $("#aiVisionModelName").val();
    const prompt = $("#aiVisionOcrPrompt").val();

    if(!apiKey) {
        ui.showGeneralMessage("请输入API Key", "error");
        return;
    }
    if(!modelName) {
        ui.showGeneralMessage("请输入模型名称", "error");
        return;
    }

    // VVVVVV 新增：如果选择自定义服务，检查 Base URL 是否填写 VVVVVV
    if (provider === constants.CUSTOM_AI_VISION_PROVIDER_ID_FRONTEND) {
        const customBaseUrl = $("#customAiVisionBaseUrl").val().trim();
        if (!customBaseUrl) {
            ui.showGeneralMessage("自定义AI视觉服务需要填写Base URL！", "error");
            return; // 阻止后续 API 调用
        }
        // 注意：state.customAiVisionBaseUrl 应该已经被 #customAiVisionBaseUrl 的 input 事件更新了
        // 所以 api.js 中的 testAiVisionOcrApi 会自动从 state 中获取
    }
    // ^^^^^^ 结束新增 ^^^^^^

    ui.showGeneralMessage("正在测试AI视觉OCR连接...", "info", false);

    // api.js 中的 testAiVisionOcrApi 将负责从 state 中获取 customAiVisionBaseUrl
    api.testAiVisionOcrApi(provider, apiKey, modelName, prompt)
        .then(response => {
            if(response.success) {
                ui.showGeneralMessage(`测试成功: ${response.message}`, "success");
                // 状态已在各自的 change/input 事件中更新，这里无需重复设置
            } else {
                ui.showGeneralMessage(`测试失败: ${response.message}`, "error");
            }
        })
        .catch(error => {
            ui.showGeneralMessage(`测试出错: ${error.message || "未知错误"}`, "error");
        });
}

/**
 * 处理漫画翻译 JSON 提示词切换按钮点击
 */
function handleToggleTranslateJsonPrompt() {
    state.setTranslatePromptMode(!state.isTranslateJsonMode); // 切换模式并加载对应默认提示词
    ui.updateTranslatePromptUI(); // 更新UI显示
}

/**
 * 处理 AI 视觉 OCR JSON 提示词切换按钮点击
 */
function handleToggleAiVisionJsonPrompt() {
    state.setAiVisionOcrPromptMode(!state.isAiVisionOcrJsonMode); // 切换模式并加载对应默认提示词
    ui.updateAiVisionOcrPromptUI(); // 更新UI显示
}

/**
 * 处理漫画翻译提示词模式选择器变更
 */
function handleTranslatePromptModeChange() {
    const mode = $(this).val();
    const useJson = mode === 'json';
    state.setTranslatePromptMode(useJson); // 设置模式并加载对应默认提示词
    ui.updateTranslatePromptUI(); // 更新UI显示
}

/**
 * 处理AI视觉OCR提示词模式选择器变更
 */
function handleAiVisionPromptModeChange() {
    const mode = $(this).val();
    const useJson = mode === 'json';
    state.setAiVisionOcrPromptMode(useJson); // 设置模式并加载对应默认提示词
    ui.updateAiVisionOcrPromptUI(); // 更新UI显示
}

/**
 * 处理导出文本按钮点击
 */
function handleExportText() {
    main.exportText();
}

/**
 * 处理导入文本按钮点击 - 触发文件选择
 */
function handleImportTextClick() {
    $("#importTextFileInput").click();
}

/**
 * 处理导入文本文件选择变更
 */
function handleImportTextFile(e) {
    if (this.files && this.files.length > 0) {
        main.importText(this.files[0]);
        // 重置文件输入框，以便同一文件可以再次选择
        $(this).val('');
    }
}

/**
 * 处理高质量翻译服务商变更
 */
function handleHqTranslateProviderChange() {
    const provider = $(this).val();
    state.setHqTranslateProvider(provider);
    
    // 根据服务商切换UI显示
    if (provider === 'custom_openai') {
        $('#hqCustomBaseUrlDiv').show();
    } else {
        $('#hqCustomBaseUrlDiv').hide();
    }
}

/**
 * 处理高质量翻译API Key变更
 */
function handleHqApiKeyChange() {
    const apiKey = $(this).val();
    state.setHqApiKey(apiKey);
}

/**
 * 处理高质量翻译模型名称变更
 */
function handleHqModelNameChange() {
    const modelName = $(this).val();
    state.setHqModelName(modelName);
}

/**
 * 处理高质量翻译自定义Base URL变更
 */
function handleHqCustomBaseUrlChange() {
    const url = $(this).val();
    state.setHqCustomBaseUrl(url);
}

/**
 * 处理高质量翻译每批次图片数变更
 */
function handleHqBatchSizeChange() {
    const size = $(this).val();
    state.setHqBatchSize(size);
}

/**
 * 处理高质量翻译会话重置频率变更
 */
function handleHqSessionResetChange() {
    const reset = $(this).val();
    state.setHqSessionReset(reset);
}

/**
 * 处理高质量翻译RPM限制变更
 */
function handleHqRpmLimitChange() {
    const limit = $(this).val();
    state.setHqRpmLimit(limit);
}

/**
 * 处理高质量翻译低推理模式变更
 */
function handleHqLowReasoningChange() {
    const low = $('#hqLowReasoning').prop('checked');
    state.setHqLowReasoning(low);
}

/**
 * 处理高质量翻译强制JSON输出选项变更
 */
function handleHqForceJsonOutputChange() {
    const force = $('#hqForceJsonOutput').prop('checked');
    state.setHqForceJsonOutput(force);
}

function handleHqPromptChange() {
    const prompt = $('#hqPrompt').val();
    state.setHqPrompt(prompt);
}

/**
 * 处理源语言变更
 */
function handleSourceLanguageChange() {
    const lang = $(this).val();
    state.setSourceLanguage(lang);
}

/**
 * 处理目标语言变更
 */
function handleTargetLanguageChange() {
    const lang = $(this).val();
    state.setTargetLanguage(lang);
}

// 新增描边事件处理函数
function handleEnableTextStrokeChange(event) {
    const isEnabled = $(this).is(':checked');
    state.setEnableTextStroke(isEnabled); // 更新状态
    $("#textStrokeOptions").toggle(isEnabled); // 根据复选框状态显示/隐藏描边参数选项

    // 触发全局设置变更，这将调用 handleGlobalSettingChange
    // handleGlobalSettingChange 会判断是否需要重渲染
    console.log("描边启用状态改变，触发全局设置变更处理...");
    handleGlobalSettingChange({ target: this }); // 模拟事件对象传递给全局处理器
    session.triggerAutoSave(); // 状态改变，触发自动存档
}

function handleTextStrokeSettingChange(event) {
    // 只有在 "启用文本描边" 被勾选时，描边颜色和宽度的改变才应该生效并触发重渲染
    if ($("#enableTextStroke").is(':checked')) {
        const color = $("#textStrokeColor").val();
        const width = parseInt($("#textStrokeWidth").val());

        // 更新状态
        state.setTextStrokeColor(color);
        state.setTextStrokeWidth(width); // state.setTextStrokeWidth 内部会处理 NaN

        // 触发全局设置变更
        console.log("描边颜色或宽度改变，触发全局设置变更处理...");
        handleGlobalSettingChange({ target: this });
        session.triggerAutoSave(); // 状态改变，触发自动存档
    } else {
        // 如果未启用描边，仅更新状态，但不触发重渲染
        const color = $("#textStrokeColor").val();
        const width = parseInt($("#textStrokeWidth").val());
        state.setTextStrokeColor(color);
        state.setTextStrokeWidth(width);
        console.log("描边未启用，仅更新描边参数状态，不触发重渲染。");
    }
}

// === 新增：气泡描边控件事件处理 START ===
/**
 * 处理单个气泡描边启用状态变更
 */
function handleBubbleEnableStrokeChange(event) {
    const enabled = $(event.target).is(':checked');
    const index = state.selectedBubbleIndex;
    if (index < 0) return;
    
    $(".bubble-stroke-options").toggle(enabled);
    state.updateSingleBubbleSetting(index, { enableStroke: enabled });
    console.log(`气泡 #${index + 1} 描边状态设置为 ${enabled}`);
    
    // 立即更新预览
    import('./edit_mode.js').then(editMode => {
        editMode.renderBubblePreview(index);
    });
}

/**
 * 处理单个气泡描边颜色和宽度变更
 */
function handleBubbleStrokeSettingChange(event) {
    const index = state.selectedBubbleIndex;
    if (index < 0) return;
    
    const settingId = event.target.id;
    const enabled = $("#bubbleEnableStroke").is(':checked');
    
    // 只有在启用描边时才应用更改
    if (!enabled) return;
    
    if (settingId === 'bubbleStrokeColor') {
        const color = $(event.target).val();
        state.updateSingleBubbleSetting(index, { strokeColor: color });
        console.log(`气泡 #${index + 1} 描边颜色设置为 ${color}`);
    } else if (settingId === 'bubbleStrokeWidth') {
        const width = parseInt($(event.target).val());
        if (isNaN(width) || width < 0) {
            $(event.target).val(0);
            state.updateSingleBubbleSetting(index, { strokeWidth: 0 });
            console.log(`气泡 #${index + 1} 描边宽度设置为 0`);
        } else {
            state.updateSingleBubbleSetting(index, { strokeWidth: width });
            console.log(`气泡 #${index + 1} 描边宽度设置为 ${width}`);
        }
    }
    
    // 立即更新预览
    import('./edit_mode.js').then(editMode => {
        editMode.renderBubblePreview(index);
    });
}
// === 新增：气泡描边控件事件处理 END ===

/**
 * 处理AI校对按钮点击
 */
function handleProofread() {
    if (state.images.length === 0) {
        ui.showGeneralMessage("请先添加图片", "warning");
        return;
    }
    
    // 导入AI校对模块并启动校对流程
    import('./ai_proofreading.js').then(proofreading => {
        proofreading.startProofreading();
    }).catch(error => {
        console.error("启动AI校对失败:", error);
        ui.showGeneralMessage("启动AI校对失败: " + error.message, "error");
    });
}

/**
 * 处理校对设置按钮点击
 */
function handleProofreadSettings() {
    // 导入AI校对模块并初始化设置UI，无需检查批量操作状态
    import('./ai_proofreading.js').then(proofreading => {
        proofreading.initProofreadingUI();
        ui.showProofreadingSettingsModal();
    }).catch(error => {
        console.error("打开校对设置失败:", error);
        ui.showGeneralMessage("打开校对设置失败: " + error.message, "error");
    });
}


