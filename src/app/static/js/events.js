import * as state from './state.js';
import * as ui from './ui.js';
import * as api from './api.js';
import * as editMode from './edit_mode.js'; // 引入编辑模式逻辑
import * as labelingMode from './labeling_mode.js'; // <--- 新增导入
import * as main from './main.js';
import * as session from './session.js'; // <--- 新增导入

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
    $("#removeTextOnlyButton").on('click', handleRemoveTextOnly); // 仅消除文字
    $("#translateAllButton").on('click', handleTranslateAll);
    $("#clearAllImagesButton").on('click', handleClearAll);
    $("#deleteCurrentImageButton").on('click', handleDeleteCurrent);
    $("#applyFontSettingsToAllButton").on('click', handleApplySettingsToAll); // 应用设置到全部

    // --- 导航与显示 ---
    $("#prevImageButton").on('click', handlePrevImage);
    $("#nextImageButton").on('click', handleNextImage);
    $("#toggleImageButton").on('click', handleToggleImageDisplay);
    $("#imageSizeSlider").on('input', handleImageSizeChange);
    $("#thumbnail-sidebar #thumbnailList").on('click', '.thumbnail-item', handleThumbnailClick); // 事件委托

    // --- 下载 ---
    $("#downloadButton").on('click', handleDownloadCurrent);
    $("#downloadAllImagesButton").on('click', handleDownloadAll);

    // --- 设置项变更 ---
    $("#fontSize").on('change', handleGlobalSettingChange);
    $("#autoFontSize").on('change', handleGlobalSettingChange); // 自动字号也触发
    $("#fontFamily").on('change', handleGlobalSettingChange);
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
    
    // AI视觉OCR相关事件
    $("#aiVisionProvider, #aiVisionModelName, #aiVisionOcrPrompt").on('change', handleGlobalSettingChange);
    $("#testAiVisionOcrButton").on('click', handleTestAiVisionOcr); // AI视觉OCR测试按钮

    // --- 模型与提示词 ---
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
    $('#toggleTranslateJsonPromptButton').on('click', handleToggleTranslateJsonPrompt);
    $('#toggleAiVisionJsonPromptButton').on('click', handleToggleAiVisionJsonPrompt);
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
    $("#bubbleFillColor").on('input', handleBubbleSettingChange);
    $("#bubbleRotationAngle").on('input', handleBubbleRotationChange);
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

    // --- 新增：RPD 设置变更事件 ---
    $("#rpdTranslation").on('change input', function() { // 'input' 事件可实现更实时的更新（可选）
        const value = $(this).val();
        state.setRpdLimitTranslation(value);
        // 可选：如果需要，可以在这里触发一个保存用户偏好的操作，例如到 localStorage
        // localStorage.setItem('rpdLimitTranslation', state.rpdLimitTranslation);
    });

    $("#rpdAiVisionOcr").on('change input', function() {
        const value = $(this).val();
        state.setRpdLimitAiVisionOcr(value);
        // localStorage.setItem('rpdLimitAiVisionOcr', state.rpdLimitAiVisionOcr);
    });
    // ---------------------------

    console.log("事件监听器绑定完成。");
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
    // downloadCurrentImage 函数需要在 main.js 中定义并导出
    import('./main.js').then(main => main.downloadCurrentImage());
}

function handleDownloadAll() {
    // downloadAllImages 函数需要在 main.js 中定义并导出
    import('./main.js').then(main => main.downloadAllImages());
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
    } else if (changedElement.type === 'number' || settingId === 'fontSize') {
        newValue = parseInt(changedElement.value) || state.defaultFontSize; // 处理字号
    } else if (changedElement.type === 'range') {
        newValue = parseFloat(changedElement.value); // 处理滑块
    } else {
        newValue = changedElement.value; // 其他 (select, color, text)
    }

    let settingsUpdated = false; // 用于决定是否触发通用重渲染

    // 更新全局默认值状态（用于新图片）
    switch (settingId) {
        case 'fontSize': state.setDefaultFontSize(newValue); break;
        case 'fontFamily': state.setDefaultFontFamily(newValue); break;
        case 'layoutDirection': state.setDefaultLayoutDirection(newValue); break;
        case 'textColor': state.setDefaultTextColor(newValue); break;
        case 'fillColor': 
            state.setDefaultFillColor(newValue);
            if (currentImage && !state.isLabelingModeActive) {
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
        editMode.reRenderFullImage();
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
    if (confirm('确定要清理所有调试文件吗？这将释放磁盘空间，但不会影响您的翻译图片。')) {
        ui.showLoading("正在清理调试文件...");
        api.cleanDebugFilesApi()
            .then(response => {
                ui.hideLoading();
                ui.showGeneralMessage(response.message, response.success ? "success" : "error");
            })
            .catch(error => {
                ui.hideLoading();
                ui.showGeneralMessage(`清理调试文件失败: ${error.message}`, "error");
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
            fillColor: $('#bubbleFillColor').val() // <--- 获取独立填充色
        };

        state.updateSingleBubbleSetting(index, settingUpdate);
        // 触发预览 (renderBubblePreview 内部调用 reRenderFullImage)
        // reRenderFullImage 内部现在会处理独立填充色
        editMode.renderBubblePreview(index);
    }
}

// 旋转角度变化 (带延迟渲染)
let rotationTimer = null;
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

// 位置偏移变化 (带延迟渲染)
let positionTimer = null;
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

// 处理测试AI视觉OCR按钮点击
function handleTestAiVisionOcr() {
    const provider = $("#aiVisionProvider").val();
    const apiKey = $("#aiVisionApiKey").val();
    const modelName = $("#aiVisionModelName").val();
    const prompt = $("#aiVisionOcrPrompt").val();
    
    // 参数验证
    if(!apiKey) {
        ui.showGeneralMessage("请输入API Key", "error");
        return;
    }
    
    if(!modelName) {
        ui.showGeneralMessage("请输入模型名称", "error");
        return;
    }
    
    // 显示加载提示
    ui.showGeneralMessage("正在测试AI视觉OCR连接...", "info", false);
    
    // 调用测试API
    api.testAiVisionOcrApi(provider, apiKey, modelName, prompt)
        .then(response => {
            if(response.success) {
                ui.showGeneralMessage(`测试成功: ${response.message}`, "success");
                
                // 保存设置到state中
                state.setAiVisionProvider(provider);
                state.setAiVisionModelName(modelName);
                state.setAiVisionOcrPrompt(prompt || "");
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


