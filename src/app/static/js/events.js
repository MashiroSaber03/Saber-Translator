import * as state from './state.js';
import * as ui from './ui.js';
import * as api from './api.js';
import * as editMode from './edit_mode.js'; // 引入编辑模式逻辑
import * as main from './main.js';
import * as session from './session.js';
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
    $("#pauseTranslationButton").on('click', handlePauseTranslation); // 暂停/继续按钮
    $("#proofreadButton").on('click', handleProofread);
    $("#proofreadSettingsButton").on('click', handleProofreadSettings);
    $("#removeTextOnlyButton").on('click', handleRemoveTextOnly); // 仅消除文字
    $("#removeAllTextButton").on('click', handleRemoveAllText); // 消除所有图片文字
    $("#deleteCurrentImageButton").on('click', handleDeleteCurrent);
    $("#clearAllImagesButton").on('click', handleClearAll);
    $("#applyFontSettingsToAllButton").on('click', handleApplySettingsToAll); // 应用设置到全部
    $("#applySettingsOptionsBtn").on('click', handleToggleApplySettingsDropdown); // 齿轮按钮切换下拉菜单
    $("#apply_selectAll").on('change', handleApplySelectAllToggle); // 全选切换
    // 单个复选框变化时更新全选状态
    $("#applySettingsDropdown input[type='checkbox']:not(#apply_selectAll)").on('change', updateSelectAllState);
    // 点击外部关闭下拉菜单
    $(document).on('click', function(e) {
        if (!$(e.target).closest('.apply-settings-group').length) {
            $("#applySettingsDropdown").removeClass('show');
        }
    });

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
    $("#autoFontSize").on('change', handleAutoFontSizeChange); // 自动字号切换
    // 初始化时设置字号输入框状态
    toggleFontSizeInput();
    $("#layoutDirection").on('change', handleGlobalSettingChange);
    $("#textColor").on('input', handleGlobalSettingChange); // 颜色实时变化
    $("#useInpainting").on('change', handleInpaintingMethodChange);
    $("#fillColor").on('input', handleGlobalSettingChange); // 填充颜色实时变化
    // 文本检测器、OCR引擎、AI视觉OCR、翻译服务等设置事件已移至设置模态框(settings_modal.js)处理
    // Ollama/Sakura测试按钮和模型列表事件也已移至设置模态框

    $("#enableTextboxPrompt").on('change', handleEnableTextboxPromptChange);

    // --- 系统与其他 ---
    $("#cleanDebugFilesButton").on('click', handleCleanDebugFiles);
    $(document).on('keydown', handleGlobalKeyDown); // 全局快捷键
    $("#themeToggle").on('click', handleThemeToggle); // 主题切换
    $(document).on('click', '#donateButton', handleDonateClick); // 赞助按钮
    $(document).on('click', '.donate-close', handleDonateClose);
    $(window).on('click', handleWindowClickForModal); // 点击模态框外部关闭

    // === 会话管理按钮事件（仅书架模式下显示保存按钮）===
    $("#saveCurrentSessionButton").on('click', handleSaveCurrentSession);
    // === 结束会话管理 ===

    // --- 编辑模式 ---
    $("#toggleEditModeButton").on('click', editMode.toggleEditMode);
    // 旧版编辑模式事件绑定已删除，新版事件在 edit_mode.js 的 bindNewEditModeEvents() 中绑定

    // 模型推荐列表相关的全局点击事件已移除 - 现在由设置模态框处理

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

    // --- 新增：重试次数设置变更事件 ---
    $("#translationMaxRetries, #settingsTranslationMaxRetries").on('change input', function() {
        const value = $(this).val();
        state.setTranslationMaxRetries(value);
    });

    $("#hqMaxRetries, #settingsHqMaxRetries").on('change input', function() {
        const value = $(this).val();
        state.setHqTranslationMaxRetries(value);
    });

    $("#proofreadingMaxRetries, #settingsProofreadingMaxRetries").on('change input', function() {
        const value = $(this).val();
        state.setProofreadingMaxRetries(value);
    });

    // AI校对启用状态变更事件
    $("#proofreadingEnabled, #settingsProofreadingEnabled").on('change', function() {
        const isEnabled = $(this).is(':checked');
        state.setProofreadingEnabled(isEnabled);
        console.log(`AI校对状态更新: ${isEnabled ? '启用' : '禁用'}`);
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

    // 初始化高质量翻译模块UI
    hqTranslation.initHqTranslationUI();
    
    // 侧边栏的高质量翻译按钮
    $("#startHqTranslationBtn").on('click', function() {
        hqTranslation.startHqTranslation();
    });

    // 高质量翻译设置事件已移至设置模态框(settings_modal.js)处理

    $("#strokeEnabled").on('change', handleStrokeEnabledChange);
    $("#strokeColor").on('input', handleStrokeSettingChange);
    $("#strokeWidth").on('input', handleStrokeSettingChange);

    // 侧边栏OCR设置事件已移至设置模态框(settings_modal.js)处理
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
        main.translateCurrentImage();
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
        main.removeBubbleTextOnly();
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

/**
 * 处理暂停/继续翻译按钮点击
 */
function handlePauseTranslation() {
    if (!state.isBatchTranslationInProgress) {
        return; // 如果没有在批量翻译，忽略
    }
    
    if (state.isBatchTranslationPaused) {
        // 当前是暂停状态，点击继续
        state.resumeBatchTranslation();
    } else {
        // 当前是翻译中，点击暂停
        state.setBatchTranslationPaused(true);
        ui.updatePauseButton(true);
        // 清除之前的提示消息
        $(".message.info").fadeOut(300, function() { $(this).remove(); });
    }
}

function handleClearAll() {
    if (confirm('确定要清除所有图片吗？这将丢失所有未保存的进度。')) {
        state.clearImages();
        ui.renderThumbnails();
        ui.showResultSection(false);
        ui.updateButtonStates();
        ui.updateProgressBar(0, ''); // 重置进度条
        editMode.exitEditMode(); // 如果在编辑模式则退出
        
        // 注意：不再清除书籍/章节上下文，也不隐藏保存按钮
        // 书架模式下用户可能只是想清空重新开始，但仍保持在当前章节
        // 保存按钮会提示"没有图片数据，无法保存"
    }
}

function handleDeleteCurrent() {
    if (state.currentImageIndex !== -1) {
        if (confirm(`确定要删除当前图片 (${state.images[state.currentImageIndex].fileName}) 吗？`)) {
            const deletedIndex = state.currentImageIndex;
            
            // 如果在编辑模式，先安全退出（不保存被删除图片的状态到其他图片）
            if (state.editModeActive) {
                // 清空当前 bubbleStates 避免被错误保存到其他图片
                state.setBubbleStates([]);
                state.setEditModeActive(false);
                state.setSelectedBubbleIndex(-1);
                ui.toggleEditModeUI(false);
            }
            
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
            }
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

function handleToggleApplySettingsDropdown(e) {
    e.stopPropagation(); // 阻止冒泡，防止触发document点击关闭
    $("#applySettingsDropdown").toggleClass('show');
}

function handleApplySelectAllToggle() {
    const isChecked = $("#apply_selectAll").is(':checked');
    // 设置所有参数复选框的状态
    $("#applySettingsDropdown input[type='checkbox']:not(#apply_selectAll)").prop('checked', isChecked);
}

function updateSelectAllState() {
    // 检查是否所有参数复选框都被选中
    const allCheckboxes = $("#applySettingsDropdown input[type='checkbox']:not(#apply_selectAll)");
    const allChecked = allCheckboxes.length === allCheckboxes.filter(':checked').length;
    $("#apply_selectAll").prop('checked', allChecked);
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

/**
 * 处理全局设置变化 (简化版)
 * 核心逻辑：修改全局参数 → 更新所有气泡的对应参数 → 重渲染
 */
function handleGlobalSettingChange(e) {
    e = e || { target: null };
    const changedElement = e.target;
    if (!changedElement) return;
    
    let settingId = changedElement.id || $(changedElement).attr('id');
    if (!settingId) return;
    
    const currentImage = state.getCurrentImage();
    
    // 获取新值
    let newValue;
    if (changedElement.type === 'checkbox') {
        newValue = changedElement.checked;
    } else if (settingId === 'fontSize' || settingId === 'strokeWidth') {
        newValue = parseInt(changedElement.value) || 0;
    } else {
        newValue = changedElement.value;
    }

    // 1. 更新全局默认值（用于新图片）
    const globalSetters = {
        'fontSize': () => state.setDefaultFontSize(newValue),
        'fontFamily': () => state.setDefaultFontFamily(newValue),
        'layoutDirection': () => state.setDefaultLayoutDirection(newValue),
        'textColor': () => state.setDefaultTextColor(newValue),
        'fillColor': () => state.setDefaultFillColor(newValue),
        'aiVisionOcrPrompt': () => state.setAiVisionOcrPrompt(newValue),
    };
    if (globalSetters[settingId]) globalSetters[settingId]();

    // 需要触发重渲染的设置
    const renderSettings = ['fontSize', 'autoFontSize', 'fontFamily', 'layoutDirection', 'textColor', 
                           'strokeEnabled', 'strokeColor', 'strokeWidth', 'fillColor'];
    const needsRender = renderSettings.includes(settingId);
    
    // 如果不需要重渲染，直接返回
    if (!needsRender) return;
    
    // 如果是切换图片时触发的变更，跳过重渲染
    if (window._isChangingFromSwitchImage) {
        console.log("检测到来自切换图片的设置变更，跳过重渲染");
        return;
    }

    // 2. 如果没有已翻译的图片，不需要重渲染
    if (!currentImage || !currentImage.translatedDataURL) return;
    
    // 3. 获取或初始化 bubbleStates
    let bubbleStates = state.editModeActive ? state.bubbleStates : currentImage.bubbleStates;
    
    // 如果没有 bubbleStates，不需要继续处理
    if (!bubbleStates || !Array.isArray(bubbleStates) || bubbleStates.length === 0) {
        console.log("当前图片没有 bubbleStates，跳过更新");
        return;
    }

    // 4. 更新所有气泡的对应参数（只更新被修改的那个参数）
    // 注意：autoFontSize 只用于首次翻译，翻译后不再影响已有气泡
    const propertyMap = {
        'fontSize': 'fontSize',
        'fontFamily': 'fontFamily',
        'layoutDirection': 'textDirection',  // UI 是 layoutDirection，状态是 textDirection
        'textColor': 'textColor',
        'strokeEnabled': 'strokeEnabled',
        'strokeColor': 'strokeColor',
        'strokeWidth': 'strokeWidth',
        'fillColor': 'fillColor'
    };
    
    const stateProperty = propertyMap[settingId];
    if (stateProperty) {
        // 【重要】特殊处理 layoutDirection
        // 当用户选择 'auto' 时，不修改 textDirection（保持原值）
        // reRenderFullImage 会根据 autoTextDirection 来决定实际渲染方向
        if (settingId === 'layoutDirection' && newValue === 'auto') {
            console.log("排版方向设置为 'auto'，不修改气泡的 textDirection，使用 autoTextDirection 渲染");
            // 不修改 textDirection，让 reRenderFullImage 使用 autoTextDirection
        } else {
            bubbleStates.forEach(bs => {
                bs[stateProperty] = newValue;
            });
        }
    }

    // 5. 保存更新后的状态
    if (state.editModeActive) {
        state.setBubbleStates([...bubbleStates]);
    }
    currentImage.bubbleStates = JSON.parse(JSON.stringify(bubbleStates));

    // 6. 触发重渲染
    console.log(`全局设置变更 (${settingId}=${newValue})，重渲染...`);
    editMode.reRenderFullImage();
}

// handleTextDetectorChange 已移至设置模态框(settings_modal.js)

function handleInpaintingMethodChange() {
    const repairMethod = $(this).val();
    // 判断是否使用 LAMA（lama_mpe 或 litelama）
    const isLama = repairMethod === 'lama_mpe' || repairMethod === 'litelama';
    ui.toggleInpaintingOptions(isLama, repairMethod === 'false');
    // 这个变化不直接触发重渲染，下次翻译或手动重渲染时生效
}

// handleModelProviderChange, handleModelNameFocus, handleModelNameBlur, handleModelButtonClick
// 已移至设置模态框(settings_modal.js)处理

// handleSavePrompt 和 handleSaveTextboxPrompt 已移至设置模态框的提示词管理功能

/**
 * 切换字号输入框的禁用状态
 * 当勾选"自动计算初始字号"时，禁用字号输入框
 */
function toggleFontSizeInput() {
    const isAutoFontSize = $('#autoFontSize').is(':checked');
    const fontSizeInput = $('#fontSize');
    
    fontSizeInput.prop('disabled', isAutoFontSize);
    
    // 视觉反馈：禁用时降低透明度
    if (isAutoFontSize) {
        fontSizeInput.css('opacity', '0.5');
        fontSizeInput.attr('title', '已启用自动字号，首次翻译时将自动计算');
    } else {
        fontSizeInput.css('opacity', '1');
        fontSizeInput.attr('title', '');
    }
}

/**
 * 处理自动字号复选框变更
 * 开启/关闭自动字号都会触发重渲染
 */
function handleAutoFontSizeChange(e) {
    const isAutoFontSize = $('#autoFontSize').is(':checked');
    
    // 更新字号输入框状态
    toggleFontSizeInput();
    
    // 如果是切换图片时触发的变更，跳过重渲染
    if (window._isChangingFromSwitchImage) {
        console.log("检测到来自切换图片的自动字号变更，跳过重渲染");
        return;
    }
    
    // 如果有已翻译的图片，触发重渲染
    const currentImage = state.getCurrentImage();
    if (currentImage && currentImage.translatedDataURL) {
        const bubbleStates = state.editModeActive ? state.bubbleStates : currentImage.bubbleStates;
        if (bubbleStates && Array.isArray(bubbleStates) && bubbleStates.length > 0) {
            if (isAutoFontSize) {
                // 开启自动字号：重新计算每个气泡的字号
                console.log('自动字号已开启，重新计算字号并渲染...');
                editMode.reRenderFullImage(false, false, true);
            } else {
                // 关闭自动字号：将所有气泡设为输入框中的固定字号
                const fixedFontSize = parseInt($('#fontSize').val()) || state.defaultFontSize;
                console.log(`自动字号已关闭，使用固定字号 ${fixedFontSize} 渲染...`);
                bubbleStates.forEach(bs => {
                    bs.fontSize = fixedFontSize;
                });
                // 同步状态
                if (state.editModeActive) {
                    state.setBubbleStates([...bubbleStates]);
                    // 刷新编辑面板显示
                    if (state.selectedBubbleIndex >= 0) {
                        editMode.selectBubbleNew(state.selectedBubbleIndex);
                    }
                }
                currentImage.bubbleStates = JSON.parse(JSON.stringify(bubbleStates));
                editMode.reRenderFullImage(false, false, false);
            }
            return;
        }
    }
    
    console.log(`自动字号设置变更: ${isAutoFontSize} (仅影响下次翻译)`);
}

function handleEnableTextboxPromptChange(e) {
    const use = e.target.checked;
    state.setUseTextboxPrompt(use);
    // 同步到设置模态框中的文本框提示词区域
    const settingsTextboxArea = document.getElementById('settingsTextboxPromptArea');
    if (settingsTextboxArea) {
        settingsTextboxArea.style.display = use ? 'block' : 'none';
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
    
    // 编辑模式下的快捷键已在 edit_mode.js 中统一处理
    // 注意：已移除 ←/→ 切换气泡的快捷键

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

// --- 编辑模式事件处理 ---
// 旧版编辑模式事件处理函数已删除
// 新版事件处理在 edit_mode.js 的 bindNewEditModeEvents() 中实现

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

// === 保存会话按钮的处理函数（仅书架模式下可用）===
function handleSaveCurrentSession() {
    session.triggerSaveCurrentSession();
}

// 侧边栏OCR/AI视觉相关事件已移至设置模态框(settings_modal.js)处理

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

// 高质量翻译和源/目标语言处理函数已移至设置模态框(settings_modal.js)

function handleStrokeEnabledChange(event) {
    const isEnabled = $(this).is(':checked');
    state.setStrokeEnabled(isEnabled);
    
    const currentImage = state.getCurrentImage();
    if (currentImage) {
        state.updateCurrentImageProperty('strokeEnabled', isEnabled);
    }
    
    $("#strokeOptions").toggle(isEnabled);

    console.log("描边启用状态改变，触发全局设置变更处理...");
    handleGlobalSettingChange({ target: this });
}

function handleStrokeSettingChange(event) {
    if ($("#strokeEnabled").is(':checked')) {
        const color = $("#strokeColor").val();
        const width = parseInt($("#strokeWidth").val());

        state.setStrokeColor(color);
        state.setStrokeWidth(width);
        
        const currentImage = state.getCurrentImage();
        if (currentImage) {
            state.updateCurrentImageProperty('strokeColor', color);
            state.updateCurrentImageProperty('strokeWidth', width);
        }

        console.log("描边颜色或宽度改变，触发全局设置变更处理...");
        handleGlobalSettingChange({ target: this });
    } else {
        const color = $("#strokeColor").val();
        const width = parseInt($("#strokeWidth").val());
        state.setStrokeColor(color);
        state.setStrokeWidth(width);
        
        const currentImage = state.getCurrentImage();
        if (currentImage) {
            state.updateCurrentImageProperty('strokeColor', color);
            state.updateCurrentImageProperty('strokeWidth', width);
        }
        
        console.log("描边未启用，仅更新描边参数状态，不触发重渲染。");
    }
}

// 旧版气泡描边相关函数已删除，使用新版编辑模式

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


