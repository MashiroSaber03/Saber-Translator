import * as state from './state.js';
import * as ui from './ui.js';
import * as api from './api.js';
import * as editMode from './edit_mode.js'; // 引入编辑模式逻辑
// import $ from 'jquery'; // 假设 jQuery 已全局加载

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

    // --- 模型与提示词 ---
    $("#modelProvider").on('change', handleModelProviderChange);
    $("#modelName").on('focus', handleModelNameFocus);
    $("#modelName").on('blur', handleModelNameBlur);
    // Ollama/Sakura 测试按钮 (如果存在)
    // 注意：按钮是动态添加的，需要使用事件委托或在按钮创建后绑定
    $(document).on('click', '#testOllamaButton', api.testOllamaConnectionApi); // 直接调用 API 测试
    $(document).on('click', '#testSakuraButton', api.testSakuraConnectionApi); // 直接调用 API 测试
    // 模型按钮点击 (事件委托)
    $(document).on('click', '#ollamaModelsList .model-button, #sakuraModelsList .model-button', handleModelButtonClick);


    $("#savePromptButton").on('click', handleSavePrompt);
    $("#promptDropdownButton").on('click', (e) => { e.stopPropagation(); $("#promptDropdown").toggle(); });
    // 下拉列表项和删除按钮的点击事件在 populate 函数中绑定

    $("#enableTextboxPrompt").on('change', handleEnableTextboxPromptChange);
    $("#saveTextboxPromptButton").on('click', handleSaveTextboxPrompt);
    $("#textboxPromptDropdownButton").on('click', (e) => { e.stopPropagation(); $("#textboxPromptDropdown").toggle(); });
    // 下拉列表项和删除按钮的点击事件在 populate 函数中绑定

    // --- 系统与其他 ---
    $("#cleanDebugFilesButton").on('click', handleCleanDebugFiles);
    $(document).on('keydown', handleGlobalKeyDown); // 全局快捷键
    $("#themeToggle").on('click', handleThemeToggle); // 主题切换
    $(document).on('click', '#donateButton', handleDonateClick); // 赞助按钮
    $(document).on('click', '.donate-close', handleDonateClose);
    $(window).on('click', handleWindowClickForModal); // 点击模态框外部关闭

    // --- 编辑模式 ---
    $("#toggleEditModeButton").on('click', editMode.toggleEditMode);
    // 气泡列表项点击 (事件委托)
    $("#bubbleList").on('click', '.bubble-item', handleBubbleItemClick);
    // 编辑区域输入/变更事件
    $("#bubbleTextEditor").on('input', handleBubbleEditorChange); // 文本实时变化
    $("#bubbleFontSize").on('change', handleBubbleSettingChange);
    $("#autoBubbleFontSize").on('change', handleBubbleSettingChange); // 自动字号
    $("#bubbleFontFamily").on('change', handleBubbleSettingChange);
    $("#bubbleTextDirection").on('change', handleBubbleSettingChange);
    $("#bubbleTextColor").on('input', handleBubbleSettingChange); // 颜色实时变化
    $("#bubbleRotationAngle").on('input', handleBubbleRotationChange); // 旋转实时变化 (带延迟)
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
    $("#pluginListContainer").on('change', '.plugin-enable-toggle', handlePluginToggleChange);
    $("#pluginListContainer").on('click', '.plugin-delete-button', handlePluginDeleteClick);
    // 添加插件设置按钮 (事件委托)
    $("#pluginListContainer").on('click', '.plugin-settings-button', handlePluginSettingsClick);
    
    // 插件配置模态框关闭按钮 (事件委托，因为模态框是动态创建的)
    $(document).on('click', '#pluginConfigModal .plugin-modal-close', handlePluginConfigModalClose);
    
    // 插件配置保存按钮 (事件委托)
    $(document).on('submit', '#pluginConfigForm', handlePluginConfigSave);

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
    import('./main.js').then(main => main.translateCurrentImage());
}

function handleRemoveTextOnly() {
    if (state.currentImageIndex === -1) {
        ui.showGeneralMessage("请先选择图片", "warning");
        return;
    }
    // 使用showGeneralMessage替代showLoading
    ui.showGeneralMessage("消除文字中...", "info", false, 0);
    // removeBubbleTextOnly 函数需要在 main.js 或 api.js 中定义并导出
    import('./main.js').then(main => main.removeBubbleTextOnly());
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

function handleGlobalSettingChange() {
    // 当全局字体/大小/方向/颜色等改变时，如果当前有已翻译图片，触发重渲染
    const currentImage = state.getCurrentImage();
    if (currentImage && currentImage.translatedDataURL) {
        console.log("全局设置变更，触发重新渲染...");
        
        // 检测是否是从自动字号切换到非自动字号
        const isAutoFontSizeChange = $(this).attr('id') === 'autoFontSize';
        const isPrevAutoFontSize = currentImage.autoFontSize;
        const isNowAutoFontSize = $('#autoFontSize').is(':checked');
        
        // 如果是自动字号切换到非自动字号，特殊处理
        if (isAutoFontSizeChange && isPrevAutoFontSize && !isNowAutoFontSize) {
            console.log("检测到从自动字号切换到非自动字号");
            // 启用字号输入，并设置为默认值
            $('#fontSize').prop('disabled', false).val(state.defaultFontSize);
        }
        
        // reRenderTranslatedImage 函数需要在 main.js 或 edit_mode.js 中定义并导出
        import('./edit_mode.js').then(edit => edit.reRenderFullImage(true, isPrevAutoFontSize && !isNowAutoFontSize)); // 传递参数表明是全局设置变更以及是否从自动切换到非自动
    }
    // 更新状态中的默认值 (如果需要)
    // state.setDefaultFontSize(...) 等
}

function handleInpaintingMethodChange() {
    const repairMethod = $(this).val();
    const showInpaintingOptions = repairMethod === 'true' || repairMethod === 'lama';
    const showSolidOptions = repairMethod === 'false';
    ui.toggleInpaintingOptions(showInpaintingOptions, showSolidOptions);
    // 这个变化不直接触发重渲染，下次翻译或手动重渲染时生效
}

function handleInpaintingStrengthChange(e) {
    $('#inpaintingStrengthValue').text(e.target.value);
    // 强度变化不实时重渲染，在下次翻译或手动重渲染时生效
}

function handleModelProviderChange() {
    const selectedProvider = $(this).val();
    ui.updateApiKeyInputState(selectedProvider === 'ollama' || selectedProvider === 'sakura',
                              selectedProvider === 'ollama' || selectedProvider === 'sakura' ? '本地部署无需API Key' : '请输入API Key');
    ui.toggleOllamaUI(selectedProvider === 'ollama');
    ui.toggleSakuraUI(selectedProvider === 'sakura');

    if (selectedProvider === 'ollama') {
        import('./main.js').then(main => main.fetchOllamaModels()); // 获取模型
    } else if (selectedProvider === 'sakura') {
        import('./main.js').then(main => main.fetchSakuraModels()); // 获取模型
    } else {
         $('#ollamaModelsList').empty().hide(); // 隐藏其他服务商的模型列表
         $('#sakuraModelsList').empty().hide();
    }
    // 获取历史模型建议
    api.getUsedModelsApi(selectedProvider)
        .then(response => ui.updateModelSuggestions(response.models))
        .catch(error => console.error("获取模型建议失败:", error));
}

function handleModelNameFocus() {
    const modelProvider = $('#modelProvider').val();
    if (modelProvider && modelProvider !== 'ollama' && modelProvider !== 'sakura') { // 本地模型不显示历史建议
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
    // 编辑模式下禁用部分快捷键，或赋予不同功能
    if (state.editModeActive) {
        // 例如，在编辑模式下左右箭头可以切换气泡
        if (e.keyCode == 37) { // Left Arrow
             e.preventDefault();
             editMode.selectPrevBubble();
        } else if (e.keyCode == 39) { // Right Arrow
             e.preventDefault();
             editMode.selectNextBubble();
        }
        // 可以添加其他编辑模式快捷键，如 Alt+Up/Down 调整字号
        return; // 阻止后续的全局快捷键处理
    }

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
    if (event.target === $("#donateModal")[0]) {
        $("#donateModal").css("display", "none");
    }
}

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
function handleBubbleSettingChange() {
    const index = state.selectedBubbleIndex;
    if (index >= 0) {
        const currentSetting = state.bubbleSettings[index];
        const isAuto = $('#autoBubbleFontSize').is(':checked');
        
        // 检测是否从自动字号切换到非自动字号
        const isPrevAuto = currentSetting.autoFontSize;
        if (this.id === 'autoBubbleFontSize' && isPrevAuto && !isAuto) {
            console.log("单气泡从自动字号切换到手动字号，使用默认字号");
            // 启用字号输入，设置为默认值
            $('#bubbleFontSize').prop('disabled', false).val(state.defaultFontSize);
        }
        
        const settingUpdate = {
            fontSize: isAuto ? 'auto' : parseInt($('#bubbleFontSize').val()),
            autoFontSize: isAuto,
            fontFamily: $('#bubbleFontFamily').val(),
            textDirection: $('#bubbleTextDirection').val(),
            textColor: $('#bubbleTextColor').val()
        };
        state.updateSingleBubbleSetting(index, settingUpdate);
        editMode.renderBubblePreview(index); // 触发预览
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

    api.getPluginsApi()
        .then(response => {
            if (response.success) {
                ui.renderPluginList(response.plugins); // 使用 UI 模块渲染列表
            } else {
                throw new Error(response.error || "无法加载插件列表");
            }
        })
        .catch(error => {
            container.html(`<p class="error">加载插件列表失败: ${error.message}</p>`);
        });
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
