// src/app/static/js/main.js

// 引入所有需要的模块
import * as state from './state.js';
import * as ui from './ui.js';
import * as api from './api.js';
import * as events from './events.js';
import * as editMode from './edit_mode.js';
import * as constants from './constants.js'; // 导入前端常量
// import $ from 'jquery'; // 假设 jQuery 已全局加载

// --- 初始化函数 ---

/**
 * 初始化应用状态和 UI
 */
function initializeApp() {
    console.log("初始化应用程序...");

    // 1. 从 DOM 读取初始/默认设置并更新状态
    state.setDefaultFontSize(parseInt($('#fontSize').val()) || 25);
    state.setDefaultFontFamily($('#fontFamily').val() || 'fonts/STSONG.TTF'); // 确保与 HTML 默认值一致
    state.setDefaultLayoutDirection($('#layoutDirection').val() || 'vertical');
    state.setDefaultTextColor($('#textColor').val() || '#000000');
    state.setDefaultFillColor($('#fillColor').val() || constants.DEFAULT_FILL_COLOR);
    state.setUseTextboxPrompt($('#enableTextboxPrompt').is(':checked'));

    // 2. 初始化提示词设置 (调用 API)
    initializePromptSettings();
    initializeTextboxPromptSettings();

    // 3. 初始化可折叠面板
    initializeCollapsiblePanels();

    // 4. 初始化亮暗模式
    initializeThemeMode();

    // 5. 检查初始模型提供商并更新 UI
    checkInitialModelProvider();

    // 6. 绑定所有事件监听器
    events.bindEventListeners();

    // 7. 更新初始按钮状态
    ui.updateButtonStates();

    // 8. 初始化修复选项的显示状态
    const initialRepairMethod = $('#useInpainting').val();
    ui.toggleInpaintingOptions(
        initialRepairMethod === 'true' || initialRepairMethod === 'lama',
        initialRepairMethod === 'false'
    );

    console.log("应用程序初始化完成。");
}

// --- 辅助函数 (从原始 script.js 迁移) ---

/**
 * 初始化漫画翻译提示词设置
 */
export function initializePromptSettings() { // 导出以便外部调用（如果需要）
    api.getPromptsApi()
        .then(response => {
            state.setPromptState(response.default_prompt_content, response.default_prompt_content, response.prompt_names || []);
            $('#promptContent').val(state.currentPromptContent);
            ui.populatePromptDropdown(state.savedPromptNames, $('#promptDropdown'), $('#promptDropdownButton'), loadPromptContent, deletePrompt);
        })
        .catch(error => {
            console.error("获取提示词信息失败:", error);
            const errorMsg = "获取默认提示词失败";
            state.setPromptState(errorMsg, errorMsg, []);
            $('#promptContent').val(errorMsg);
            ui.populatePromptDropdown([], $('#promptDropdown'), $('#promptDropdownButton'), loadPromptContent, deletePrompt);
        });
}

/**
 * 初始化文本框提示词设置
 */
export function initializeTextboxPromptSettings() { // 导出
    api.getTextboxPromptsApi()
        .then(response => {
            state.setTextboxPromptState(response.default_prompt_content, response.default_prompt_content, response.prompt_names || []);
            $('#textboxPromptContent').val(state.currentTextboxPromptContent);
            ui.populatePromptDropdown(state.savedTextboxPromptNames, $('#textboxPromptDropdown'), $('#textboxPromptDropdownButton'), loadTextboxPromptContent, deleteTextboxPrompt);
        })
        .catch(error => {
            console.error("获取文本框提示词信息失败:", error);
            const errorMsg = "获取默认文本框提示词失败";
            state.setTextboxPromptState(errorMsg, errorMsg, []);
            $('#textboxPromptContent').val(errorMsg);
            ui.populatePromptDropdown([], $('#textboxPromptDropdown'), $('#textboxPromptDropdownButton'), loadTextboxPromptContent, deleteTextboxPrompt);
        });
}

/**
 * 加载指定名称的漫画翻译提示词内容
 * @param {string} promptName - 提示词名称
 */
function loadPromptContent(promptName) { // 私有辅助函数
    if (promptName === constants.DEFAULT_PROMPT_NAME) {
        state.setPromptState(state.defaultPromptContent, state.defaultPromptContent, state.savedPromptNames);
        $('#promptContent').val(state.currentPromptContent);
    } else {
        api.getPromptContentApi(promptName)
            .then(response => {
                state.setPromptState(response.prompt_content, state.defaultPromptContent, state.savedPromptNames);
                $('#promptContent').val(state.currentPromptContent);
            })
            .catch(error => {
                console.error("加载提示词内容失败:", error);
                ui.showGeneralMessage(`加载提示词 "${promptName}" 失败: ${error.message}`, "error");
            });
    }
}

/**
 * 删除指定名称的漫画翻译提示词
 * @param {string} promptName - 提示词名称
 */
function deletePrompt(promptName) { // 私有辅助函数
    ui.showLoading("删除提示词...");
    api.deletePromptApi(promptName)
        .then(response => {
            ui.hideLoading();
            ui.showGeneralMessage(`提示词 "${promptName}" 删除成功！`, "success");
            initializePromptSettings(); // 重新加载列表
        })
        .catch(error => {
            ui.hideLoading();
            ui.showGeneralMessage(`删除提示词失败: ${error.message}`, "error");
        });
}

/**
 * 加载指定名称的文本框提示词内容
 * @param {string} promptName - 提示词名称
 */
function loadTextboxPromptContent(promptName) { // 私有辅助函数
    if (promptName === constants.DEFAULT_PROMPT_NAME) {
        state.setTextboxPromptState(state.defaultTextboxPromptContent, state.defaultTextboxPromptContent, state.savedTextboxPromptNames);
        $('#textboxPromptContent').val(state.currentTextboxPromptContent);
    } else {
        api.getTextboxPromptContentApi(promptName)
            .then(response => {
                state.setTextboxPromptState(response.prompt_content, state.defaultTextboxPromptContent, state.savedTextboxPromptNames);
                $('#textboxPromptContent').val(state.currentTextboxPromptContent);
            })
            .catch(error => {
                console.error("加载文本框提示词内容失败:", error);
                ui.showGeneralMessage(`加载文本框提示词 "${promptName}" 失败: ${error.message}`, "error");
            });
    }
}

/**
 * 删除指定名称的文本框提示词
 * @param {string} promptName - 提示词名称
 */
function deleteTextboxPrompt(promptName) { // 私有辅助函数
    ui.showLoading("删除文本框提示词...");
    api.deleteTextboxPromptApi(promptName)
        .then(response => {
            ui.hideLoading();
            ui.showGeneralMessage(`文本框提示词 "${promptName}" 删除成功！`, "success");
            initializeTextboxPromptSettings(); // 重新加载列表
        })
        .catch(error => {
            ui.hideLoading();
            ui.showGeneralMessage(`删除文本框提示词失败: ${error.message}`, "error");
        });
}


/**
 * 初始化可折叠面板
 */
function initializeCollapsiblePanels() { // 私有辅助函数
    const collapsibleHeaders = $(".collapsible-header");
    collapsibleHeaders.on("click", function() {
        const header = $(this);
        const content = header.next(".collapsible-content");
        header.toggleClass("collapsed");
        content.toggleClass("collapsed");
        const icon = header.find(".toggle-icon");
        icon.text(header.hasClass("collapsed") ? "▶" : "▼");
    });
    collapsibleHeaders.each(function(index) {
        if (index > 0) {
            const header = $(this);
            header.addClass("collapsed");
            header.next(".collapsible-content").addClass("collapsed");
            header.find(".toggle-icon").text("▶");
        }
    });
}

/**
 * 初始化亮暗模式切换
 */
function initializeThemeMode() { // 私有辅助函数
    const savedTheme = localStorage.getItem('themeMode');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
        document.body.classList.remove('light-mode');
    } else {
        document.body.classList.add('light-mode');
        document.body.classList.remove('dark-mode');
    }
    // 事件绑定在 events.js 中处理
}

/**
 * 检查初始模型提供商并更新 UI
 */
function checkInitialModelProvider() { // 私有辅助函数
    const selectedProvider = $('#modelProvider').val();
    ui.updateApiKeyInputState(selectedProvider === 'ollama' || selectedProvider === 'sakura',
                              selectedProvider === 'ollama' || selectedProvider === 'sakura' ? '本地部署无需API Key' : '请输入API Key');
    ui.toggleOllamaUI(selectedProvider === 'ollama');
    ui.toggleSakuraUI(selectedProvider === 'sakura');
    if (selectedProvider === 'ollama') {
        fetchOllamaModels();
    } else if (selectedProvider === 'sakura') {
        fetchSakuraModels();
    }
}

/**
 * 获取 Ollama 模型列表并更新 UI
 */
export function fetchOllamaModels() { // 导出
    ui.showLoading("正在获取本地 Ollama 模型...");
    api.testOllamaConnectionApi()
        .then(response => {
            ui.hideLoading();
            ui.updateOllamaModelList(response.models);
            if (response.models && response.models.length > 0 && !$('#modelName').val()) {
                $('#modelName').val(response.models[0]);
                $('#ollamaModelsList .model-button').first().addClass('selected');
            }
        })
        .catch(error => {
            ui.hideLoading();
            ui.updateOllamaModelList([]);
            console.error("获取 Ollama 模型列表失败:", error);
            ui.showGeneralMessage(`获取 Ollama 模型列表失败: ${error.message}`, "error");
        });
}

/**
 * 获取 Sakura 模型列表并更新 UI
 */
export function fetchSakuraModels() { // 导出
    ui.showLoading("正在获取本地 Sakura 模型...");
    api.testSakuraConnectionApi()
        .then(response => {
            ui.hideLoading();
            ui.updateSakuraModelList(response.models);
            if (response.models && response.models.length > 0 && !$('#modelName').val()) {
                $('#modelName').val(response.models[0]);
                $('#sakuraModelsList .model-button').first().addClass('selected');
            }
        })
        .catch(error => {
            ui.hideLoading();
            ui.updateSakuraModelList([]);
            console.error("获取 Sakura 模型列表失败:", error);
            ui.showGeneralMessage(`获取 Sakura 模型列表失败: ${error.message}`, "error");
        });
}


/**
 * 处理文件（图片或 PDF）
 * @param {FileList} files - 用户选择或拖放的文件列表
 */
export function handleFiles(files) { // 导出
    if (!files || files.length === 0) return;

    ui.showLoading("处理文件中...");
    ui.hideError();

    const imagePromises = [];
    const pdfFiles = [];

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (file.type.startsWith('image/')) {
            imagePromises.push(processImageFile(file));
        } else if (file.type === 'application/pdf') {
            pdfFiles.push(file);
        } else {
            console.warn(`不支持的文件类型: ${file.name} (${file.type})`);
        }
    }

    Promise.all(imagePromises)
        .then(() => {
            if (pdfFiles.length > 0) {
                return processPDFFiles(pdfFiles);
            }
        })
        .then(() => {
            ui.hideLoading();
            if (state.images.length > 0) {
                sortImagesByName();
                ui.renderThumbnails();
                switchImage(0); // 显示第一张
            } else {
                ui.showError("未能成功加载任何图片。");
            }
            ui.updateButtonStates();
        })
        .catch(error => {
            ui.hideLoading();
            console.error("处理文件失败:", error);
            ui.showError(`处理文件时出错: ${error.message || error}`);
            ui.updateButtonStates();
        });
}

/**
 * 处理单个图片文件
 * @param {File} file - 图片文件
 * @returns {Promise<void>}
 */
function processImageFile(file) { // 私有
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            const originalDataURL = e.target.result;
            state.addImage({
                originalDataURL: originalDataURL,
                translatedDataURL: null, cleanImageData: null,
                bubbleTexts: [], bubbleCoords: [], originalTexts: [], textboxTexts: [],
                bubbleSettings: null, fileName: file.name,
                fontSize: state.defaultFontSize, autoFontSize: $('#autoFontSize').is(':checked'),
                fontFamily: state.defaultFontFamily, layoutDirection: state.defaultLayoutDirection,
                showOriginal: false, translationFailed: false,
                originalUseInpainting: undefined, originalUseLama: undefined,
            });
            resolve();
        };
        reader.onerror = (error) => reject(error);
        reader.readAsDataURL(file);
    });
}

/**
 * 处理 PDF 文件列表
 * @param {Array<File>} pdfFiles - PDF 文件数组
 * @returns {Promise<void>}
 */
function processPDFFiles(pdfFiles) { // 私有
    return pdfFiles.reduce((promiseChain, file) => {
        return promiseChain.then(() => {
            ui.showLoading(`处理 PDF: ${file.name}...`);
            const formData = new FormData();
            formData.append('pdfFile', file);
            return api.uploadPdfApi(formData)
                .then(response => {
                    if (response.images && response.images.length > 0) {
                        response.images.forEach((imageData, idx) => {
                            const originalDataURL = "data:image/png;base64," + imageData;
                            const pdfFileName = `${file.name}_页面${idx+1}`;
                            state.addImage({
                                originalDataURL: originalDataURL,
                                translatedDataURL: null, cleanImageData: null,
                                bubbleTexts: [], bubbleCoords: [], originalTexts: [], textboxTexts: [],
                                bubbleSettings: null, fileName: pdfFileName,
                                fontSize: state.defaultFontSize, autoFontSize: $('#autoFontSize').is(':checked'),
                                fontFamily: state.defaultFontFamily, layoutDirection: state.defaultLayoutDirection,
                                showOriginal: false, translationFailed: false,
                                originalUseInpainting: undefined, originalUseLama: undefined,
                            });
                        });
                    } else {
                        ui.showGeneralMessage(`PDF文件 ${file.name} 中没有检测到图片`, "warning");
                    }
                })
                .catch(error => {
                    console.error(`处理PDF文件 ${file.name} 失败:`, error);
                    ui.showGeneralMessage(`处理PDF文件 ${file.name} 失败: ${error.message}`, "error");
                });
        });
    }, Promise.resolve());
}


/**
 * 按文件名对图片状态数组进行排序
 */
function sortImagesByName() { // 私有
    state.images.sort((a, b) => {
        return a.fileName.localeCompare(b.fileName, undefined, { numeric: true, sensitivity: 'base' });
    });
}

/**
 * 切换显示的图片
 * @param {number} index - 要显示的图片索引
 */
export function switchImage(index) { // 导出
    if (index < 0 || index >= state.images.length) return;

    if (state.editModeActive) {
        const currentImage = state.getCurrentImage();
        if (currentImage) {
            currentImage.bubbleSettings = JSON.parse(JSON.stringify(state.bubbleSettings));
            currentImage.bubbleTexts = state.bubbleSettings.map(s => s.text);
        }
    }

    state.setCurrentImageIndex(index);
    const imageData = state.getCurrentImage();
    console.log("切换到图片:", index, imageData.fileName);

    ui.hideError();
    ui.hideLoading();
    $('#translatingMessage').hide();

    ui.updateTranslatedImage(imageData.showOriginal ? imageData.originalDataURL : (imageData.translatedDataURL || imageData.originalDataURL));
    $('#toggleImageButton').text(imageData.showOriginal ? '显示翻译图' : '显示原图');
    ui.updateImageSizeDisplay($('#imageSize').val());
    ui.showResultSection(true);
    ui.updateDetectedTextDisplay();
    ui.updateRetranslateButton();

    // 更新全局设置控件
    $('#autoFontSize').prop('checked', imageData.autoFontSize);
    $('#fontSize').prop('disabled', imageData.autoFontSize).val(imageData.autoFontSize ? '-' : imageData.fontSize);
    $('#fontFamily').val(imageData.fontFamily);
    $('#layoutDirection').val(imageData.layoutDirection);
    const useInpainting = imageData.originalUseInpainting;
    const useLama = imageData.originalUseLama;
    let repairMethod = 'false';
    if (useLama) repairMethod = 'lama';
    else if (useInpainting) repairMethod = 'true';
    $('#useInpainting').val(repairMethod);
    ui.toggleInpaintingOptions(useInpainting || useLama, !useInpainting && !useLama);
    if(useInpainting || useLama) {
        $('#inpaintingStrength').val(imageData.inpaintingStrength || constants.DEFAULT_INPAINTING_STRENGTH);
        $('#inpaintingStrengthValue').text($('#inpaintingStrength').val());
        $('#blendEdges').prop('checked', imageData.blendEdges !== undefined ? imageData.blendEdges : true);
    }
     $('#fillColor').val(imageData.fillColor || constants.DEFAULT_FILL_COLOR);


    ui.updateButtonStates();
    $('.thumbnail-item').removeClass('active');
    $(`.thumbnail-item[data-index="${index}"]`).addClass('active');
    ui.scrollToActiveThumbnail();

    if (state.editModeActive) {
        editMode.initBubbleSettings();
        if (state.bubbleSettings.length > 0) {
            editMode.selectBubble(0);
        } else {
            ui.updateBubbleEditArea(-1);
            ui.updateBubbleHighlight(-1);
        }
    } else {
        editMode.exitEditMode();
    }
}

/**
 * 翻译当前图片
 */
export function translateCurrentImage() { // 导出
    const currentImage = state.getCurrentImage();
    if (!currentImage) return;

    // 移除重复的showLoading调用，因为events.js已经显示了消息
    ui.showTranslatingIndicator(state.currentImageIndex);

    const repairSettings = ui.getRepairSettings(); // 需要在 ui.js 中导出
    const isAutoFontSize = $('#autoFontSize').is(':checked');
    const fontSize = isAutoFontSize ? 'auto' : $('#fontSize').val();

    const params = {
        image: currentImage.originalDataURL.split(',')[1],
        target_language: $('#targetLanguage').val(),
        source_language: $('#sourceLanguage').val(),
        fontSize: fontSize,
        autoFontSize: isAutoFontSize,
        api_key: $('#apiKey').val(),
        model_name: $('#modelName').val(),
        model_provider: $('#modelProvider').val(),
        fontFamily: $('#fontFamily').val(),
        textDirection: $('#layoutDirection').val(),
        prompt_content: $('#promptContent').val(),
        use_textbox_prompt: $('#enableTextboxPrompt').prop('checked'),
        textbox_prompt_content: $('#textboxPromptContent').val(),
        use_inpainting: repairSettings.useInpainting,
        use_lama: repairSettings.useLama,
        blend_edges: $('#blendEdges').prop('checked'),
        inpainting_strength: parseFloat($('#inpaintingStrength').val()),
        fill_color: $('#fillColor').val(),
        text_color: $('#textColor').val(),
        rotation_angle: parseFloat($('#rotationAngle').val() || '0'),
        skip_translation: false, skip_ocr: false, remove_only: false,
    };

    api.translateImageApi(params)
        .then(response => {
            ui.hideTranslatingIndicator(state.currentImageIndex);

            state.updateCurrentImageProperty('translatedDataURL', 'data:image/png;base64,' + response.translated_image);
            state.updateCurrentImageProperty('cleanImageData', response.clean_image);
            state.updateCurrentImageProperty('bubbleTexts', response.bubble_texts);
            state.updateCurrentImageProperty('bubbleCoords', response.bubble_coords);
            state.updateCurrentImageProperty('originalTexts', response.original_texts);
            state.updateCurrentImageProperty('textboxTexts', response.textbox_texts);
            state.updateCurrentImageProperty('fontSize', fontSize);
            state.updateCurrentImageProperty('autoFontSize', isAutoFontSize);
            state.updateCurrentImageProperty('fontFamily', params.fontFamily);
            state.updateCurrentImageProperty('layoutDirection', params.textDirection);
            state.updateCurrentImageProperty('showOriginal', false);
            state.updateCurrentImageProperty('translationFailed', false);
            state.updateCurrentImageProperty('originalUseInpainting', params.use_inpainting);
            state.updateCurrentImageProperty('originalUseLama', params.use_lama);
            state.updateCurrentImageProperty('inpaintingStrength', params.inpainting_strength);
            state.updateCurrentImageProperty('blendEdges', params.blend_edges);
            state.updateCurrentImageProperty('fillColor', params.fill_color);
            state.updateCurrentImageProperty('textColor', params.text_color);
            state.updateCurrentImageProperty('bubbleSettings', null);
            state.setBubbleSettings([]);

            switchImage(state.currentImageIndex); // 重新加载以更新所有 UI
            ui.updateDetectedTextDisplay();
            ui.updateRetranslateButton();
            ui.updateButtonStates();

            api.saveModelInfoApi(params.model_provider, params.model_name);
        })
        .catch(error => {
            ui.hideTranslatingIndicator(state.currentImageIndex);
            state.updateCurrentImageProperty('translationFailed', true);
            ui.renderThumbnails();
            ui.showError(`翻译失败: ${error.message}`);
            ui.updateButtonStates();
            ui.updateRetranslateButton();
        });
}

/**
 * 翻译所有图片
 */
export function translateAllImages() { // 导出
    const targetLanguage = $('#targetLanguage').val();
    const sourceLanguage = $('#sourceLanguage').val();
    const isAutoFontSize = $('#autoFontSize').is(':checked');
    const fontSize = isAutoFontSize ? 'auto' : $('#fontSize').val();
    const apiKey = $('#apiKey').val();
    const modelName = $('#modelName').val();
    const modelProvider = $('#modelProvider').val();
    const fontFamily = $('#fontFamily').val();
    const textDirection = $('#layoutDirection').val();
    const useTextboxPrompt = $('#enableTextboxPrompt').prop('checked');
    const textboxPromptContent = $('#textboxPromptContent').val();
    const fillColor = $('#fillColor').val();
    const repairSettings = ui.getRepairSettings();
    const useInpainting = repairSettings.useInpainting;
    const useLama = repairSettings.useLama;
    const inpaintingStrength = parseFloat($('#inpaintingStrength').val());
    const blendEdges = $('#blendEdges').prop('checked');
    const promptContent = $('#promptContent').val();

    // 参数检查... (省略，假设与 translateCurrentImage 类似)

    let currentIndex = 0;
    const totalImages = state.images.length;
    ui.updateProgressBar(0, `0/${totalImages}`);
    ui.showGeneralMessage("批量翻译中...", "info", false, 0);

    function processNextImage() {
        if (currentIndex >= totalImages) {
            ui.updateProgressBar(100, `${totalImages}/${totalImages}`);
            $(".message.info").fadeOut(300, function() { $(this).remove(); });
            ui.updateButtonStates();
            ui.showGeneralMessage('所有图片翻译完成', "success");
            return;
        }

        ui.updateProgressBar((currentIndex / totalImages) * 100, `${currentIndex}/${totalImages}`);
        ui.showTranslatingIndicator(currentIndex);
        const imageData = state.images[currentIndex];

        const data = {
            image: imageData.originalDataURL.split(',')[1],
            target_language: targetLanguage, source_language: sourceLanguage,
            fontSize: fontSize, autoFontSize: isAutoFontSize,
            api_key: apiKey, model_name: modelName, model_provider: modelProvider,
            fontFamily: fontFamily, textDirection: textDirection,
            prompt_content: promptContent, use_textbox_prompt: useTextboxPrompt,
            textbox_prompt_content: textboxPromptContent, use_inpainting: useInpainting,
            use_lama: useLama, blend_edges: blendEdges, inpainting_strength: inpaintingStrength,
            fill_color: fillColor, 
            text_color: $('#textColor').val(),
            rotation_angle: parseFloat($('#rotationAngle').val() || '0'),
            skip_translation: false, skip_ocr: false, remove_only: false,
        };

        api.translateImageApi(data)
            .then(response => {
                ui.hideTranslatingIndicator(currentIndex);
                // 更新图片状态
                imageData.translatedDataURL = 'data:image/png;base64,' + response.translated_image;
                imageData.cleanImageData = response.clean_image;
                imageData.bubbleTexts = response.bubble_texts;
                imageData.bubbleCoords = response.bubble_coords;
                imageData.originalTexts = response.original_texts;
                imageData.textboxTexts = response.textbox_texts;
                imageData.fontSize = fontSize; imageData.autoFontSize = isAutoFontSize;
                imageData.fontFamily = fontFamily; imageData.layoutDirection = textDirection;
                imageData.showOriginal = false; imageData.translationFailed = false;
                imageData.originalUseInpainting = useInpainting; imageData.originalUseLama = useLama;
                imageData.inpaintingStrength = inpaintingStrength; imageData.blendEdges = blendEdges;
                imageData.fillColor = fillColor;
                imageData.textColor = $('#textColor').val();
                imageData.bubbleSettings = null; // 清空旧设置

                ui.renderThumbnails(); // 更新缩略图
                if (currentIndex === state.currentImageIndex) {
                    switchImage(state.currentImageIndex); // 更新当前显示
                }
            })
            .catch(error => {
                ui.hideTranslatingIndicator(currentIndex);
                console.error(`图片 ${currentIndex} 翻译失败:`, error);
                imageData.translationFailed = true;
                ui.renderThumbnails(); // 更新缩略图显示失败
            })
            .finally(() => {
                currentIndex++;
                processNextImage(); // 处理下一张
            });
    }
    processNextImage(); // 开始处理
}

// --- 其他需要导出的函数 ---
// (downloadCurrentImage, downloadAllImages, applySettingsToAll, removeBubbleTextOnly 等)
// 需要将它们的实现从 script.js 移到这里，并添加 export

/**
 * 下载当前翻译后的图片
 */
export function downloadCurrentImage() {
    const currentImage = state.getCurrentImage();
    if (currentImage && currentImage.translatedDataURL) {
        ui.showDownloadingMessage(true);
        try {
            const base64Data = currentImage.translatedDataURL.split(',')[1];
            const byteCharacters = atob(base64Data);
            const byteArrays = [];
            for (let offset = 0; offset < byteCharacters.length; offset += 512) {
                const slice = byteCharacters.slice(offset, offset + 512);
                const byteNumbers = new Array(slice.length);
                for (let i = 0; i < slice.length; i++) byteNumbers[i] = slice.charCodeAt(i);
                byteArrays.push(new Uint8Array(byteNumbers));
            }
            const blob = new Blob(byteArrays, {type: 'image/png'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            let fileName = currentImage.fileName || `image_${state.currentImageIndex}.png`;
            fileName = `translated_${fileName.replace(/\.[^/.]+$/, "")}.png`;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (e) {
            console.error("下载图片失败:", e);
            ui.showGeneralMessage("下载图片失败", "error");
        } finally {
            ui.showDownloadingMessage(false);
        }
    } else {
        ui.showGeneralMessage("没有可下载的翻译图片", "warning");
    }
}

/**
 * 下载所有翻译后的图片
 */
export function downloadAllImages() {
    const selectedFormat = $('#downloadFormat').val();
    ui.showDownloadingMessage(true); // 显示下载中

    // 延迟执行下载，给 UI 更新时间
    setTimeout(() => {
        try {
            switch(selectedFormat) {
                case 'zip': downloadAllAsZip(); break;
                case 'pdf': downloadAllAsPDF(); break;
                case 'cbz': downloadAllAsCBZ(); break;
                default: downloadAllAsZip();
            }
        } catch (e) {
            console.error("下载所有图片时出错:", e);
            ui.showGeneralMessage("下载失败", "error");
        } finally {
             // 注意：ZIP/PDF/CBZ 函数内部会隐藏进度条和消息
             // ui.showDownloadingMessage(false); // 不在这里隐藏
        }
    }, 100); // 短暂延迟
}

// --- ZIP/PDF/CBZ 下载辅助函数 (私有) ---
function downloadAllAsZip() {
    ui.updateProgressBar(0, "准备 ZIP 下载...");
    $("#translationProgressBar").show();
    const zip = new JSZip(); // 确保 JSZip 已加载
    const translatedImages = state.images.filter(img => img.translatedDataURL);
    if (translatedImages.length === 0) { /* ... */ $("#translationProgressBar").hide(); return; }
    let processed = 0;
    translatedImages.forEach((imgData, i) => {
        const base64Data = imgData.translatedDataURL.split(',')[1];
        const fileName = imgData.fileName || `image_${i}.png`;
        const safeFileName = `translated_${String(i).padStart(3, '0')}_${fileName.replace(/[^a-zA-Z0-9.]/g, '_')}`; // 清理文件名
        zip.file(safeFileName + '.png', base64Data, {base64: true});
        processed++;
        ui.updateProgressBar((processed / translatedImages.length) * 100, `压缩进度: ${processed}/${translatedImages.length}`);
    });
    zip.generateAsync({type:"blob"})
        .then(content => {
            const link = document.createElement('a');
            link.href = URL.createObjectURL(content);
            link.download = "translated_images.zip";
            link.click();
            URL.revokeObjectURL(link.href);
            $("#translationProgressBar").hide();
            ui.showGeneralMessage(`已成功下载 ${translatedImages.length} 张翻译图片 (ZIP)`, "success");
        })
        .catch(error => { /* ... */ $("#translationProgressBar").hide(); });
}
function downloadAllAsPDF() {
    ui.updateProgressBar(0, "准备 PDF 下载...");
    $("#translationProgressBar").show();
    const { jsPDF } = window.jspdf; // 确保 jsPDF 已加载
    const pdf = new jsPDF({orientation: 'p', unit: 'px', format: 'a4'}); // 使用像素单位可能更直观
    const translatedImages = state.images.filter(img => img.translatedDataURL);
    if (translatedImages.length === 0) { /* ... */ $("#translationProgressBar").hide(); return; }
    let processed = 0;
    const processImage = (index) => {
        if (index >= translatedImages.length) {
            pdf.save('translated_images.pdf');
            $("#translationProgressBar").hide();
            ui.showGeneralMessage(`已成功创建 PDF 文件`, "success");
            return;
        }
        if (index > 0) pdf.addPage();
        const img = new Image();
        img.onload = function() {
            const pdfWidth = pdf.internal.pageSize.getWidth();
            const pdfHeight = pdf.internal.pageSize.getHeight();
            const ratio = Math.min(pdfWidth / img.naturalWidth, pdfHeight / img.naturalHeight);
            const imgWidth = img.naturalWidth * ratio;
            const imgHeight = img.naturalHeight * ratio;
            const x = (pdfWidth - imgWidth) / 2;
            const y = (pdfHeight - imgHeight) / 2;
            pdf.addImage(img, 'PNG', x, y, imgWidth, imgHeight);
            processed++;
            ui.updateProgressBar((processed / translatedImages.length) * 100, `PDF 创建进度: ${processed}/${translatedImages.length}`);
            processImage(index + 1);
        };
        img.onerror = () => { // 处理图片加载失败
             console.error(`加载图片 ${index} 失败，跳过`);
             processed++;
             ui.updateProgressBar((processed / translatedImages.length) * 100, `PDF 创建进度: ${processed}/${translatedImages.length}`);
             processImage(index + 1);
        }
        img.src = translatedImages[index].translatedDataURL;
    };
    processImage(0);
}
function downloadAllAsCBZ() {
    ui.updateProgressBar(0, "准备 CBZ 下载...");
    $("#translationProgressBar").show();
    const zip = new JSZip();
    const translatedImages = state.images.filter(img => img.translatedDataURL);
    if (translatedImages.length === 0) { /* ... */ $("#translationProgressBar").hide(); return; }
    let processed = 0;
    translatedImages.forEach((imgData, i) => {
        const base64Data = imgData.translatedDataURL.split(',')[1];
        const fileName = imgData.fileName || `image_${i}.png`;
        const safeFileName = `${String(i).padStart(3, '0')}_${fileName.replace(/[^a-zA-Z0-9.]/g, '_')}`;
        zip.file(safeFileName + '.png', base64Data, {base64: true}); // 确保是 png
        processed++;
        ui.updateProgressBar((processed / translatedImages.length) * 100, `CBZ 创建进度: ${processed}/${translatedImages.length}`);
    });
    zip.generateAsync({type:"blob"})
        .then(content => {
            const link = document.createElement('a');
            link.href = URL.createObjectURL(content);
            link.download = "translated_images.cbz";
            link.click();
            URL.revokeObjectURL(link.href);
            $("#translationProgressBar").hide();
            ui.showGeneralMessage(`已成功创建 CBZ 文件`, "success");
        })
        .catch(error => { /* ... */ $("#translationProgressBar").hide(); });
}

/**
 * 将当前字体设置应用到所有图片
 */
export function applySettingsToAll() { // 导出
    const currentImage = state.getCurrentImage();
    if (!currentImage) {
        ui.showGeneralMessage("请先选择一张图片以应用其设置", "warning");
        return;
    }
    if (state.images.length <= 1) {
        ui.showGeneralMessage("只有一张图片，无需应用到全部", "info");
        return;
    }

    const settingsToApply = {
        fontSize: currentImage.fontSize,
        autoFontSize: currentImage.autoFontSize,
        fontFamily: currentImage.fontFamily,
        textDirection: currentImage.layoutDirection,
        textColor: $('#textColor').val(), // 使用当前的全局颜色
        rotationAngle: 0 // 全局应用时不应用旋转
    };

    ui.showLoading("应用设置到所有图片...");

    // 准备数据给后端（如果需要后端重新渲染）
    // 或者在前端直接修改状态并调用 reRenderFullImage (如果性能允许)

    // --- 前端直接修改状态并渲染 ---
    state.images.forEach((img, index) => {
        if (img.translatedDataURL) { // 只修改已翻译的图片
            img.fontSize = settingsToApply.fontSize;
            img.autoFontSize = settingsToApply.autoFontSize;
            img.fontFamily = settingsToApply.fontFamily;
            img.layoutDirection = settingsToApply.textDirection;
            // 更新 bubbleSettings (如果存在)
            if (img.bubbleSettings) {
                img.bubbleSettings = img.bubbleSettings.map(setting => ({
                    ...setting,
                    fontSize: settingsToApply.fontSize,
                    autoFontSize: settingsToApply.autoFontSize,
                    fontFamily: settingsToApply.fontFamily,
                    textDirection: settingsToApply.textDirection,
                    textColor: settingsToApply.textColor,
                    rotationAngle: settingsToApply.rotationAngle
                }));
            }
        }
    });

    // 重新渲染当前图片以立即看到效果
    if (state.getCurrentImage()?.translatedDataURL) {
        editMode.reRenderFullImage(true); // 标记为全局变更
    }
    ui.hideLoading();
    ui.showGeneralMessage("设置已应用到所有已翻译图片（下次查看时生效）", "success");
    // 注意：这种前端修改方式不会重新渲染所有图片，只在切换到对应图片时生效
    // 如果需要立即重新渲染所有图片，需要调用后端 API
}

/**
 * 仅消除当前图片的气泡文字
 */
export function removeBubbleTextOnly() { // 导出
    const currentImage = state.getCurrentImage();
    if (!currentImage) return;

    // 移除showLoading调用，events.js已处理显示消息
    ui.showTranslatingIndicator(state.currentImageIndex);

    const repairSettings = ui.getRepairSettings();
    const isAutoFontSize = $('#autoFontSize').is(':checked');
    const fontSize = isAutoFontSize ? 'auto' : $('#fontSize').val();

    const params = {
        image: currentImage.originalDataURL.split(',')[1],
        target_language: $('#targetLanguage').val(),
        source_language: $('#sourceLanguage').val(),
        fontSize: fontSize, autoFontSize: isAutoFontSize,
        api_key: $('#apiKey').val(), 
        model_name: $('#modelName').val(),
        model_provider: $('#modelProvider').val(), 
        fontFamily: $('#fontFamily').val(),
        textDirection: $('#layoutDirection').val(),
        prompt_content: $('#promptContent').val(),
        use_textbox_prompt: $('#enableTextboxPrompt').prop('checked'),
        textbox_prompt_content: $('#textboxPromptContent').val(),
        use_inpainting: repairSettings.useInpainting,
        use_lama: repairSettings.useLama,
        blend_edges: $('#blendEdges').prop('checked'),
        inpainting_strength: parseFloat($('#inpaintingStrength').val()),
        fill_color: $('#fillColor').val(),
        text_color: $('#textColor').val(),
        rotation_angle: parseFloat($('#rotationAngle').val() || '0'),
        skip_translation: true, // 告诉后端跳过翻译步骤
        skip_ocr: false, // 仍然需要OCR来定位文本
        remove_only: true, // 仅消除文字
    };

    api.translateImageApi(params)
        .then(response => {
            // 移除hideLoading调用
            ui.hideTranslatingIndicator(state.currentImageIndex);

            // 更新当前图片对象
            currentImage.translatedDataURL = 'data:image/png;base64,' + response.translated_image;
            currentImage.cleanImageData = response.clean_image;
            currentImage.bubbleTexts = response.bubble_texts || [];  // 可能为空
            currentImage.bubbleCoords = response.bubble_coords || [];
            currentImage.originalTexts = response.original_texts || [];
            currentImage.textboxTexts = response.textbox_texts || [];
            currentImage.fontSize = fontSize;
            currentImage.autoFontSize = isAutoFontSize;
            currentImage.fontFamily = $('#fontFamily').val();
            currentImage.layoutDirection = $('#layoutDirection').val();
            currentImage.showOriginal = false;
            // 移除之前的翻译失败标记（如果有）
            currentImage.translationFailed = false;

            currentImage.originalUseInpainting = params.use_inpainting;
            currentImage.originalUseLama = params.use_lama;
            currentImage.inpaintingStrength = params.inpainting_strength;
            currentImage.blendEdges = params.blend_edges;
            currentImage.fillColor = params.fill_color;
            currentImage.textColor = params.text_color;
            currentImage.bubbleSettings = null;

            switchImage(state.currentImageIndex); // 重新加载以更新所有 UI
            ui.showGeneralMessage("文字已消除", "success");
            ui.updateButtonStates();
            ui.updateRetranslateButton();
            ui.updateDetectedTextDisplay(); // 显示检测到的文本
        })
        .catch(error => {
            // 移除hideLoading调用
            ui.hideTranslatingIndicator(state.currentImageIndex);
            ui.showGeneralMessage(`操作失败: ${error.message}`, "error");
            ui.updateButtonStates();
        });
}

/**
 * 检查当前图片是否有翻译失败的句子
 * @returns {boolean}
 */
export function checkForFailedTranslations() { // 导出
    const currentImage = state.getCurrentImage();
    if (!currentImage) return false;
    const bubbleTexts = currentImage.bubbleTexts || [];
    const textboxTexts = currentImage.textboxTexts || [];
    const checkList = state.useTextboxPrompt ? textboxTexts : bubbleTexts;
    return checkList.some(text => text && text.includes("翻译失败"));
}


// --- 应用启动 ---
$(document).ready(initializeApp);
