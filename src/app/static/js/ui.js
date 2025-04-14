// src/app/static/js/ui.js

// 引入状态模块和常量模块
import * as state from './state.js';
import * as constants from './constants.js'; // <--- 添加导入
// 引入 jQuery (假设全局加载)
// import $ from 'jquery';

// --- DOM 元素引用 ---
// 将获取 DOM 元素的操作放在函数内部或确保在 DOM Ready 后执行
// 或者在 main.js 中获取并传递，但为了减少改动，我们假设在调用时 DOM 已准备好

// --- UI 更新函数 ---

/**
 * 显示加载状态
 * @param {string} [message="处理中，请稍候..."] - 显示的消息文本
 */
export function showLoading(message = "处理中，请稍候...") {
    // 使用通用消息提示代替直接显示加载消息
    showGeneralMessage(message, "info", false, 0);
    $("#loadingAnimation").show();
    $("#errorMessage").hide();
    // 禁用按钮在 updateButtonStates 中处理
    updateButtonStates(); // 调用更新按钮状态
}

/**
 * 隐藏加载状态
 */
export function hideLoading() {
    // 移除所有info类型的通用消息
    $(".message.info").fadeOut(300, function() { $(this).remove(); });
    $("#loadingAnimation").hide();
    updateButtonStates(); // 调用更新按钮状态
}

/**
 * 显示错误消息
 * @param {string} message - 错误消息文本
 */
export function showError(message) {
    $("#errorMessage").text(message).show();
    hideLoading(); // 出错时隐藏加载
}

/**
 * 隐藏错误消息
 */
export function hideError() {
    $("#errorMessage").hide();
}

/**
 * 显示/隐藏结果区域
 * @param {boolean} show - 是否显示
 */
export function showResultSection(show) {
    if (show) {
        $("#result-section").show();
        $("#detectedTextInfo").show();
    } else {
        $("#result-section").hide();
        $("#detectedTextInfo").hide();
    }
}

/**
 * 更新翻译后的图片显示
 * @param {string | null} dataURL - 图片的 Base64 Data URL，或 null 清除图片
 */
export function updateTranslatedImage(dataURL) {
    const translatedImageDisplay = $("#translatedImageDisplay");
    const downloadButton = $("#downloadButton");
    const toggleImageButton = $('#toggleImageButton');

    if (dataURL) {
        translatedImageDisplay.attr('src', dataURL).show();
        downloadButton.show();
        toggleImageButton.show();
    } else {
        translatedImageDisplay.removeAttr('src').hide();
        downloadButton.hide();
        toggleImageButton.hide();
    }
}

/**
 * 更新缩略图列表
 */
export function renderThumbnails() {
    const thumbnailList = $("#thumbnail-sidebar #thumbnailList"); // 在函数内获取
    thumbnailList.empty();
    state.images.forEach((imageData, index) => {
        const thumbnailItem = $("<div class='thumbnail-item' data-index='" + index + "'></div>");
        const thumbnailImage = $("<img class='thumbnail-image'>").attr('src', imageData.originalDataURL);
        thumbnailItem.append(thumbnailImage);

        if (index === state.currentImageIndex) {
            thumbnailItem.addClass('active');
        }

        if (imageData.translationFailed) {
             thumbnailItem.addClass('translation-failed');
             thumbnailItem.attr('title', '翻译失败，点击可重试');
             thumbnailItem.append('<span class="error-indicator">!</span>');
        }

        thumbnailItem.data('index', index);
        thumbnailList.append(thumbnailItem);
    });
    scrollToActiveThumbnail();
}

/**
 * 滚动到当前激活的缩略图
 */
export function scrollToActiveThumbnail() {
    const thumbnailList = $("#thumbnail-sidebar #thumbnailList"); // 在函数内获取
    const activeItem = thumbnailList.find('.thumbnail-item.active');
    if (activeItem.length) {
        const listContainer = thumbnailList.parent();
        // 确保 listContainer 是可滚动的元素
        if (listContainer.css('overflow-y') === 'auto' || listContainer.css('overflow-y') === 'scroll') {
            const containerScrollTop = listContainer.scrollTop();
            const containerHeight = listContainer.height();
            // position() 相对于 offset parent，可能不是 listContainer，需要调整
            // 使用 offsetTop 相对于父元素更可靠
            const itemTopRelativeToParent = activeItem[0].offsetTop;
            const itemHeight = activeItem.outerHeight();

            if (itemTopRelativeToParent < containerScrollTop) {
                listContainer.scrollTop(itemTopRelativeToParent);
            } else if (itemTopRelativeToParent + itemHeight > containerScrollTop + containerHeight) {
                listContainer.scrollTop(itemTopRelativeToParent + itemHeight - containerHeight);
            }
        }
    }
}


/**
 * 更新导航按钮（上一张/下一张）的状态
 */
export function updateNavigationButtons() {
    const prevImageButton = $("#prevImageButton"); // 在函数内获取
    const nextImageButton = $("#nextImageButton"); // 在函数内获取
    const numImages = state.images.length;
    const currentIndex = state.currentImageIndex;
    prevImageButton.prop('disabled', currentIndex <= 0);
    nextImageButton.prop('disabled', currentIndex >= numImages - 1);
}

/**
 * 更新所有操作按钮的状态（翻译、清除、删除等）
 */
export function updateButtonStates() {
    const translateButton = $("#translateButton"); // 在函数内获取
    const removeTextOnlyButton = $("#removeTextOnlyButton");
    const translateAllButton = $("#translateAllButton");
    const clearAllImagesButton = $("#clearAllImagesButton");
    const deleteCurrentImageButton = $("#deleteCurrentImageButton");
    const applyFontSettingsToAllButton = $("#applyFontSettingsToAllButton"); // 在函数内获取
    const downloadButton = $("#downloadButton");
    const downloadAllImagesButton = $("#downloadAllImagesButton");
    const toggleImageButton = $('#toggleImageButton');

    const hasImages = state.images.length > 0;
    const hasCurrentImage = state.currentImageIndex >= 0 && state.currentImageIndex < state.images.length;
    // 检查加载动画是否可见来判断是否在加载状态
    const isLoading = $("#loadingAnimation").is(":visible");

    translateButton.prop('disabled', !hasCurrentImage || isLoading);
    removeTextOnlyButton.prop('disabled', !hasCurrentImage || isLoading);
    translateAllButton.prop('disabled', !hasImages || isLoading);
    clearAllImagesButton.prop('disabled', !hasImages || isLoading);
    deleteCurrentImageButton.prop('disabled', !hasCurrentImage || isLoading);
    // 修复 TypeError: applyFontSettingsToAllButton.prop is not a function
    // 确保 applyFontSettingsToAllButton 是有效的 jQuery 对象
    if (applyFontSettingsToAllButton && applyFontSettingsToAllButton.length > 0) {
        applyFontSettingsToAllButton.prop('disabled', !hasImages || isLoading);
    } else {
        console.warn("#applyFontSettingsToAllButton 未找到!");
    }


    let hasTranslated = false;
    if (hasCurrentImage && state.images[state.currentImageIndex].translatedDataURL) {
        hasTranslated = true;
    }
    downloadButton.toggle(hasTranslated && !isLoading);
    toggleImageButton.toggle(hasTranslated && !isLoading);

    const hasAnyTranslated = state.images.some(img => img.translatedDataURL);
    downloadAllImagesButton.toggle(hasAnyTranslated && !isLoading);
    $('#downloadFormat').toggle(hasAnyTranslated && !isLoading);

    updateNavigationButtons(); // 更新翻页按钮 (也应考虑 isLoading)
    $("#prevImageButton").prop('disabled', state.currentImageIndex <= 0 || isLoading);
    $("#nextImageButton").prop('disabled', state.currentImageIndex >= state.images.length - 1 || isLoading);
}


/**
 * 更新检测到的文本显示区域
 */
export function updateDetectedTextDisplay() {
    const detectedTextList = $("#detectedTextList"); // 在函数内获取
    const currentImage = state.getCurrentImage();
    detectedTextList.empty();

    if (currentImage && currentImage.originalTexts && currentImage.originalTexts.length > 0) {
        const originalTexts = currentImage.originalTexts;
        const translatedTexts = state.useTextboxPrompt ?
            (currentImage.textboxTexts || currentImage.bubbleTexts || []) :
            (currentImage.bubbleTexts || []);

        for (let i = 0; i < originalTexts.length; i++) {
            const original = originalTexts[i] || "";
            const translated = translatedTexts[i] || "";
            // 使用 formatTextDisplay 返回的 HTML，所以用 .append() 而不是 .text()
            const formattedHtml = formatTextDisplay(original, translated);
            // 为了正确显示换行和样式，需要将 pre 元素的内容设置为 HTML
            // 或者修改 formatTextDisplay 返回纯文本，然后在这里处理样式
            // 这里选择修改追加方式
            const textNode = document.createElement('span'); // 创建一个临时 span
            textNode.innerHTML = formattedHtml.replace(/\n/g, '<br>'); // 替换换行为 <br>
            detectedTextList.append(textNode);
        }
    } else {
        detectedTextList.text("未检测到文本或尚未翻译");
    }
}

/**
 * 格式化文本显示 (原文 -> 译文) - 返回 HTML 字符串
 * @param {string} originalText - 原文
 * @param {string} translatedText - 译文
 * @returns {string} 格式化后的 HTML 字符串
 */
function formatTextDisplay(originalText, translatedText) {
    let formattedOriginal = (originalText || "").trim();
    formattedOriginal = wrapText(formattedOriginal);

    let formattedTranslation = (translatedText || "").trim();
    if (formattedTranslation.includes("翻译失败")) {
        formattedTranslation = `<span class="translation-error">${formattedTranslation}</span>`;
    } else {
        formattedTranslation = wrapText(formattedTranslation);
    }
    // 返回包含换行符的字符串，让 updateDetectedTextDisplay 处理 <br>
    return `${formattedOriginal}\n${formattedTranslation}\n──────────────────────────\n\n`;
}

/**
 * 文本自动换行
 * @param {string} text - 输入文本
 * @returns {string} 处理换行后的文本
 */
function wrapText(text) {
    // 这个函数保持不变
    const MAX_LINE_LENGTH = 60;
    if (!text || text.length <= MAX_LINE_LENGTH) return text;
    let result = "";
    let currentLine = "";
    for (let i = 0; i < text.length; i++) {
        currentLine += text[i];
        if (currentLine.length >= MAX_LINE_LENGTH) {
            let breakPoint = -1;
            for (let j = currentLine.length - 1; j >= 0; j--) {
                if (['。', '！', '？', '.', '!', '?', '；', ';', '，', ','].includes(currentLine[j])) {
                    breakPoint = j + 1;
                    break;
                }
            }
            if (breakPoint > MAX_LINE_LENGTH * 0.6) {
                result += currentLine.substring(0, breakPoint) + "\n";
                currentLine = currentLine.substring(breakPoint);
            } else {
                result += currentLine + "\n";
                currentLine = "";
            }
        }
    }
    if (currentLine) {
        result += currentLine;
    }
    return result;
}


/**
 * 更新翻译进度条
 * @param {number} percentage - 百分比 (0-100)
 * @param {string} [text=''] - 显示的文本
 */
export function updateProgressBar(percentage, text = '') {
    const translationProgressBar = $("#translationProgressBar"); // 在函数内获取
    const progressBar = $("#translationProgressBar .progress");
    const progressPercent = $("#progressPercent");

    percentage = Math.max(0, Math.min(100, percentage));
    progressBar.css('width', percentage + '%');
    progressPercent.text(text || `${percentage.toFixed(0)}%`);
    if (percentage > 0 && percentage < 100) {
        translationProgressBar.show();
    } else if (percentage >= 100) {
        setTimeout(() => translationProgressBar.hide(), 1000);
    } else {
        translationProgressBar.hide();
    }
}

/**
 * 显示/隐藏下载消息
 * @param {boolean} show - 是否显示
 */
export function showDownloadingMessage(show) {
    if (show) {
        $("#downloadingMessage").show();
        $("#downloadButton").prop('disabled', true);
        $("#downloadAllImagesButton").prop('disabled', true);
    } else {
        $("#downloadingMessage").hide();
        $("#downloadButton").prop('disabled', false);
        $("#downloadAllImagesButton").prop('disabled', false);
    }
}

/**
 * 填充提示词下拉列表
 * @param {Array<string>} promptNames - 提示词名称列表
 * @param {JQuery<HTMLElement>} dropdownElement - 下拉列表的 jQuery 对象
 * @param {JQuery<HTMLElement>} dropdownButton - 触发下拉按钮的 jQuery 对象
 * @param {Function} loadCallback - 选择提示词后的回调函数
 * @param {Function} deleteCallback - 删除提示词后的回调函数
 */
export function populatePromptDropdown(promptNames, dropdownElement, dropdownButton, loadCallback, deleteCallback) {
    dropdownElement.empty();
    const ul = $("<ul></ul>");

    // 添加默认提示词选项
    // 使用常量模块中的 DEFAULT_PROMPT_NAME
    const defaultLi = $("<li></li>").text(constants.DEFAULT_PROMPT_NAME).click(function() {
        loadCallback(constants.DEFAULT_PROMPT_NAME);
        dropdownElement.hide();
    });
    ul.append(defaultLi);

    // 添加已保存的提示词
    if (promptNames && promptNames.length > 0) {
        promptNames.forEach(function(name) {
            const li = $("<li></li>").text(name).click(function() {
                loadCallback(name);
                dropdownElement.hide();
            });
            const deleteButton = $('<span class="delete-prompt-button" title="删除此提示词">×</span>');
            deleteButton.click(function(e) {
                e.stopPropagation();
                if (confirm(`确定要删除提示词 "${name}" 吗？`)) {
                    deleteCallback(name);
                }
            });
            li.append(deleteButton);
            ul.append(li);
        });
    }

    dropdownElement.append(ul);
    dropdownButton.show(); // 总是显示按钮，至少有默认选项
}


/**
 * 更新模型建议列表
 * @param {Array<string>} models - 模型名称列表
 */
export function updateModelSuggestions(models) {
    const modelSuggestionsDiv = $("#model-suggestions"); // 在函数内获取
    const modelNameInput = $("#modelName"); // 在函数内获取
    modelSuggestionsDiv.empty();
    if (models && models.length > 0) {
        const ul = $("<ul></ul>");
        models.forEach(function(model) {
            const li = $("<li></li>").text(model).click(function() {
                modelNameInput.val(model);
                modelSuggestionsDiv.hide();
            });
            ul.append(li);
        });
        modelSuggestionsDiv.append(ul).show();
    } else {
        modelSuggestionsDiv.hide();
    }
}

/**
 * 更新 Ollama 模型按钮列表
 * @param {Array<string>} models - 模型名称列表
 */
export function updateOllamaModelList(models) {
    const modelNameInput = $("#modelName"); // 在函数内获取
    if ($('#ollamaModelsList').length === 0) {
        $('<div id="ollamaModelsList" class="ollama-models-list"></div>').insertAfter(modelNameInput.parent());
    }
    const container = $('#ollamaModelsList');
    container.empty().show();

    if (models && models.length > 0) {
        container.append('<h4>本地可用模型：</h4>');
        const buttonsDiv = $('<div class="model-buttons"></div>');
        models.forEach(model => {
            const button = $(`<button type="button" class="model-button">${model}</button>`);
            buttonsDiv.append(button);
        });
        container.append(buttonsDiv);
    } else {
        container.append('<p class="no-models">未检测到本地模型，请使用命令 <code>ollama pull llama3</code> 拉取模型</p>');
    }
}

/**
 * 更新 Sakura 模型按钮列表
 * @param {Array<string>} models - 模型名称列表
 */
export function updateSakuraModelList(models) {
    const modelNameInput = $("#modelName"); // 在函数内获取
    if ($('#sakuraModelsList').length === 0) {
        $('<div id="sakuraModelsList" class="sakura-models-list"></div>').insertAfter(modelNameInput.parent());
    }
    const container = $('#sakuraModelsList');
    container.empty().show();

    if (models && models.length > 0) {
        container.append('<h4>本地可用Sakura模型：</h4>');
        const buttonsDiv = $('<div class="model-buttons"></div>');
        models.forEach(model => {
            const button = $(`<button type="button" class="model-button">${model}</button>`);
            buttonsDiv.append(button);
        });
        container.append(buttonsDiv);
    } else {
        container.append('<p class="no-models">未检测到本地Sakura模型</p>');
    }
}


/**
 * 更新 API Key 输入框状态
 * @param {boolean} disabled - 是否禁用
 * @param {string} [placeholder='请输入API Key'] - 占位符文本
 */
export function updateApiKeyInputState(disabled, placeholder = '请输入API Key') {
    $('#apiKey').attr('placeholder', placeholder).prop('disabled', disabled);
    if (disabled) {
        $('#apiKey').val('');
    }
}

/**
 * 显示或隐藏 Ollama 相关 UI 元素
 * @param {boolean} show - 是否显示
 */
export function toggleOllamaUI(show) {
    if (show && $('#testOllamaButton').length === 0) {
        $('<button id="testOllamaButton" type="button" class="settings-button">测试Ollama连接</button>').insertAfter($('#modelName').parent());
    }
    $('#testOllamaButton').toggle(show);
    $('#ollamaModelsList').toggle(show);
}

/**
 * 显示或隐藏 Sakura 相关 UI 元素
 * @param {boolean} show - 是否显示
 */
export function toggleSakuraUI(show) {
    if (show && $('#testSakuraButton').length === 0) {
        $('<button id="testSakuraButton" type="button" class="settings-button">测试Sakura连接</button>').insertAfter($('#modelName').parent());
    }
    $('#testSakuraButton').toggle(show);
    $('#sakuraModelsList').toggle(show);
}

/**
 * 更新图片大小滑块和显示
 * @param {number} value - 滑块值 (百分比)
 */
export function updateImageSizeDisplay(value) {
    $("#imageSizeValue").text(value + "%");
    $("#translatedImageDisplay").css("width", value + "%");
}

/**
 * 显示/隐藏修复选项
 * @param {boolean} showInpaintingOptions - 是否显示 MI-GAN/LAMA 选项
 * @param {boolean} showSolidOptions - 是否显示纯色填充选项
 */
export function toggleInpaintingOptions(showInpaintingOptions, showSolidOptions) {
    const inpaintingOptionsDiv = $("#inpaintingOptions"); // 在函数内获取
    const solidColorOptionsDiv = $("#solidColorOptions"); // 在函数内获取
    if (showInpaintingOptions) {
        inpaintingOptionsDiv.slideDown();
    } else {
        inpaintingOptionsDiv.slideUp();
    }
    if (showSolidOptions) {
        solidColorOptionsDiv.slideDown();
    } else {
        solidColorOptionsDiv.slideUp();
    }
}

/**
 * 显示通用消息
 * @param {string} message - 消息内容 (可以是 HTML)
 * @param {'info' | 'success' | 'warning' | 'error'} [type='info'] - 消息类型
 * @param {boolean} [isHTML=false] - 消息内容是否为 HTML
 * @param {number} [duration=5000] - 自动消失时间 (毫秒)，0 表示不自动消失
 */
export function showGeneralMessage(message, type = 'info', isHTML = false, duration = 5000) {
    let messageContainer = $('#messageContainer');
    if (messageContainer.length === 0) {
        messageContainer = $('<div id="messageContainer" class="message-container"></div>');
        $('body').append(messageContainer);
    }
    const messageElement = $('<div class="message"></div>').addClass(type);
    if (isHTML) {
        messageElement.html(message);
    } else {
        messageElement.text(message);
    }
    const closeButton = $('<button class="close-message" title="关闭消息">×</button>');
    closeButton.on('click', function() {
        messageElement.fadeOut(300, function() { $(this).remove(); });
    });
    messageElement.append(closeButton);
    messageContainer.append(messageElement);
    if (duration > 0) {
        setTimeout(function() {
            messageElement.fadeOut(300, function() { $(this).remove(); });
        }, duration);
    }
}

// --- 编辑模式 UI 更新 ---

/**
 * 更新气泡列表 UI
 */
export function updateBubbleListUI() {
    const bubbleList = $("#bubbleList"); // 在函数内获取
    const bubbleCount = $("#bubbleCount"); // 在函数内获取
    bubbleList.empty();
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.bubbleCoords) {
        bubbleCount.text("0");
        return;
    }
    const coords = currentImage.bubbleCoords;
    const settings = state.bubbleSettings;

    bubbleCount.text(coords.length);

    for (let i = 0; i < coords.length; i++) {
        const bubbleItem = $("<div>").addClass("bubble-item").attr("data-index", i);
        if (i === state.selectedBubbleIndex) {
            bubbleItem.addClass("active");
        }
        // 从 settings 获取文本
        const text = (settings[i] && settings[i].text !== undefined) ? settings[i].text : "";
        const preview = text.length > 20 ? text.substring(0, 20) + "..." : text;
        const bubblePreview = $("<div>").addClass("bubble-preview").text(`#${i+1}: ${preview}`);
        bubbleItem.append(bubblePreview);
        bubbleList.append(bubbleItem);
    }
}

/**
 * 更新编辑区域的显示内容
 * @param {number} index - 要显示的设置的气泡索引
 */
export function updateBubbleEditArea(index) {
    const currentBubbleIndexDisplay = $("#currentBubbleIndex"); // 在函数内获取
    const bubbleTextEditor = $("#bubbleTextEditor");
    const bubbleFontSize = $("#bubbleFontSize");
    const autoBubbleFontSizeCheckbox = $("#autoBubbleFontSize");
    const bubbleFontFamily = $("#bubbleFontFamily");
    const bubbleTextDirection = $("#bubbleTextDirection");
    const bubbleTextColor = $("#bubbleTextColor");
    const bubbleRotationAngle = $("#bubbleRotationAngle");
    const bubbleRotationAngleValue = $("#bubbleRotationAngleValue");
    const positionOffsetX = $("#positionOffsetX");
    const positionOffsetY = $("#positionOffsetY");
    const positionOffsetXValue = $("#positionOffsetXValue");
    const positionOffsetYValue = $("#positionOffsetYValue");

    if (index < 0 || index >= state.bubbleSettings.length) {
        // 清空编辑区
        currentBubbleIndexDisplay.text("-");
        bubbleTextEditor.val("");
        autoBubbleFontSizeCheckbox.prop('checked', false);
        bubbleFontSize.prop('disabled', false).val(state.defaultFontSize);
        bubbleFontFamily.val(state.defaultFontFamily);
        bubbleTextDirection.val(state.defaultLayoutDirection);
        bubbleTextColor.val(state.defaultTextColor);
        bubbleRotationAngle.val(0);
        bubbleRotationAngleValue.text('0°');
        positionOffsetX.val(0);
        positionOffsetY.val(0);
        positionOffsetXValue.text(0);
        positionOffsetYValue.text(0);
        return;
    }

    const setting = state.bubbleSettings[index];
    currentBubbleIndexDisplay.text(index + 1);
    bubbleTextEditor.val(setting.text || "");

    if (setting.autoFontSize) {
        autoBubbleFontSizeCheckbox.prop('checked', true);
        bubbleFontSize.prop('disabled', true).val('-');
    } else {
        autoBubbleFontSizeCheckbox.prop('checked', false);
        bubbleFontSize.prop('disabled', false).val(setting.fontSize || state.defaultFontSize);
    }

    bubbleFontFamily.val(setting.fontFamily || state.defaultFontFamily);
    bubbleTextDirection.val(setting.textDirection || state.defaultLayoutDirection);
    bubbleTextColor.val(setting.textColor || state.defaultTextColor);
    bubbleRotationAngle.val(setting.rotationAngle || 0);
    bubbleRotationAngleValue.text((setting.rotationAngle || 0) + '°');

    const position = setting.position || { x: 0, y: 0 };
    positionOffsetX.val(position.x);
    positionOffsetY.val(position.y);
    positionOffsetXValue.text(position.x);
    positionOffsetYValue.text(position.y);
}

/**
 * 添加或更新气泡高亮效果
 * @param {number} bubbleIndex - 要高亮的索引, -1 表示移除高亮
 */
export function updateBubbleHighlight(bubbleIndex) {
    $('.highlight-bubble').remove();

    if (bubbleIndex < 0) return;

    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.bubbleCoords || bubbleIndex >= currentImage.bubbleCoords.length) return;

    const [x1, y1, x2, y2] = currentImage.bubbleCoords[bubbleIndex];
    const highlightElement = $('<div class="highlight-bubble"></div>');
    const imageElement = $('#translatedImageDisplay');
    const imageContainer = $('.image-container');

    // 确保图片已加载且尺寸有效
    const imageNaturalWidth = imageElement[0].naturalWidth;
    const imageNaturalHeight = imageElement[0].naturalHeight;
    if (!imageNaturalWidth || !imageNaturalHeight || imageNaturalWidth === 0 || imageNaturalHeight === 0) {
        console.warn("无法更新高亮：图像尺寸无效或未加载");
        return;
    }

    const imageDisplayWidth = imageElement.width();
    const imageDisplayHeight = imageElement.height();
    const scaleX = imageDisplayWidth / imageNaturalWidth;
    const scaleY = imageDisplayHeight / imageNaturalHeight;
    const imageOffset = imageElement.position();
    const imageLeft = imageOffset ? imageOffset.left : 0;
    const imageTop = imageOffset ? imageOffset.top : 0;

    highlightElement.css({
        'left': `${imageLeft + x1 * scaleX}px`,
        'top': `${imageTop + y1 * scaleY}px`,
        'width': `${(x2 - x1) * scaleX}px`,
        'height': `${(y2 - y1) * scaleY}px`
    });
    imageContainer.append(highlightElement);
}

/**
 * 切换编辑模式的 UI 显示
 * @param {boolean} isActive - 编辑模式是否激活
 */
export function toggleEditModeUI(isActive) {
    const toggleEditModeButton = $("#toggleEditModeButton"); // 在函数内获取
    const editModeContainer = $("#editModeContainer");
    const detectedTextInfo = $("#detectedTextInfo");

    if (isActive) {
        toggleEditModeButton.text("退出编辑模式").addClass("active");
        editModeContainer.show();
        detectedTextInfo.hide();
        $('body').addClass('edit-mode-active');
        updateBubbleListUI();
    } else {
        toggleEditModeButton.text("切换编辑模式").removeClass("active");
        editModeContainer.hide();
        detectedTextInfo.show();
        $('.highlight-bubble').remove();
        $('body').removeClass('edit-mode-active');
        $(window).off('resize.bubbleHighlight');
    }
}

/**
 * 更新重新翻译按钮状态
 */
export function updateRetranslateButton() {
    const retranslateFailedButton = $('#retranslateFailedButton'); // 在函数内获取
    // checkForFailedTranslations 函数需要在 main.js 或 state.js 中定义
    import('./main.js').then(main => {
        if (main.checkForFailedTranslations()) {
            retranslateFailedButton.show();
        } else {
            retranslateFailedButton.hide();
        }
    });
}

/**
 * 显示缩略图上的处理指示器
 * @param {number} index - 缩略图索引
 */
export function showTranslatingIndicator(index) {
    const item = $(`.thumbnail-item[data-index="${index}"]`);
    // 避免重复添加
    if (item.find('.thumbnail-processing-indicator').length === 0) {
        item.append('<div class="thumbnail-processing-indicator">⟳</div>');
        item.addClass('processing'); // 添加处理中样式
    }
}

/**
 * 隐藏缩略图上的处理指示器
 * @param {number} index - 缩略图索引
 */
export function hideTranslatingIndicator(index) {
    const item = $(`.thumbnail-item[data-index="${index}"]`);
    item.find('.thumbnail-processing-indicator').remove();
    item.removeClass('processing'); // 移除处理中样式
}

/**
 * 获取当前选择的气泡填充/修复方式设置
 * @returns {{useInpainting: boolean, useLama: boolean}}
 */
export function getRepairSettings() {
    const repairMethod = $('#useInpainting').val(); // 在函数内获取元素
    // console.log("获取修复设置:", repairMethod); // 可以取消注释用于调试
    return {
        useInpainting: repairMethod === 'true', // MI-GAN
        useLama: repairMethod === 'lama'      // LAMA
    };
}

/**
 * 渲染插件列表到模态窗口
 * @param {Array<object>} plugins - 插件信息数组
 */
export function renderPluginList(plugins) {
    const container = $("#pluginListContainer");
    container.empty();

    if (!plugins || plugins.length === 0) {
        container.html("<p>未找到任何插件。</p>");
        return;
    }

    plugins.forEach(plugin => {
        const pluginDiv = $('<div class="plugin-item"></div>');
        pluginDiv.attr('data-plugin-name', plugin.name);

        const header = $('<div class="plugin-header"></div>');
        header.append(`<span class="plugin-name">${plugin.name}</span>`);
        header.append(`<span class="plugin-version">v${plugin.version}</span>`);
        if (plugin.author) header.append(`<span class="plugin-author">作者: ${plugin.author}</span>`);
        pluginDiv.append(header);
        if (plugin.description) pluginDiv.append(`<p class="plugin-description">${plugin.description}</p>`);

        const actions = $('<div class="plugin-actions"></div>');
        // 启用/禁用开关
        const toggleLabel = $('<label class="plugin-toggle"></label>');
        const toggleCheckbox = $('<input type="checkbox" class="plugin-enable-toggle">');
        toggleCheckbox.prop('checked', plugin.enabled);
        toggleLabel.append(toggleCheckbox);
        toggleLabel.append(plugin.enabled ? ' 已启用' : ' 已禁用');
        actions.append(toggleLabel);

        // --- 添加设置按钮 (如果插件有配置) ---
        // 需要后端在 /api/plugins 返回的数据中包含一个标记，指示是否有配置
        // 或者在点击按钮时再查询 schema
        // 简单起见，我们先假设后端返回的数据包含 `has_config: true/false`
        // if (plugin.has_config) { // 假设后端返回了这个标记
        // 或者更健壮的方式是，总是显示按钮，点击时再检查 schema
        const settingsButton = $('<button class="plugin-settings-button">设置</button>');
        actions.append(settingsButton);
        // }
        // ------------------------------------

        // 删除按钮
        const deleteButton = $('<button class="plugin-delete-button">删除</button>');
        actions.append(deleteButton);

        pluginDiv.append(actions);
        container.append(pluginDiv);
    });
}

/**
 * 显示插件配置模态框
 * @param {string} pluginName - 插件名称
 * @param {Array<object>} schema - 配置项规范数组
 * @param {object} currentConfig - 当前配置值字典
 */
export function showPluginConfigModal(pluginName, schema, currentConfig) {
    // 移除旧的模态框（如果存在）
    $('#pluginConfigModal').remove();

    if (!schema || schema.length === 0) {
        showGeneralMessage(`插件 '${pluginName}' 没有可配置的选项。`, "info");
        return;
    }

    // 创建模态框骨架
    const modal = $('<div id="pluginConfigModal" class="plugin-modal" style="display: block;"></div>');
    const modalContent = $('<div class="plugin-modal-content"></div>');
    const closeButton = $('<span class="plugin-modal-close">&times;</span>');
    const title = $(`<h3>插件设置: ${pluginName}</h3>`);
    const form = $('<form id="pluginConfigForm"></form>');

    // 根据 schema 生成表单项
    schema.forEach(item => {
        const formGroup = $('<div class="plugin-config-item"></div>');
        const label = $(`<label for="plugin-config-${item.name}">${item.label}:</label>`);
        let input;
        const currentValue = currentConfig.hasOwnProperty(item.name) ? currentConfig[item.name] : item.default;

        switch (item.type) {
            case 'number':
                input = $(`<input type="number" id="plugin-config-${item.name}" name="${item.name}">`);
                input.val(currentValue);
                break;
            case 'boolean':
                input = $(`<input type="checkbox" id="plugin-config-${item.name}" name="${item.name}">`);
                input.prop('checked', currentValue);
                // 将 label 包裹 checkbox 以改善点击区域
                label.html(input); // 将 input 放入 label
                label.append(` ${item.label}`); // 在后面添加文本
                input = label; // 让 input 指向整个 label 结构
                break;
            case 'select':
                input = $(`<select id="plugin-config-${item.name}" name="${item.name}"></select>`);
                (item.options || []).forEach(option => {
                    input.append($(`<option value="${option}">${option}</option>`));
                });
                input.val(currentValue);
                break;
            case 'text':
            default:
                input = $(`<input type="text" id="plugin-config-${item.name}" name="${item.name}">`);
                input.val(currentValue);
                break;
        }

        formGroup.append(label);
        // 对于 checkbox，input 已经是 label 了，不需要再 append
        if (item.type !== 'boolean') {
             formGroup.append(input);
        }
        if (item.description) {
            formGroup.append(`<p class="plugin-config-description">${item.description}</p>`);
        }
        form.append(formGroup);
    });

    // 添加保存按钮
    const saveButton = $('<button type="submit" class="plugin-config-save">保存设置</button>');
    form.append(saveButton);

    // 组装模态框
    modalContent.append(closeButton);
    modalContent.append(title);
    modalContent.append(form);
    modal.append(modalContent);

    // 添加到页面并绑定事件
    $('body').append(modal);

    // 绑定关闭事件 (在 events.js 中处理)
    // 绑定表单提交事件 (在 events.js 中处理)
}