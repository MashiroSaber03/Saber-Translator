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
 * 更新加载状态的消息文本
 * @param {string} message - 新的消息文本
 */
export function updateLoadingMessage(message) {
    // 移除所有info类型的通用消息
    $(".message.info").fadeOut(200, function() { $(this).remove(); });
    // 显示新的消息
    showGeneralMessage(message, "info", false, 0);
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
    const toggleImageButton = $('#toggleImageButton');

    if (dataURL) {
        translatedImageDisplay.attr('src', dataURL).show();
        toggleImageButton.show();
    } else {
        translatedImageDisplay.removeAttr('src').hide();
        toggleImageButton.hide();
    }
    
    // 下载按钮的显示/隐藏现在由 updateButtonStates 函数控制
}

/**
 * 更新缩略图列表
 */
export function renderThumbnails() {
    const thumbnailList = $("#thumbnail-sidebar #thumbnailList");
    thumbnailList.empty();
    state.images.forEach((imageData, index) => {
        const thumbnailItem = $("<div class='thumbnail-item' data-index='" + index + "'></div>");
        const thumbnailImage = $("<img class='thumbnail-image'>").attr('src', imageData.originalDataURL);
        thumbnailItem.append(thumbnailImage);

        // 清除旧标记
        thumbnailItem.find('.translation-failed-indicator, .labeled-indicator').remove();

        if (index === state.currentImageIndex) {
            thumbnailItem.addClass('active');
        }

        // 优先显示失败标记
        if (imageData.translationFailed) {
            thumbnailItem.addClass('translation-failed');
            thumbnailItem.attr('title', '翻译失败，点击可重试');
            thumbnailItem.append('<span class="translation-failed-indicator error-indicator">!</span>'); // 使用特定类名
        }
        // --- 如果未失败，再检查是否有手动标注 ---
        else if (imageData.savedManualCoords && imageData.savedManualCoords.length > 0) {
            thumbnailItem.addClass('has-manual-labels'); // 添加样式类
            thumbnailItem.attr('title', '包含手动标注框');
            thumbnailItem.append('<span class="labeled-indicator">✏️</span>'); // 使用特定类名
        }
        // -------------------------------------

        thumbnailItem.data('index', index);
        thumbnailList.append(thumbnailItem);
    });
    scrollToActiveThumbnail(); // 保持滚动逻辑
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
    const removeAllTextButton = $("#removeAllTextButton");
    const translateAllButton = $("#translateAllButton");
    const clearAllImagesButton = $("#clearAllImagesButton");
    const deleteCurrentImageButton = $("#deleteCurrentImageButton");
    const applyFontSettingsToAllButton = $("#applyFontSettingsToAllButton"); // 在函数内获取
    const downloadButton = $("#downloadButton");
    const downloadAllImagesButton = $("#downloadAllImagesButton");
    const toggleImageButton = $('#toggleImageButton');
    const toggleLabelingModeButton = $("#toggleLabelingModeButton");
    const proofreadButton = $("#proofreadButton"); // 校对按钮
    const proofreadSettingsButton = $("#proofreadSettingsButton"); // 校对设置按钮

    const hasImages = state.images.length > 0;
    const hasCurrentImage = state.currentImageIndex >= 0 && state.currentImageIndex < state.images.length;
    // 检查加载动画是否可见来判断是否在加载状态
    const isLoading = $("#loadingAnimation").is(":visible");

    translateButton.prop('disabled', !hasCurrentImage || isLoading);
    removeTextOnlyButton.prop('disabled', !hasCurrentImage || isLoading);
    removeAllTextButton.prop('disabled', !hasImages || isLoading);
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
    
    // 校对按钮状态更新
    proofreadButton.prop('disabled', !hasImages || isLoading || state.isBatchTranslationInProgress);
    // 校对设置按钮始终保持启用状态，类似于"加载/管理会话"按钮
    proofreadSettingsButton.prop('disabled', false);


    let hasTranslated = false;
    if (hasCurrentImage && state.images[state.currentImageIndex].translatedDataURL) {
        hasTranslated = true;
    }
    
    // 修改：只要有当前图片就显示下载按钮，不再需要已翻译
    downloadButton.toggle(hasCurrentImage && !isLoading);
    // 只有已翻译的图片才显示切换按钮
    toggleImageButton.toggle(hasTranslated && !isLoading);

    // 修改：只要有图片就显示下载所有图片按钮，不再需要已翻译
    downloadAllImagesButton.toggle(hasImages && !isLoading);
    $('#downloadFormat').toggle(hasImages && !isLoading);
    
    // 新增：导出文本和导入文本按钮状态
    $('#exportTextButton').toggle(hasImages && !isLoading);
    $('#importTextButton').toggle(hasImages && !isLoading);

    // 标注模式按钮只在有当前图片且非加载状态时显示
    toggleLabelingModeButton.toggle(hasCurrentImage && !isLoading);
    // 如果处于标注模式，禁用翻译按钮等（除非是"使用手动框翻译"按钮）
    if (state.isLabelingModeActive) {
        translateButton.prop('disabled', true);
        removeTextOnlyButton.prop('disabled', true);
        translateAllButton.prop('disabled', true);
        // ... 可能需要禁用更多按钮 ...
    }

    updateNavigationButtons();
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
    
    // 修改进度条显示逻辑：只要不是完成状态，就显示进度条
    if (percentage < 100) {
        translationProgressBar.show(); // 确保进度条在开始时就显示
    } else {
        setTimeout(() => translationProgressBar.hide(), 1000); // 完成后延迟隐藏
    }
}

/**
 * 显示/隐藏下载状态并禁用/启用按钮
 * @param {boolean} show - 是否显示下载状态 (现在只控制按钮禁用)
 */
export function showDownloadingMessage(show) {
    // 不再控制 #downloadingMessage 的显示/隐藏
    // $("#downloadingMessage").toggle(show);

    // 仍然控制按钮的禁用状态
    $("#downloadButton").prop('disabled', show);
    $("#downloadAllImagesButton").prop('disabled', show);
    // 可能还需要禁用其他在下载时不应操作的按钮
    // $("#translateButton").prop('disabled', show);
    // $("#clearAllImagesButton").prop('disabled', show);
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
    // --- 修改：确保自定义服务商时启用 API Key ---
    const selectedProvider = $('#modelProvider').val();
    if (selectedProvider === 'custom_openai') { // 使用常量会更好，但为了简单这里用字符串
        disabled = false; // 自定义服务商总是需要 API Key
        placeholder = '请输入 API Key (自定义服务)';
    }
    // ----------------------------------------
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
 * 显示或隐藏彩云小译相关 UI 元素
 * @param {boolean} show - 是否显示
 */
export function toggleCaiyunUI(show) {
    // 为彩云小译设置一个提示文本
    if (show) {
        $('#modelName').attr('placeholder', '可选填：自动/日语/英语等').val('auto');
        // 移除其他服务商的模型列表
        $('#ollamaModelsList').empty().hide();
        $('#sakuraModelsList').empty().hide();
    }
}

/**
 * 显示或隐藏百度翻译相关 UI 元素
 * @param {boolean} show - 是否显示
 */
export function toggleBaiduTranslateUI(show) {
    if (show) {
        // 将API Key输入框的标签和placeholder改为百度翻译对应的内容
        $('label[for="apiKey"]').text('App ID:');
        $('#apiKey').attr('placeholder', '请输入百度翻译App ID');
        
        // 将模型输入框的标签和placeholder改为百度翻译对应的内容
        $('label[for="modelName"]').text('App Key:');
        $('#modelName').attr('placeholder', '请输入百度翻译App Key');
        
        // 添加测试连接按钮，如果不存在
        if ($('#testBaiduTranslateButton').length === 0) {
            const testButton = $('<button id="testBaiduTranslateButton" class="test-connection-btn">测试连接</button>');
            $('#modelName').after(testButton);
            
            // 绑定测试按钮点击事件
            testButton.on('click', function(e) {
                e.preventDefault();
                testBaiduTranslateConnection();
            });
        } else {
            $('#testBaiduTranslateButton').show();
        }
        
        // 移除其他服务商的模型列表
        $('#ollamaModelsList').empty().hide();
        $('#sakuraModelsList').empty().hide();
    } else {
        // 恢复默认标签和placeholder
        $('label[for="apiKey"]').text('API Key:');
        $('#apiKey').attr('placeholder', '请输入API Key');
        
        $('label[for="modelName"]').text('大模型型号:');
        $('#modelName').attr('placeholder', '请输入模型型号');
        
        // 隐藏测试按钮
        $('#testBaiduTranslateButton').hide();
    }
}

/**
 * 测试百度翻译API连接
 */
function testBaiduTranslateConnection() {
    const appId = $('#apiKey').val().trim();
    const appKey = $('#modelName').val().trim();
    
    if (!appId || !appKey) {
        showGeneralMessage('请填写App ID和App Key', 'warning');
        return;
    }
    
    // 显示加载指示器
    showLoadingMessage('正在测试百度翻译API连接...');
    
    // 发送测试请求
    $.ajax({
        url: '/test_baidu_translate_connection',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            app_id: appId,
            app_key: appKey
        }),
        success: function(response) {
            if (response.success) {
                showGeneralMessage(response.message, 'success');
            } else {
                showGeneralMessage(response.message, 'error');
            }
        },
        error: function(xhr, status, error) {
            let errorMsg = '测试连接失败';
            try {
                const response = JSON.parse(xhr.responseText);
                errorMsg = response.message || errorMsg;
            } catch (e) {
                console.error('解析错误响应失败', e);
            }
            showGeneralMessage(errorMsg, 'error');
        },
        complete: function() {
            hideLoadingMessage();
        }
    });
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
 * @param {boolean} showLamaOptions - 是否显示 LAMA 选项（现在不再有 MI-GAN 选项）
 * @param {boolean} showSolidOptions - 是否显示纯色填充选项
 */
export function toggleInpaintingOptions(showLamaOptions, showSolidOptions) {
    const inpaintingOptionsDiv = $("#inpaintingOptions"); // 在函数内获取
    const solidColorOptionsDiv = $("#solidColorOptions"); // 在函数内获取
    
    // 如果 LAMA 有独立的选项（比如强度、融合），在这里控制显示
    if (showLamaOptions) {
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
 * @param {string} [messageId=''] - 消息唯一标识符，用于后续清除特定消息
 * @returns {string} 消息ID，如果未提供则自动生成
 */
export function showGeneralMessage(message, type = 'info', isHTML = false, duration = 5000, messageId = '') {
    let messageContainer = $('#messageContainer');
    if (messageContainer.length === 0) {
        messageContainer = $('<div id="messageContainer" class="message-container"></div>');
        $('body').append(messageContainer);
    }
    
    // 生成唯一消息ID或使用提供的ID
    const msgId = messageId || 'msg_' + Date.now() + '_' + Math.floor(Math.random() * 1000);
    
    // 如果使用指定ID，先移除具有相同ID的现有消息
    if (messageId) {
        $(`.message[data-msg-id="${messageId}"]`).remove();
    }
    
    const messageElement = $('<div class="message"></div>').addClass(type);
    messageElement.attr('data-msg-id', msgId);
    
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
    
    // 添加自动超时安全机制，即使是无限消息，也在30秒后自动消失
    const safetyTimeout = Math.max(duration, 30000); 
    
    // 设置定时消失
    if (duration > 0 || safetyTimeout > 0) {
        setTimeout(function() {
            messageElement.fadeOut(300, function() { $(this).remove(); });
        }, duration > 0 ? duration : safetyTimeout);
    }
    
    return msgId;
}

/**
 * 清除指定ID的消息
 * @param {string} messageId - 要清除的消息ID
 */
export function clearGeneralMessageById(messageId) {
    if (!messageId) return;
    
    $(`.message[data-msg-id="${messageId}"]`).fadeOut(300, function() { 
        $(this).remove(); 
    });
}

/**
 * 清除所有特定类型的消息
 * @param {'info' | 'success' | 'warning' | 'error' | ''} [type=''] - 消息类型，空字符串表示清除所有类型
 */
export function clearAllGeneralMessages(type = '') {
    const selector = type ? `.message.${type}` : '.message';
    $(selector).fadeOut(300, function() { 
        $(this).remove(); 
    });
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
    if (!currentImage) {
        bubbleCount.text("0");
        bubbleList.append($("<div>").addClass("no-bubbles-message").text("当前无图片"));
        return;
    }
    
    if (!currentImage.bubbleCoords || currentImage.bubbleCoords.length === 0) {
        bubbleCount.text("0");
        bubbleList.append($("<div>").addClass("no-bubbles-message").text("当前图片没有气泡"));
        bubbleList.append($("<div>").addClass("no-bubbles-hint").text("您可以使用标注模式添加气泡"));
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
    const bubbleFillColorInput = $("#bubbleFillColor"); // 新的颜色选择器
    // === 新增：获取描边控件 START ===
    const bubbleEnableStroke = $("#bubbleEnableStroke");
    const bubbleStrokeColor = $("#bubbleStrokeColor");
    const bubbleStrokeWidth = $("#bubbleStrokeWidth");
    const bubbleStrokeOptions = $(".bubble-stroke-options");
    // === 新增：获取描边控件 END ===

    if (index < 0 || index >= state.bubbleSettings.length) {
        // 清空编辑区
        currentBubbleIndexDisplay.text("-");
        bubbleTextEditor.val("");
        autoBubbleFontSizeCheckbox.prop('checked', false);
        bubbleFontSize.prop('disabled', false).val(state.defaultFontSize);
        bubbleFontFamily.val(state.defaultFontFamily);
        bubbleTextDirection.val(state.defaultLayoutDirection);
        bubbleTextColor.val(state.defaultTextColor);
        bubbleFillColorInput.val(state.getCurrentImage()?.fillColor || state.defaultFillColor); // 清空时设为当前图片的全局填充色或默认
        bubbleRotationAngle.val(0);
        bubbleRotationAngleValue.text('0°');
        positionOffsetX.val(0);
        positionOffsetY.val(0);
        positionOffsetXValue.text(0);
        positionOffsetYValue.text(0);
        // === 新增：重置描边控件 START ===
        bubbleEnableStroke.prop('checked', state.enableTextStroke);
        bubbleStrokeColor.val(state.textStrokeColor);
        bubbleStrokeWidth.val(state.textStrokeWidth);
        bubbleStrokeOptions.toggle(state.enableTextStroke);
        // === 新增：重置描边控件 END ===
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
    bubbleFillColorInput.val(setting.fillColor || state.getCurrentImage()?.fillColor || state.defaultFillColor);
    bubbleRotationAngle.val(setting.rotationAngle || 0);
    bubbleRotationAngleValue.text((setting.rotationAngle || 0) + '°');

    const position = setting.position || { x: 0, y: 0 };
    positionOffsetX.val(position.x);
    positionOffsetY.val(position.y);
    positionOffsetXValue.text(position.x);
    positionOffsetYValue.text(position.y);
    
    // === 新增：更新描边控件 START ===
    const enableStroke = setting.enableStroke !== undefined ? setting.enableStroke : state.enableTextStroke;
    bubbleEnableStroke.prop('checked', enableStroke);
    bubbleStrokeColor.val(setting.strokeColor || state.textStrokeColor);
    bubbleStrokeWidth.val(setting.strokeWidth !== undefined ? setting.strokeWidth : state.textStrokeWidth);
    bubbleStrokeOptions.toggle(enableStroke);
    // === 新增：更新描边控件 END ===
}

/**
 * 添加或更新气泡高亮效果
 * @param {number} selectedBubbleIndex - 当前选中的气泡索引, -1 表示没有选中
 */
export function updateBubbleHighlight(selectedBubbleIndex) {
    // 移除旧的高亮框
    $('.highlight-bubble').remove();

    // 如果不在编辑模式，不显示高亮框
    if (!state.editModeActive) return;

    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.bubbleCoords || currentImage.bubbleCoords.length === 0) return;

    // 获取图片元素和容器
    const imageElement = $('#translatedImageDisplay');
    const imageContainer = $('.image-container');

    // 使用新的辅助函数获取精确的显示指标
    const metrics = calculateImageDisplayMetrics(imageElement);
    if (!metrics) {
        imageElement.off('load.updateHighlight').one('load.updateHighlight', () => updateBubbleHighlight(selectedBubbleIndex));
        console.warn("updateBubbleHighlight: 图片指标无效，等待加载后重试。");
        return;
    }

    // 遍历所有气泡坐标并创建高亮框
    currentImage.bubbleCoords.forEach((coords, index) => {
        const [x1, y1, x2, y2] = coords;
        const isSelected = (index === selectedBubbleIndex);
        
        // 创建高亮元素
        const highlightElement = $('<div class="highlight-bubble"></div>');
        if (isSelected) {
            highlightElement.addClass('selected');
        }
        
        // 使用 metrics 中的精确值进行转换
        highlightElement.css({
            'left': `${metrics.visualContentOffsetX + x1 * metrics.scaleX}px`,
            'top': `${metrics.visualContentOffsetY + y1 * metrics.scaleY}px`,
            'width': `${(x2 - x1) * metrics.scaleX}px`,
            'height': `${(y2 - y1) * metrics.scaleY}px`
        });
        
        // 添加气泡索引数据属性，用于点击事件
        highlightElement.attr('data-bubble-index', index);
        
        // 添加到容器
        imageContainer.append(highlightElement);
    });
    
    // 为高亮框添加点击事件，用于选择气泡
    $('.highlight-bubble').on('click', function(e) {
        e.preventDefault();
        e.stopPropagation(); // 阻止冒泡，避免触发图片点击事件
        
        const bubbleIndex = parseInt($(this).attr('data-bubble-index'));
        // 导入edit_mode模块并调用selectBubble
        import('./edit_mode.js').then(editMode => {
            editMode.selectBubble(bubbleIndex);
        });
    });
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
        
        // 初始显示所有气泡高亮
        if (state.selectedBubbleIndex >= 0) {
            updateBubbleHighlight(state.selectedBubbleIndex);
        }
        
        // 添加窗口大小改变事件，重新计算高亮位置
        $(window).on('resize.bubbleHighlight', function() {
            updateBubbleHighlight(state.selectedBubbleIndex);
        });
    } else {
        toggleEditModeButton.text("切换编辑模式").removeClass("active");
        editModeContainer.hide();
        detectedTextInfo.show();
        $('.highlight-bubble').remove(); // 移除所有高亮框
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
 * @param {object} defaultStates - 插件默认启用状态字典 { pluginName: boolean }
 */
export function renderPluginList(plugins, defaultStates = {}) {
    const container = $("#pluginListContainer");
    container.empty();

    // 添加QQ群信息
    const groupInfo = $('<div class="plugin-group-info" style="margin-bottom: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 5px;"></div>');
    groupInfo.append('<p style="margin: 0;"><strong>🎉 欢迎插件开发者加入QQ群：1041505784</strong></p>');
    groupInfo.append('<p style="margin: 5px 0 0 0; color: #666;">在这里分享你制作的插件，与其他开发者交流经验！</p>');
    container.append(groupInfo);

    if (!plugins || plugins.length === 0) {
        container.append("<p>未找到任何插件。</p>");
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

        const controls = $('<div class="plugin-controls" style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;"></div>'); // 新增容器

        // --- 新增: 默认启用状态控制 ---
        const defaultEnableLabel = $('<label class="plugin-default-toggle-label" style="font-size: 0.9em; display: flex; align-items: center; gap: 5px;"></label>');
        const defaultEnableCheckbox = $(`<input type="checkbox" class="plugin-default-toggle">`);
        defaultEnableCheckbox.attr('data-plugin-name', plugin.name); // 关联插件名
        // 设置初始状态
        const isDefaultEnabled = defaultStates[plugin.name] === true; // 从传入的参数获取
        defaultEnableCheckbox.prop('checked', isDefaultEnabled);
        defaultEnableLabel.append(defaultEnableCheckbox);
        defaultEnableLabel.append('默认启用');
        controls.append(defaultEnableLabel); // 添加到 controls 容器
        // ------------------------------

        const actions = $('<div class="plugin-actions" style="display: flex; align-items: center; gap: 10px;"></div>'); // 放到右侧

        // 实时启用/禁用开关
        const toggleLabel = $('<label class="plugin-toggle"></label>');
        const toggleCheckbox = $('<input type="checkbox" class="plugin-enable-toggle">');
        toggleCheckbox.prop('checked', plugin.enabled); // 当前实时状态
        toggleCheckbox.attr('data-plugin-name', plugin.name); // 关联插件名
        toggleLabel.append(toggleCheckbox);
        toggleLabel.append(plugin.enabled ? ' 已启用' : ' 已禁用');
        actions.append(toggleLabel);

        // 设置按钮
        const settingsButton = $('<button class="plugin-settings-button">设置</button>');
        settingsButton.attr('data-plugin-name', plugin.name); // 关联插件名
        actions.append(settingsButton);

        // 删除按钮
        const deleteButton = $('<button class="plugin-delete-button">删除</button>');
        deleteButton.attr('data-plugin-name', plugin.name); // 关联插件名
        actions.append(deleteButton);

        controls.append(actions); // 将 actions 添加到 controls 容器右侧
        pluginDiv.append(controls); // 将 controls 添加到插件项
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

/**
 * 在图片上绘制边界框
 * @param {Array<Array<number>>} coords - 坐标列表 [[x1, y1, x2, y2], ...]
 * @param {number} [selectedIndex=-1] - 当前选中的框的索引，用于高亮和显示手柄
 */
export function drawBoundingBoxes(coords, selectedIndex = -1) {
    clearBoundingBoxes();

    const imageElement = $('#translatedImageDisplay');
    const imageContainer = $('.image-container');

    if (imageElement.length === 0 || imageContainer.length === 0) return;

    // 使用新的辅助函数获取精确的显示指标
    const metrics = calculateImageDisplayMetrics(imageElement); // 修改点
    if (!metrics) {
        // 如果图片未加载或指标无效，尝试在图片加载后重新绘制
        imageElement.off('load.drawBoxes').one('load.drawBoxes', () => drawBoundingBoxes(coords, selectedIndex));
        console.warn("drawBoundingBoxes: 图片指标无效，等待加载后重试。");
        return;
    }

    if (!coords || coords.length === 0) return;

    coords.forEach((coord, index) => {
        const [x1, y1, x2, y2] = coord;
        if (x1 === undefined || y1 === undefined || x2 === undefined || y2 === undefined || x1 >= x2 || y1 >= y2) {
            console.warn(`drawBoundingBoxes: 跳过无效坐标: Box ${index}`, coord);
            return;
        }

        // 使用 metrics 中的精确值进行转换
        const leftPos = metrics.visualContentOffsetX + x1 * metrics.scaleX;
        const topPos = metrics.visualContentOffsetY + y1 * metrics.scaleY;
        const widthPos = (x2 - x1) * metrics.scaleX;
        const heightPos = (y2 - y1) * metrics.scaleY;

        const boxElement = $('<div class="manual-bounding-box draggable-box"></div>');
        boxElement.attr('data-index', index);
        boxElement.css({
            position: 'absolute',
            left: `${leftPos}px`,
            top: `${topPos}px`,
            width: `${widthPos}px`,
            height: `${heightPos}px`,
            border: '2px solid rgba(255, 0, 0, 0.7)',
            boxSizing: 'border-box',
            pointerEvents: 'auto',
            cursor: 'grab',
            zIndex: 10,
            backgroundColor: 'rgba(255, 0, 0, 0.05)'
        });

        if (index === selectedIndex) {
            boxElement.addClass('selected');
            boxElement.css({
                border: '3px solid rgba(0, 255, 0, 0.9)',
                backgroundColor: 'rgba(0, 255, 0, 0.1)',
                cursor: 'grabbing', // 或者 'move'
                boxShadow: '0 0 5px rgba(0, 255, 0, 0.7)'
            });

            const handles = ['top-left', 'top-right', 'bottom-left', 'bottom-right'];
            handles.forEach(handleType => {
                const handle = $('<div class="resize-handle"></div>');
                handle.addClass(handleType);
                handle.attr('data-handle', handleType);
                handle.attr('data-parent-index', index);
                boxElement.append(handle);
            });
        }
        imageContainer.append(boxElement);
    });
}

/**
 * 清除所有绘制的边界框
 */
export function clearBoundingBoxes() {
    $('.manual-bounding-box').remove();
    console.log("清除了所有标注框");
}

/**
 * 切换标注模式的 UI 显示
 * @param {boolean} isActive - 标注模式是否激活
 */
export function toggleLabelingModeUI(isActive) {
    const toggleButton = $("#toggleLabelingModeButton");
    const editModeButton = $("#toggleEditModeButton");
    const labelingToolButtons = $(".labeling-tool-button"); // 获取所有标注工具按钮
    const body = $('body');

    // --- 获取需要禁用的操作按钮 ---
    const translateButton = $("#translateButton");
    const removeTextOnlyButton = $("#removeTextOnlyButton");
    const translateAllButton = $("#translateAllButton");
    const applyFontSettingsToAllButton = $("#applyFontSettingsToAllButton"); // 如果存在全局应用按钮

    if (isActive) {
        toggleButton.text("退出标注模式").addClass("active");
        editModeButton.prop('disabled', true); // 禁用编辑模式切换按钮
        body.addClass('labeling-mode-active');
        labelingToolButtons.show(); // 显示所有标注工具按钮

        // --- 明确禁用冲突的操作按钮 ---
        translateButton.prop('disabled', true);
        removeTextOnlyButton.prop('disabled', true);
        translateAllButton.prop('disabled', true);
        if (applyFontSettingsToAllButton && applyFontSettingsToAllButton.length) {
             applyFontSettingsToAllButton.prop('disabled', true); // 禁用全局应用设置
        }
        // -------------------------------

        // --- 确保设置面板保持可用 (移除之前的禁用逻辑) ---
        // $('#settings-sidebar .settings-card .collapsible-content').css({ 'opacity': '1', 'pointer-events': 'auto' }); // 如果之前有禁用样式，恢复它
        // $('#settings-sidebar').css({ 'opacity': '1', 'pointer-events': 'auto' }); // 如果之前有禁用样式，恢复它
        // ------------------------------------------------

        // 根据状态启用/禁用标注工具按钮
        const hasCoords = state.manualBubbleCoords && state.manualBubbleCoords.length > 0;
        $('#clearManualBoxesButton').prop('disabled', !hasCoords);
        $('#useManualBoxesButton').prop('disabled', !hasCoords);
        $('#deleteSelectedBoxButton').prop('disabled', state.selectedManualBoxIndex === -1);
        // 启用"检测所有图片"按钮，但仅当有多张图片时
        $('#detectAllImagesButton').prop('disabled', state.images.length <= 1);

        // 设置图片容器光标为十字准星 (移到 labeling_mode.js 中处理可能更好，但这里也放一份确保生效)
        $('.image-container').css('cursor', 'crosshair');

        console.log("UI 更新：进入标注模式，显示工具按钮，设置面板保持可用");

    } else {
        toggleButton.text("进入标注模式").removeClass("active");
        // 恢复编辑模式按钮状态 (如果当前有图片)
        editModeButton.prop('disabled', state.currentImageIndex === -1);
        body.removeClass('labeling-mode-active');
        labelingToolButtons.hide(); // 隐藏所有标注工具按钮

        // --- 恢复操作按钮状态 (应由 updateButtonStates 统一管理) ---
        // 不需要在这里单独启用，updateButtonStates 会处理
        // translateButton.prop('disabled', state.currentImageIndex === -1);
        // removeTextOnlyButton.prop('disabled', state.currentImageIndex === -1);
        // translateAllButton.prop('disabled', state.images.length === 0);
         if (applyFontSettingsToAllButton && applyFontSettingsToAllButton.length) {
             // applyFontSettingsToAllButton.prop('disabled', state.images.length === 0);
         }
        // 调用 updateButtonStates 来正确设置按钮状态
        updateButtonStates();
        // ------------------------------------------------------

        clearBoundingBoxes(); // 退出时清除绘制的框

        // 恢复图片容器默认光标
        $('.image-container').css('cursor', 'default');

        console.log("UI 更新：退出标注模式，隐藏工具按钮");
    }
    // 确保标注模式切换按钮本身在没有图片时不显示
    toggleButton.toggle(state.currentImageIndex !== -1);
}

/**
 * 渲染会话列表到模态框中。
 * @param {Array<object>} sessions - 从后端获取的会话信息数组。
 */
export function renderSessionList(sessions) {
    const container = $("#sessionListContainer");
    container.empty();

    if (!sessions || sessions.length === 0) {
        container.append("<p>没有找到已保存的会话。</p>");
        return;
    }

    let autoSaveData = null;
    const regularSessions = [];

    // --- 新增：分离自动存档和常规存档 ---
    sessions.forEach(session => {
        if (session.name === constants.AUTO_SAVE_SLOT_NAME) {
            autoSaveData = session;
        } else {
            regularSessions.push(session);
        }
    });
    // ---------------------------------

    // --- 新增：渲染自动存档槽位 ---
    if (autoSaveData) {
        const itemDiv = $('<div class="session-item autosave-item"></div>'); // 添加特殊类名
        itemDiv.attr('data-session-name', autoSaveData.name); // 仍然使用内部名称

        const infoDiv = $('<div class="session-info"></div>');
        // 使用常量中的显示名称
        infoDiv.append(`<span class="session-name" style="color: #007bff; font-weight: bold;"><i class="fas fa-save"></i> ${constants.AUTO_SAVE_DISPLAY_NAME}</span>`); // 添加图标和样式
        infoDiv.append(`<span class="session-details">最后更新: ${autoSaveData.saved_at || '未知'} | 图片: ${autoSaveData.image_count || 0}</span>`);

        const actionsDiv = $('<div class="session-actions"></div>');
        // 只添加加载按钮
        const loadButton = $('<button class="session-load-button">加载</button>');
        actionsDiv.append(loadButton);

        itemDiv.append(infoDiv);
        itemDiv.append(actionsDiv);
        container.append(itemDiv); // 添加到容器顶部

        // 添加分隔线
        container.append('<hr style="border-top: 1px solid var(--border-color); margin: 10px 0;">');
    } else {
        // 如果没有自动存档数据，可以显示一个提示
         const noAutoSaveDiv = $('<div class="session-item autosave-item disabled-item"></div>');
         noAutoSaveDiv.append(`<span style="color: #888; font-style: italic;">尚未创建自动存档</span>`);
         container.append(noAutoSaveDiv);
         container.append('<hr style="border-top: 1px solid var(--border-color); margin: 10px 0;">');
    }
    // ---------------------------

    // --- 渲染常规会话列表 (过滤掉自动存档) ---
    if (regularSessions.length === 0 && !autoSaveData) {
        // 如果常规和自动都没有，显示原始提示
        container.empty().append("<p>没有找到已保存的会话。</p>");
        return;
    } else if (regularSessions.length === 0 && autoSaveData) {
        // 只有自动存档，不显示 "没有找到常规会话"
    } else {
        // 渲染常规会话
        regularSessions.forEach(session => {
            const itemDiv = $('<div class="session-item"></div>');
            itemDiv.attr('data-session-name', session.name);

            const infoDiv = $('<div class="session-info"></div>');
            infoDiv.append(`<span class="session-name">${session.name}</span>`);
            infoDiv.append(`<span class="session-details">保存于: ${session.saved_at || '未知'} | 图片: ${session.image_count || 0} | 版本: ${session.version || '未知版本'}</span>`);

            const actionsDiv = $('<div class="session-actions"></div>');
            const loadButton = $('<button class="session-load-button">加载</button>');
            const renameButton = $('<button class="session-rename-button" style="background-color: #ffc107; color: #333;">重命名</button>');
            const deleteButton = $('<button class="session-delete-button">删除</button>');

            actionsDiv.append(loadButton);
            actionsDiv.append(renameButton);
            actionsDiv.append(deleteButton);

            itemDiv.append(infoDiv);
            itemDiv.append(actionsDiv);
            container.append(itemDiv);
        });
    }
    // ---------------------------------
}

/**
 * 更新自动存档开关的状态
 */
export function updateAutoSaveToggle() {
    const isEnabled = state.getAutoSaveEnabled();
    $('#autoSaveToggle').prop('checked', isEnabled);
}

/**
 * 显示会话管理模态框。
 */
export function showSessionManagerModal() {
    const modal = $("#sessionManagerModal");
    modal.css("display", "block");
    // 更新自动存档开关状态
    updateAutoSaveToggle();
    // 可选：在显示时先显示加载提示，由 session.js 获取数据后再调用 renderSessionList
    // $("#sessionListContainer").html("<p>正在加载会话列表...</p>");
}

/**
 * 隐藏会话管理模态框。
 */
export function hideSessionManagerModal() {
    const modal = $("#sessionManagerModal");
    modal.css("display", "none");
}

/**
 * 计算图像内容在其 <img> 元素中的实际显示指标。
 * 考虑到 object-fit: contain 的影响。
 *
 * @param {jQuery} imagejQueryElement - 图像的 jQuery 元素 (例如 $('#translatedImageDisplay'))
 * @returns {object|null} 包含以下属性的对象，如果图片未加载或无尺寸则返回 null:
 *   - visualContentWidth (number): 图像内容在屏幕上的实际渲染宽度
 *   - visualContentHeight (number): 图像内容在屏幕上的实际渲染高度
 *   - visualContentOffsetX (number): 图像内容左上角相对于其 offsetParent (通常是 .image-container) 的X轴偏移
 *   - visualContentOffsetY (number): 图像内容左上角相对于其 offsetParent (通常是 .image-container) 的Y轴偏移
 *   - scaleX (number): 水平缩放比例 (visualContentWidth / naturalWidth)
 *   - scaleY (number): 垂直缩放比例 (visualContentHeight / naturalHeight)
 *   - naturalWidth (number): 图像的原始宽度
 *   - naturalHeight (number): 图像的原始高度
 *   - elementWidth (number): <img> 元素本身的宽度
 *   - elementHeight (number): <img> 元素本身的高度
 */
export function calculateImageDisplayMetrics(imagejQueryElement) {
    if (!imagejQueryElement || imagejQueryElement.length === 0) {
        console.error("calculateImageDisplayMetrics: 提供的图像元素无效。");
        return null;
    }
    const imageNative = imagejQueryElement[0];

    if (!imageNative.complete || imageNative.naturalWidth === 0 || imageNative.naturalHeight === 0) {
        console.warn("calculateImageDisplayMetrics: 图像未完全加载或尺寸为0。");
        return null;
    }

    const naturalWidth = imageNative.naturalWidth;
    const naturalHeight = imageNative.naturalHeight;

    // <img> 元素在屏幕上的实际渲染尺寸
    const elementWidth = imagejQueryElement.width();
    const elementHeight = imagejQueryElement.height();

    let visualContentWidth, visualContentHeight;
    const naturalAspectRatio = naturalWidth / naturalHeight;
    const elementAspectRatio = elementWidth / elementHeight;

    if (naturalAspectRatio > elementAspectRatio) {
        // 图片比元素框更"宽" (相对于元素框的比例)，所以图片的宽度会填满元素框，高度按比例缩放
        // 这会导致上下留白 (letterboxed)
        visualContentWidth = elementWidth;
        visualContentHeight = elementWidth / naturalAspectRatio;
    } else {
        // 图片比元素框更"高"，所以图片的高度会填满元素框，宽度按比例缩放
        // 这会导致左右留白 (pillarboxed)
        visualContentHeight = elementHeight;
        visualContentWidth = elementHeight * naturalAspectRatio;
    }

    // 图像内容在其元素框内的偏移 (由于 object-fit: contain，内容会居中)
    const offsetXInsideElement = (elementWidth - visualContentWidth) / 2;
    const offsetYInsideElement = (elementHeight - visualContentHeight) / 2;

    // <img> 元素本身相对于其 offsetParent 的偏移。
    // 对于绝对定位的子元素（如标注框），其 left/top 是相对于其最近的具有 position:relative/absolute/fixed 的祖先元素的内边距边缘。
    // 在我们的例子中，.image-container 有 position:relative，标注框是它的子元素。
    // imageNative.offsetLeft/Top 就是 <img> 相对于 .image-container 内边距边缘的偏移。
    const elementOffsetX = imageNative.offsetLeft;
    const elementOffsetY = imageNative.offsetTop;

    // 最终，图像内容左上角相对于 .image-container 的偏移
    const finalVisualContentOffsetX = elementOffsetX + offsetXInsideElement;
    const finalVisualContentOffsetY = elementOffsetY + offsetYInsideElement;

    const finalScaleX = naturalWidth > 0 ? visualContentWidth / naturalWidth : 0;
    const finalScaleY = naturalHeight > 0 ? visualContentHeight / naturalHeight : 0;

    return {
        visualContentWidth,
        visualContentHeight,
        visualContentOffsetX: finalVisualContentOffsetX,
        visualContentOffsetY: finalVisualContentOffsetY,
        scaleX: finalScaleX,
        scaleY: finalScaleY,
        naturalWidth,
        naturalHeight,
        elementWidth, // 方便调试
        elementHeight // 方便调试
    };
}

/**
 * 设置 AI 视觉 OCR 提示词文本框的值。
 * @param {string} prompt - 要设置的提示词。
 */
export function setAiVisionOcrPrompt(prompt) {
    $('#aiVisionOcrPrompt').val(prompt);
}

/**
 * 显示或隐藏有道翻译相关 UI 元素
 * @param {boolean} show - 是否显示
 */
export function toggleYoudaoTranslateUI(show) {
    if (show) {
        // 将API Key输入框的标签和placeholder改为有道翻译对应的内容
        $('label[for="apiKey"]').text('App Key:');
        $('#apiKey').attr('placeholder', '请输入有道翻译应用ID');
        
        // 将模型输入框的标签和placeholder改为有道翻译对应的内容
        $('label[for="modelName"]').text('App Secret:');
        $('#modelName').attr('placeholder', '请输入有道翻译应用密钥');
        
        // 添加测试连接按钮，如果不存在
        if ($('#testYoudaoTranslateButton').length === 0) {
            const testButton = $('<button id="testYoudaoTranslateButton" class="test-connection-btn">测试连接</button>');
            $('#modelName').after(testButton);
            
            // 绑定测试按钮点击事件
            testButton.on('click', function(e) {
                e.preventDefault();
                testYoudaoTranslateConnection();
            });
        } else {
            $('#testYoudaoTranslateButton').show();
        }
        
        // 移除其他服务商的模型列表
        $('#ollamaModelsList').empty().hide();
        $('#sakuraModelsList').empty().hide();
    } else {
        // 隐藏测试按钮
        $('#testYoudaoTranslateButton').hide();
    }
}

/**
 * 测试有道翻译API连接
 */
function testYoudaoTranslateConnection() {
    const appKey = $('#apiKey').val().trim();
    const appSecret = $('#modelName').val().trim();
    
    if (!appKey || !appSecret) {
        showGeneralMessage('请填写App Key和App Secret', 'warning');
        return;
    }
    
    // 显示加载指示器
    showLoadingMessage('正在测试有道翻译API连接...');
    
    // 发送测试请求
    $.ajax({
        url: '/test_youdao_translate',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            appKey: appKey,
            appSecret: appSecret
        }),
        success: function(response) {
            if (response.success) {
                showGeneralMessage(response.message, 'success');
            } else {
                showGeneralMessage(response.message, 'error');
            }
        },
        error: function(xhr, status, error) {
            let errorMsg = '测试连接失败';
            try {
                const response = JSON.parse(xhr.responseText);
                errorMsg = response.message || errorMsg;
            } catch (e) {
                console.error('解析错误响应失败', e);
            }
            showGeneralMessage(errorMsg, 'error');
        },
        complete: function() {
            hideLoadingMessage();
        }
    });
}

/**
 * 更新漫画翻译提示词区域的UI
 */
export function updateTranslatePromptUI() {
    $('#promptContent').val(state.currentPromptContent);
    // 更新选择器的选中值
    if (state.isTranslateJsonMode) {
        $('#translatePromptModeSelect').val('json');
    } else {
        $('#translatePromptModeSelect').val('normal');
    }
}

/**
 * 更新AI视觉OCR提示词区域的UI
 */
export function updateAiVisionOcrPromptUI() {
    $('#aiVisionOcrPrompt').val(state.aiVisionOcrPrompt);
    // 更新选择器的选中值
    if (state.isAiVisionOcrJsonMode) {
        $('#aiVisionPromptModeSelect').val('json');
    } else {
        $('#aiVisionPromptModeSelect').val('normal');
    }
}

/**
 * 显示或隐藏自定义 OpenAI 兼容服务相关的 UI 元素 (Base URL 输入框)
 * @param {boolean} show - 是否显示
 */
export function toggleCustomOpenAiUI(show) {
    const customBaseUrlDiv = $("#customBaseUrlDiv");
    if (show) {
        customBaseUrlDiv.slideDown();
    } else {
        customBaseUrlDiv.slideUp();
        // $('#customBaseUrl').val(''); // 可选：隐藏时清空内容
    }
}

/**
 * 更新rpm输入框的显示值
 */
export function updaterpmInputFields() {
    $('#rpmTranslation').val(state.rpmLimitTranslation);
    $('#rpmAiVisionOcr').val(state.rpmLimitAiVisionOcr);
    console.log("UI 更新: rpm输入框已更新为当前状态值。");
}

/**
 * 显示或隐藏自定义 AI 视觉 OCR 服务的 Base URL 输入框
 * @param {boolean} show - 是否显示
 */
export function toggleCustomAiVisionBaseUrlUI(show) {
    const customBaseUrlDiv = $("#customAiVisionBaseUrlDiv"); // 获取元素
    if (show) {
        customBaseUrlDiv.slideDown(); // 使用 jQuery 的 slideDown 动画显示
    } else {
        customBaseUrlDiv.slideUp();   // 使用 jQuery 的 slideUp 动画隐藏
        // 可选：当隐藏时清空输入框内容
        // $('#customAiVisionBaseUrl').val('');
    }
    console.log(`UI 更新: 自定义AI视觉Base URL输入框已 ${show ? '显示' : '隐藏'}`);
}

/**
 * 加载并渲染字体列表
 * @param {String} selectedFont - 当前选中的字体路径
 * @param {Boolean} updateBubbleFontFamily - 是否同时更新编辑模式字体选择器
 */
export function loadFontList(selectedFont, updateBubbleFontFamily = true) {
    import('./api.js').then(api => {
        api.getFontListApi(
            response => {
                // 更新主界面字体选择器
                updateFontSelector($('#fontFamily'), response, selectedFont);
                
                // 同时更新编辑模式的字体选择器
                if (updateBubbleFontFamily) {
                    updateFontSelector($('#bubbleFontFamily'), response, selectedFont);
                }
                
                console.log("字体列表加载完成，当前选中字体:", $('#fontFamily').val());
            },
            error => {
                console.error('加载字体列表失败:', error);
                showGeneralMessage('加载字体列表失败，将使用默认字体', 'error', 'font-list-error');
            }
        );
    });
}

/**
 * 更新字体选择器的选项
 * @param {jQuery} selector - 字体选择器jQuery对象
 * @param {Object} response - 字体列表API响应
 * @param {String} selectedFont - 当前选中的字体路径
 */
function updateFontSelector(selector, response, selectedFont) {
    // 清空除了"自定义字体"选项外的所有选项
    selector.find('option:not([data-custom])').remove();
    
    // 添加字体选项
    response.fonts.forEach(font => {
        const fontClass = font.display_name.replace(/\s+/g, '').toLowerCase();
        const option = $('<option>')
            .val(font.path)
            .text(font.display_name)
            .attr('style', `font-family: '${fontClass}', ${getGenericFontFamily(font.display_name)};`);
        
        // 设置选中状态
        if (selectedFont && selectedFont === font.path) {
            option.prop('selected', true);
        }
        
        // 微软雅黑作为备选默认字体（如果没有指定字体或找不到指定的字体）
        if ((!selectedFont || selector.val() === 'custom-font') && font.path === 'fonts/msyh.ttc') {
            option.prop('selected', true);
        }
        
        // 将选项添加到选择器
        selector.append(option);
    });
    
    // 如果仍然没有选中任何字体（选择器值仍为custom-font），则选择第一个实际字体
    if (selector.val() === 'custom-font') {
        selector.find('option:not([data-custom]):first').prop('selected', true);
    }
}

/**
 * 根据字体名称推断通用字体族
 * @param {String} fontName - 字体名称
 * @returns {String} - 通用字体族
 */
function getGenericFontFamily(fontName) {
    const lowerFontName = fontName.toLowerCase();
    
    // 检查是否包含关键字来决定字体族
    if (lowerFontName.includes('黑体') || lowerFontName.includes('雅黑')) {
        return 'sans-serif';
    } else if (lowerFontName.includes('宋体') || lowerFontName.includes('楷体') || 
              lowerFontName.includes('仿宋') || lowerFontName.includes('隶书')) {
        return 'serif';
    } else if (lowerFontName.includes('行楷') || lowerFontName.includes('琥珀') || 
              lowerFontName.includes('新魏')) {
        return 'cursive';
    } else {
        // 默认
        return 'sans-serif';
    }
}

/**
 * 处理自定义字体上传
 * @param {File} fontFile - 上传的字体文件
 */
export function handleFontUpload(fontFile) {
    import('./api.js').then(api => {
        showLoading('正在上传字体...');
        
        api.uploadFontApi(
            fontFile,
            response => {
                hideLoading();
                if (response.success) {
                    showGeneralMessage('字体上传成功！', 'success', 'font-upload-success');
                    
                    // 刷新字体列表并选择新上传的字体
                    loadFontList(response.path);
                } else {
                    showGeneralMessage('字体上传失败: ' + (response.error || '未知错误'), 'error', 'font-upload-error');
                }
            },
            error => {
                hideLoading();
                showGeneralMessage('字体上传失败: ' + error, 'error', 'font-upload-error');
            }
        );
    });
}

/**
 * 显示AI校对设置弹窗
 */
export function showProofreadingSettingsModal() {
    $("#proofreadingSettingsModal").show();
}

/**
 * 隐藏AI校对设置弹窗
 */
export function hideProofreadingSettingsModal() {
    $("#proofreadingSettingsModal").hide();
}