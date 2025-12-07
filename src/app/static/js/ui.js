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
    const proofreadButton = $("#proofreadButton"); // 校对按钮
    const proofreadSettingsButton = $("#proofreadSettingsButton"); // 校对设置按钮
    const startHqTranslationBtn = $("#startHqTranslationBtn"); // 侧边栏高质量翻译按钮

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
    // 同步更新齿轮按钮状态
    const applySettingsOptionsBtn = $("#applySettingsOptionsBtn");
    if (applySettingsOptionsBtn.length > 0) {
        applySettingsOptionsBtn.prop('disabled', !hasImages || isLoading);
    }
    
    // 校对按钮状态更新
    proofreadButton.prop('disabled', !hasImages || isLoading || state.isBatchTranslationInProgress);
    // 校对设置按钮始终保持启用状态，类似于"加载/管理会话"按钮
    proofreadSettingsButton.prop('disabled', false);
    
    // 高质量翻译按钮状态更新
    startHqTranslationBtn.prop('disabled', !hasImages || isLoading || state.isBatchTranslationInProgress);


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
 * 更新暂停按钮状态
 * @param {boolean} isPaused - 是否处于暂停状态
 */
export function updatePauseButton(isPaused) {
    const pauseButton = $("#pauseTranslationButton");
    const progressBar = $(".progress-bar");
    
    if (isPaused) {
        pauseButton.addClass('paused');
        pauseButton.html('<span class="pause-icon">▶</span> 继续');
        progressBar.addClass('paused');
    } else {
        pauseButton.removeClass('paused');
        pauseButton.html('<span class="pause-icon">⏸</span> 暂停');
        progressBar.removeClass('paused');
    }
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
    
    // 修复：当 percentage 为 0 且没有文本时，隐藏进度条（用于重置）
    if (percentage === 0 && !text) {
        translationProgressBar.hide();
    } else if (percentage < 100) {
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

// populatePromptDropdown, updateModelSuggestions, updateOllamaModelList, updateSakuraModelList
// 和其他翻译服务UI函数已移除 - 这些功能现在由设置模态框(settings_modal.js)处理

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
    
    // 队列模式：立即移除所有现有消息，只显示最新的一个
    messageContainer.find('.message').remove();
    
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
// 旧版 updateBubbleListUI 和 updateBubbleEditArea 已删除
// 新版编辑模式使用 edit_mode.js 中的 selectBubbleNew 等函数

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
    if (!currentImage || !currentImage.bubbleCoords) return;

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
    const toggleEditModeButton = $("#toggleEditModeButton");
    const editModeContainer = $("#editModeContainer"); // 旧版编辑模式容器
    const detectedTextInfo = $("#detectedTextInfo");
    const settingsSidebar = $("#settings-sidebar");
    const thumbnailSidebar = $("#thumbnail-sidebar");

    if (isActive) {
        toggleEditModeButton.text("退出编辑模式").addClass("active");
        
        // 隐藏旧版编辑模式容器（新版通过CSS .edit-mode-active控制）
        editModeContainer.hide();
        detectedTextInfo.hide();
        
        // 隐藏侧边栏以获得最大工作空间
        settingsSidebar.addClass('edit-mode-hidden');
        thumbnailSidebar.addClass('edit-mode-hidden');
        
        $('body').addClass('edit-mode-active');
        
        // 添加窗口大小改变事件
        $(window).on('resize.bubbleHighlight', function() {
            updateBubbleHighlight(state.selectedBubbleIndex);
        });
    } else {
        toggleEditModeButton.text("切换编辑模式").removeClass("active");
        editModeContainer.hide();
        detectedTextInfo.show();
        
        // 恢复侧边栏显示
        settingsSidebar.removeClass('edit-mode-hidden');
        thumbnailSidebar.removeClass('edit-mode-hidden');
        
        $('.highlight-bubble').remove(); // 移除所有高亮框
        $('.bubble-highlight-box').remove(); // 移除新版高亮框
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
 * @returns {{useInpainting: boolean, useLama: boolean, lamaModel: string}}
 */
export function getRepairSettings() {
    const repairMethod = $('#useInpainting').val(); // 在函数内获取元素
    // console.log("获取修复设置:", repairMethod); // 可以取消注释用于调试
    
    // 判断是否使用 LAMA（lama_mpe 或 litelama）
    const isLama = repairMethod === 'lama_mpe' || repairMethod === 'litelama';
    
    // 确定 LAMA 模型类型
    const lamaModel = (repairMethod === 'litelama') ? 'litelama' : 'lama_mpe';
    
    return {
        useInpainting: repairMethod === 'true', // MI-GAN (保留兼容)
        useLama: isLama,                        // 是否使用 LAMA
        lamaModel: lamaModel                    // LAMA 模型选择: 'lama_mpe' 或 'litelama'
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

// toggleYoudaoTranslateUI 和 testYoudaoTranslateConnection 已移除 - 这些功能现在由设置模态框处理

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

// toggleCustomOpenAiUI, toggleCustomAiVisionBaseUrlUI 已移除 - 现在由设置模态框处理

/**
 * 更新rpm输入框的显示值
 */
export function updaterpmInputFields() {
    $('#rpmTranslation').val(state.rpmLimitTranslation);
    $('#rpmAiVisionOcr').val(state.rpmLimitAiVisionOcr);
    console.log("UI 更新: rpm输入框已更新为当前状态值。");
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
                    updateFontSelector($('#fontFamilyNew'), response, selectedFont);
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