// src/app/static/js/state.js

// 导入常量模块
import * as constants from './constants.js';
import * as bubbleStateModule from './bubble_state.js';

// --- 核心状态变量 ---
export let images = [];
export let currentImageIndex = -1;
export let editModeActive = false;
export let selectedBubbleIndex = -1;
export let bubbleStates = [];    // 统一状态：BubbleState 对象数组
export let initialBubbleStates = [];  // 初始状态备份
export let currentSessionName = null;
export let capterMark = {}

// --- 书籍/章节状态 ---
export let currentBookId = null;      // 当前书籍ID
export let currentChapterId = null;   // 当前章节ID
export let currentBookTitle = '';     // 当前书籍标题
export let currentChapterTitle = '';  // 当前章节标题

// --- 批量处理状态 ---
export let isBatchTranslationInProgress = false;
export let isBatchTranslationPaused = false;  // 批量翻译暂停状态
export let batchTranslationResumeCallback = null;  // 继续翻译的回调函数

// --- 提示词状态 ---
export let currentPromptContent = "";
export let defaultPromptContent = "";
export let defaultTranslateJsonPrompt = constants.DEFAULT_TRANSLATE_JSON_PROMPT;
export let isTranslateJsonMode = false;
export let savedPromptNames = [];

export let currentTextboxPromptContent = "";
export let defaultTextboxPromptContent = "";
export let savedTextboxPromptNames = [];
export let useTextboxPrompt = false;

// --- 默认设置 ---
export let defaultFontSize = 25;
export let defaultFontFamily = 'fonts/STSONG.TTF';
export let defaultLayoutDirection = 'vertical';
export let defaultTextColor = '#000000';
export let defaultFillColor = '#FFFFFF';
export let defaultStrokeEnabled = true;
export let defaultStrokeColor = '#FFFFFF';
export let defaultStrokeWidth = 3;

// --- 文本检测器状态 ---
export let textDetector = 'ctd';  // 默认使用 CTD 检测器

// --- 文本框扩展参数状态 ---
export let boxExpandRatio = 0;      // 整体扩展比例 (%)
export let boxExpandTop = 0;        // 上边额外扩展 (%)
export let boxExpandBottom = 0;     // 下边额外扩展 (%)
export let boxExpandLeft = 0;       // 左边额外扩展 (%)
export let boxExpandRight = 0;      // 右边额外扩展 (%)

// --- 检测调试选项 ---
export let showDetectionDebug = false;  // 是否显示检测框调试

// --- PDF处理方式 ---
export let pdfProcessingMethod = 'backend';  // 'backend' 或 'frontend'

// --- 精确文字掩膜选项 ---
export let usePreciseMask = true;  // 是否使用模型生成的精确文字掩膜（仅 CTD/Default 支持），默认开启
export let maskDilateSize = 10;     // 掩膜膨胀大小（像素）
export let maskBoxExpandRatio = 20; // 标注框区域扩大比例（%）

// --- OCR 引擎状态 ---
export let baiduOcrSourceLanguage = 'auto_detect';
export let aiVisionOcrPrompt = constants.DEFAULT_AI_VISION_OCR_PROMPT;
export let defaultAiVisionOcrJsonPrompt = constants.DEFAULT_AI_VISION_OCR_JSON_PROMPT;
export let isAiVisionOcrJsonMode = false;

// VVVVVV 新增状态变量 VVVVVV
export let customAiVisionBaseUrl = ''; // 用于存储自定义AI视觉服务的Base URL
// ^^^^^^ 结束新增状态变量 ^^^^^^

// --- 新增 rpm 状态 ---
export let rpmLimitTranslation = constants.DEFAULT_rpm_TRANSLATION; // 从前端常量获取默认值
export let rpmLimitAiVisionOcr = constants.DEFAULT_rpm_AI_VISION_OCR;
// --------------------

// --- 高质量翻译模式状态 ---
export let hqTranslateProvider = 'siliconflow';
export let hqApiKey = '';
export let hqModelName = '';
export let hqCustomBaseUrl = '';
export let hqBatchSize = 3;
export let hqSessionReset = 20;
export let hqRpmLimit = 7;
export let hqLowReasoning = false;
export let hqNoThinkingMethod = 'gemini'; // 取消思考的方法: 'gemini' 或 'volcano'
export let hqPrompt = constants.DEFAULT_HQ_TRANSLATE_PROMPT;
export let hqForceJsonOutput = true;
export let hqUseStream = false; // 流式调用开关

// --- AI校对功能状态 ---
export let isProofreadingEnabled = true;
export let proofreadingRounds = [];
export let proofreadingNoThinkingMethod = 'gemini'; // 取消思考的方法: 'gemini' 或 'volcano'

// --- 重试机制状态 ---
export let translationMaxRetries = constants.DEFAULT_TRANSLATION_MAX_RETRIES;
export let hqTranslationMaxRetries = constants.DEFAULT_HQ_TRANSLATION_MAX_RETRIES;
export let proofreadingMaxRetries = constants.DEFAULT_PROOFREADING_MAX_RETRIES;

// --------------------

export let strokeEnabled = defaultStrokeEnabled;
export let strokeColor = defaultStrokeColor;
export let strokeWidth = defaultStrokeWidth;

// --- 更新状态的函数 ---

/**
 * 添加图片到状态数组
 * @param {object} imageObject - 要添加的图片对象
 */
export function addImage(imageObject) {
    const newImage = {
        ...imageObject,
        savedManualCoords: null,
        hasUnsavedChanges: false,
        fillColor: defaultFillColor,
        strokeEnabled: defaultStrokeEnabled,
        strokeColor: defaultStrokeColor,
        strokeWidth: defaultStrokeWidth,
    };
    images.push(newImage);
}

/**
 * 更新图片数组 (如果直接替换整个数组，确保新对象也有默认值)
 * @param {Array} newImages - 新的图片数组
 */
export function setImages(newImages) {
    images = newImages.map(img => ({
        ...img,
        savedManualCoords: img.savedManualCoords || null,
        hasUnsavedChanges: img.hasUnsavedChanges || false,
        strokeEnabled: img.strokeEnabled !== undefined ? img.strokeEnabled : defaultStrokeEnabled,
        strokeColor: img.strokeColor || defaultStrokeColor,
        strokeWidth: img.strokeWidth !== undefined ? img.strokeWidth : defaultStrokeWidth
    }));
}

/**
 * 设置当前图片索引
 * @param {number} index - 新的索引
 */
export function setCurrentImageIndex(index) {
    currentImageIndex = index;
}

/**
 * 获取当前显示的图片对象
 * @returns {object | null} 当前图片对象或 null
 */
export function getCurrentImage() {
    if (currentImageIndex >= 0 && currentImageIndex < images.length) {
        return images[currentImageIndex];
    }
    return null;
}

/**
 * 更新当前图片的特定属性
 * @param {string} key - 要更新的属性名
 * @param {*} value - 新的值
 */
export function updateCurrentImageProperty(key, value) {
    const currentImage = getCurrentImage();
    if (currentImage) {
        currentImage[key] = value;
        console.log(`状态更新: 图片 ${currentImageIndex} 的属性 ${key} 更新为`, value);
    } else {
        console.warn(`状态更新失败: 无法更新属性 ${key}，当前图片索引无效 ${currentImageIndex}`);
    }
}

/**
 * 更新指定索引图片的特定属性
 * @param {number} index - 要更新的图片索引
 * @param {string} key - 要更新的属性名
 * @param {*} value - 新的值
 */
export function updateImagePropertyByIndex(index, key, value) {
    if (index >= 0 && index < images.length) {
        images[index][key] = value;
        console.log(`状态更新 (索引 ${index}): 属性 ${key} 更新为`, value);
    } else {
        console.warn(`状态更新失败: 无法更新索引 ${index} 的属性 ${key}，索引无效`);
    }
}

/**
 * 设置编辑模式状态
 * @param {boolean} isActive - 是否激活编辑模式
 */
export function setEditModeActive(isActive) {
    editModeActive = isActive;
}

/**
 * 设置当前选中的气泡索引
 * @param {number} index - 选中的气泡索引
 */
export function setSelectedBubbleIndex(index) {
    selectedBubbleIndex = index;
}

// ============================================================
// 统一气泡状态管理函数 (BubbleState)
// ============================================================

/**
 * 设置当前图片的统一气泡状态数组
 * @param {Array} states - BubbleState 对象数组
 */
export function setBubbleStates(states) {
    bubbleStates = states;
    const currentImage = getCurrentImage();
    if (currentImage) {
        currentImage.bubbleStates = JSON.parse(JSON.stringify(states));
    }
    console.log(`状态更新: 设置了 ${states.length} 个气泡状态`);
}

/**
 * 从后端响应初始化气泡状态
 * @param {Object} response - 后端 API 响应
 * @param {Object} globalDefaults - 全局默认设置
 */
export function initBubbleStatesFromResponse(response, globalDefaults = {}) {
    const states = bubbleStateModule.createBubbleStatesFromResponse(response, globalDefaults);
    setBubbleStates(states);
    return states;
}

/**
 * 更新单个气泡的状态
 * @param {number} index - 气泡索引
 * @param {Object} updates - 要更新的属性
 */
export function updateSingleBubbleState(index, updates) {
    if (index >= 0 && index < bubbleStates.length) {
        bubbleStates[index] = { ...bubbleStates[index], ...updates };
        const currentImage = getCurrentImage();
        if (currentImage) {
            currentImage.bubbleStates = JSON.parse(JSON.stringify(bubbleStates));
        }
        console.log(`状态更新: 气泡 ${index} 已更新`);
    } else {
        console.warn(`状态更新失败: 无效的气泡索引 ${index}`);
    }
}

/**
 * 设置初始气泡状态备份
 * @param {Array} states - BubbleState 对象数组
 */
export function setInitialBubbleStates(states) {
    initialBubbleStates = JSON.parse(JSON.stringify(states));
}

/**
 * 获取当前气泡状态用于 API 请求
 * @returns {Array} 用于 API 请求的气泡状态数组
 */
export function getBubbleStatesForApi() {
    return bubbleStateModule.bubbleStatesToApiRequest(bubbleStates);
}

/**
 * 获取气泡的译文列表
 * @returns {Array} 译文文本数组
 */
export function getBubbleTexts() {
    return bubbleStateModule.getTextsFromStates(bubbleStates);
}

/**
 * 获取气泡的坐标列表
 * @returns {Array} 坐标数组
 */
export function getBubbleCoords() {
    return bubbleStateModule.getCoordsFromStates(bubbleStates);
}

// ============================================================
// 其他状态管理函数
// ============================================================

/**
 * 设置漫画翻译提示词状态
 * @param {string} current - 当前内容
 * @param {string} defaultContent - 默认内容
 * @param {Array<string>} names - 已保存名称列表
 */
export function setPromptState(current, defaultContent, names, defaultJsonPromptContent = constants.DEFAULT_TRANSLATE_JSON_PROMPT) {
    currentPromptContent = current;
    defaultPromptContent = defaultContent;
    defaultTranslateJsonPrompt = defaultJsonPromptContent;
    savedPromptNames = names;
}

/**
 * 设置文本框提示词状态
 * @param {string} current - 当前内容
 * @param {string} defaultContent - 默认内容
 * @param {Array<string>} names - 已保存名称列表
 */
export function setTextboxPromptState(current, defaultContent, names) {
    currentTextboxPromptContent = current;
    defaultTextboxPromptContent = defaultContent;
    savedTextboxPromptNames = names;
}

/**
 * 设置是否使用文本框提示词
 * @param {boolean} use - 是否使用
 */
export function setUseTextboxPrompt(use) {
    useTextboxPrompt = use;
}

/**
 * 设置默认字体大小 (在 main.js 初始化时调用)
 * @param {number} size
 */
export function setDefaultFontSize(size) {
    defaultFontSize = size;
}

/**
 * 设置默认字体 (在 main.js 初始化时调用)
 * @param {string} family
 */
export function setDefaultFontFamily(family) {
    defaultFontFamily = family;
}

/**
 * 设置默认排版方向 (在 main.js 初始化时调用)
 * @param {string} direction
 */
export function setDefaultLayoutDirection(direction) {
    // 注意：'auto' 不是有效的实际排版方向，只在UI选择器中表示"自动检测"
    // 作为默认值存储时，应该使用具体的方向（'vertical' 或 'horizontal'）
    // 这样在编辑模式等场景中不会传递 'auto' 给后端
    defaultLayoutDirection = (direction === 'auto') ? 'vertical' : direction;
}

/**
 * 设置默认文本颜色 (在 main.js 初始化时调用)
 * @param {string} color
 */
export function setDefaultTextColor(color) {
    defaultTextColor = color;
}

/**
 * 设置默认填充颜色 (在 main.js 初始化时调用)
 * @param {string} color
 */
export function setDefaultFillColor(color) {
    defaultFillColor = color;
}


export function deleteImage(index) {
    if (index >= 0 && index < images.length) {
        images.splice(index, 1);

        if (currentImageIndex === index) {
            currentImageIndex = Math.min(index, images.length - 1);
            if (images.length === 0) {
                currentImageIndex = -1;
                setEditModeActive(false);
            }
        } else if (currentImageIndex > index) {
            currentImageIndex--;
        }
        return true;
    }
    return false;
}

export function clearImages() {
    images = [];
    currentImageIndex = -1;
    bubbleStates = [];
    initialBubbleStates = [];
    selectedBubbleIndex = -1;
    setEditModeActive(false);
    setCurrentSessionName(null);
}

/**
 * 设置当前活动的会话名称。
 * @param {string | null} name - 会话名称，或 null 表示没有活动会话。
 */
export function setCurrentSessionName(name) {
    currentSessionName = name;
    console.log(`状态更新: 当前会话名称 -> ${name}`);
    // 可选：更新 UI，例如在标题栏显示当前会话名
    // ui.updateWindowTitle(name); // 需要在 ui.js 中实现此函数
}

/**
 * 设置文本检测器类型
 * @param {string} detector - 检测器类型 ('ctd', 'yolo', 'yolov5', 'default')
 */
export function setTextDetector(detector) {
    textDetector = detector;
    console.log(`状态更新: 文本检测器 -> ${textDetector}`);
}

/**
 * 设置文本框整体扩展比例
 * @param {number} ratio - 扩展比例 (0-50%)
 */
export function setBoxExpandRatio(ratio) {
    boxExpandRatio = Math.max(0, Math.min(50, parseInt(ratio) || 0));
    console.log(`状态更新: 整体扩展比例 -> ${boxExpandRatio}%`);
}

/**
 * 设置文本框上边额外扩展比例
 * @param {number} ratio - 扩展比例 (0-50%)
 */
export function setBoxExpandTop(ratio) {
    boxExpandTop = Math.max(0, Math.min(50, parseInt(ratio) || 0));
    console.log(`状态更新: 上边扩展 -> ${boxExpandTop}%`);
}

/**
 * 设置文本框下边额外扩展比例
 * @param {number} ratio - 扩展比例 (0-50%)
 */
export function setBoxExpandBottom(ratio) {
    boxExpandBottom = Math.max(0, Math.min(50, parseInt(ratio) || 0));
    console.log(`状态更新: 下边扩展 -> ${boxExpandBottom}%`);
}

/**
 * 设置文本框左边额外扩展比例
 * @param {number} ratio - 扩展比例 (0-50%)
 */
export function setBoxExpandLeft(ratio) {
    boxExpandLeft = Math.max(0, Math.min(50, parseInt(ratio) || 0));
    console.log(`状态更新: 左边扩展 -> ${boxExpandLeft}%`);
}

/**
 * 设置文本框右边额外扩展比例
 * @param {number} ratio - 扩展比例 (0-50%)
 */
export function setBoxExpandRight(ratio) {
    boxExpandRight = Math.max(0, Math.min(50, parseInt(ratio) || 0));
    console.log(`状态更新: 右边扩展 -> ${boxExpandRight}%`);
}

/**
 * 设置是否显示检测框调试
 * @param {boolean} enabled - 是否启用
 */
export function setShowDetectionDebug(enabled) {
    showDetectionDebug = !!enabled;
    console.log(`状态更新: 检测框调试 -> ${showDetectionDebug}`);
}

/**
 * 设置PDF处理方式
 * @param {string} method - 处理方式 ('backend' 或 'frontend')
 */
export function setPdfProcessingMethod(method) {
    pdfProcessingMethod = (method === 'frontend') ? 'frontend' : 'backend';
    console.log(`状态更新: PDF处理方式 -> ${pdfProcessingMethod}`);
}

/**
 * 设置是否使用精确文字掩膜
 * @param {boolean} enabled - 是否启用
 */
export function setUsePreciseMask(enabled) {
    usePreciseMask = !!enabled;
    console.log(`状态更新: 精确文字掩膜 -> ${usePreciseMask}`);
}

/**
 * 设置掩膜膨胀大小
 * @param {number} size - 膨胀大小（像素）
 */
export function setMaskDilateSize(size) {
    maskDilateSize = parseInt(size) || 0;
    console.log(`状态更新: 掩膜膨胀大小 -> ${maskDilateSize}`);
}

/**
 * 设置标注框区域扩大比例
 * @param {number} ratio - 扩大比例（%）
 */
export function setMaskBoxExpandRatio(ratio) {
    maskBoxExpandRatio = parseInt(ratio) || 0;
    console.log(`状态更新: 标注框区域扩大比例 -> ${maskBoxExpandRatio}%`);
}

/**
 * 设置百度OCR源语言
 * @param {string} language - 百度OCR源语言代码
 */
export function setBaiduOcrSourceLanguage(language) {
    baiduOcrSourceLanguage = language;
}

/**
 * 获取百度OCR源语言
 * @returns {string} - 百度OCR源语言代码
 */
export function getBaiduOcrSourceLanguage() {
    return baiduOcrSourceLanguage;
}

/**
 * 设置AI视觉OCR提示词
 * @param {string} prompt - 提示词内容
 */
export function setAiVisionOcrPrompt(prompt) {
    aiVisionOcrPrompt = prompt;
}

/**
 * 设置漫画翻译提示词模式和内容
 * @param {boolean|string} mode - 如果是布尔值，则使用旧逻辑(true=json,false=normal)；如果是字符串，则是新的模式名称
 * @param {string} [content=null] - 如果提供，则设置当前内容；否则根据模式使用默认值
 */
export function setTranslatePromptMode(mode, content = null) {
    // 兼容旧的布尔参数
    if (typeof mode === 'boolean') {
        isTranslateJsonMode = mode;
        if (content !== null) {
            currentPromptContent = content;
        } else {
            currentPromptContent = mode ? defaultTranslateJsonPrompt : defaultPromptContent;
        }
    } else {
        // 新逻辑，mode是字符串，表示模式名称
        switch (mode) {
            case 'json':
                isTranslateJsonMode = true;
                if (content !== null) {
                    currentPromptContent = content;
                } else {
                    currentPromptContent = defaultTranslateJsonPrompt;
                }
                break;
            case 'normal':
            default:
                isTranslateJsonMode = false;
                if (content !== null) {
                    currentPromptContent = content;
                } else {
                    currentPromptContent = defaultPromptContent;
                }
                break;
            // 未来可以在这里添加更多的case来支持更多模式
        }
    }
    console.log(`状态更新: 漫画翻译JSON模式 -> ${isTranslateJsonMode}, 当前提示词已更新`);
}

/**
 * 设置AI视觉OCR提示词模式和内容
 * @param {boolean|string} mode - 如果是布尔值，则使用旧逻辑(true=json,false=normal)；如果是字符串，则是新的模式名称
 * @param {string} [content=null] - 如果提供，则设置当前内容；否则根据模式使用默认值
 */
export function setAiVisionOcrPromptMode(mode, content = null) {
    // 兼容旧的布尔参数
    if (typeof mode === 'boolean') {
        isAiVisionOcrJsonMode = mode;
        if (content !== null) {
            aiVisionOcrPrompt = content;
        } else {
            aiVisionOcrPrompt = mode ? defaultAiVisionOcrJsonPrompt : constants.DEFAULT_AI_VISION_OCR_PROMPT;
        }
    } else {
        // 新逻辑，mode是字符串，表示模式名称
        switch (mode) {
            case 'json':
                isAiVisionOcrJsonMode = true;
                if (content !== null) {
                    aiVisionOcrPrompt = content;
                } else {
                    aiVisionOcrPrompt = defaultAiVisionOcrJsonPrompt;
                }
                break;
            case 'normal':
            default:
                isAiVisionOcrJsonMode = false;
                if (content !== null) {
                    aiVisionOcrPrompt = content;
                } else {
                    aiVisionOcrPrompt = constants.DEFAULT_AI_VISION_OCR_PROMPT;
                }
                break;
            // 未来可以在这里添加更多的case来支持更多模式
        }
    }
    console.log(`状态更新: AI视觉OCR JSON模式 -> ${isAiVisionOcrJsonMode}, 当前提示词已更新`);
}

/**
 * 设置AI视觉OCR默认提示词
 * @param {string} normalPrompt - 普通模式的默认提示词
 * @param {string} jsonPrompt - JSON模式的默认提示词
 */
export function setAiVisionOcrDefaultPrompts(normalPrompt, jsonPrompt) {
    aiVisionOcrPrompt = isAiVisionOcrJsonMode ? jsonPrompt : normalPrompt;
    defaultAiVisionOcrJsonPrompt = jsonPrompt;
}

/**
 * 设置批量翻译进行状态
 * @param {boolean} isInProgress - 批量翻译是否正在进行
 */
export function setBatchTranslationInProgress(isInProgress) {
    isBatchTranslationInProgress = isInProgress;
    // 如果结束批量翻译，同时重置暂停状态
    if (!isInProgress) {
        isBatchTranslationPaused = false;
        batchTranslationResumeCallback = null;
    }
    console.log(`状态更新: 批量翻译进度状态 -> ${isInProgress ? '进行中' : '已完成'}`);
}

/**
 * 设置批量翻译暂停状态
 * @param {boolean} isPaused - 是否暂停
 */
export function setBatchTranslationPaused(isPaused) {
    isBatchTranslationPaused = isPaused;
    console.log(`状态更新: 批量翻译暂停状态 -> ${isPaused ? '已暂停' : '继续中'}`);
}

/**
 * 设置继续翻译的回调函数
 * @param {Function|null} callback - 继续时调用的回调
 */
export function setBatchTranslationResumeCallback(callback) {
    batchTranslationResumeCallback = callback;
}

/**
 * 继续批量翻译
 */
export function resumeBatchTranslation() {
    if (isBatchTranslationPaused && batchTranslationResumeCallback) {
        isBatchTranslationPaused = false;
        const callback = batchTranslationResumeCallback;
        batchTranslationResumeCallback = null;
        callback();
        console.log('状态更新: 批量翻译已继续');
    }
}

/**
 * 设置翻译服务的rpm限制
 * @param {number} limit - 每分钟请求数 (0表示无限制)
 */
export function setrpmLimitTranslation(limit) {
    const newLimit = parseInt(limit);
    rpmLimitTranslation = isNaN(newLimit) || newLimit < 0 ? 0 : newLimit;
    console.log(`状态更新: 翻译服务 rpm -> ${rpmLimitTranslation}`);
}

/**
 * 设置AI视觉OCR服务的rpm限制
 * @param {number} limit - 每分钟请求数 (0表示无限制)
 */
export function setrpmLimitAiVisionOcr(limit) {
    const newLimit = parseInt(limit);
    rpmLimitAiVisionOcr = isNaN(newLimit) || newLimit < 0 ? 0 : newLimit;
    console.log(`状态更新: AI视觉OCR rpm -> ${rpmLimitAiVisionOcr}`);
}

/**
 * 设置普通翻译的最大重试次数
 * @param {number} value - 重试次数 (0表示不重试)
 */
export function setTranslationMaxRetries(value) {
    const newValue = parseInt(value);
    translationMaxRetries = isNaN(newValue) || newValue < 0 ? 0 : Math.min(newValue, 10);
    console.log(`状态更新: 普通翻译重试次数 -> ${translationMaxRetries}`);
}

/**
 * 设置高质量翻译的最大重试次数
 * @param {number} value - 重试次数 (0表示不重试)
 */
export function setHqTranslationMaxRetries(value) {
    const newValue = parseInt(value);
    hqTranslationMaxRetries = isNaN(newValue) || newValue < 0 ? 0 : Math.min(newValue, 10);
    console.log(`状态更新: 高质量翻译重试次数 -> ${hqTranslationMaxRetries}`);
}

/**
 * 设置AI校对的最大重试次数
 * @param {number} value - 重试次数 (0表示不重试)
 */
export function setProofreadingMaxRetries(value) {
    const newValue = parseInt(value);
    proofreadingMaxRetries = isNaN(newValue) || newValue < 0 ? 0 : Math.min(newValue, 10);
    console.log(`状态更新: AI校对重试次数 -> ${proofreadingMaxRetries}`);
}

// VVVVVV 新增 setter 函数 VVVVVV
/**
 * 设置自定义AI视觉服务的Base URL
 * @param {string} url - Base URL
 */
export function setCustomAiVisionBaseUrl(url) {
    customAiVisionBaseUrl = url;
    console.log(`状态更新: 自定义AI视觉Base URL -> ${url}`);
}
// ^^^^^^ 结束新增 setter 函数 ^^^^^^

/**
 * 设置高质量翻译服务商
 * @param {string} provider - 服务商标识
 */
export function setHqTranslateProvider(provider) {
    hqTranslateProvider = provider;
}

/**
 * 设置高质量翻译API Key
 * @param {string} apiKey - API密钥
 */
export function setHqApiKey(apiKey) {
    hqApiKey = apiKey;
}

/**
 * 设置高质量翻译模型名称
 * @param {string} modelName - 模型名称
 */
export function setHqModelName(modelName) {
    hqModelName = modelName;
}

/**
 * 设置高质量翻译自定义Base URL
 * @param {string} url - 自定义Base URL
 */
export function setHqCustomBaseUrl(url) {
    hqCustomBaseUrl = url;
}

/**
 * 设置高质量翻译每批次图片数
 * @param {number} size - 每批次图片数
 */
export function setHqBatchSize(size) {
    hqBatchSize = parseInt(size) || 3;
}

/**
 * 设置高质量翻译会话重置频率
 * @param {number} reset - 会话重置频率
 */
export function setHqSessionReset(reset) {
    hqSessionReset = parseInt(reset) || 20;
}

/**
 * 设置高质量翻译RPM限制
 * @param {number} limit - RPM限制
 */
export function setHqRpmLimit(limit) {
    hqRpmLimit = parseInt(limit) || 7;
}

/**
 * 设置高质量翻译是否使用低推理模式
 * @param {boolean} low - 是否使用低推理模式
 */
export function setHqLowReasoning(low) {
    hqLowReasoning = low;
}

/**
 * 设置高质量翻译取消思考方法
 * @param {string} method - 取消思考方法('gemini'或'volcano')
 */
export function setHqNoThinkingMethod(method) {
    hqNoThinkingMethod = method;
    console.log(`状态更新: 高质量翻译取消思考方法 -> ${method}`);
}

/**
 * 设置高质量翻译提示词
 * @param {string} prompt - 提示词
 */
export function setHqPrompt(prompt) {
    hqPrompt = prompt;
}

/**
 * 设置高质量翻译强制JSON输出选项
 * @param {boolean} force - 是否强制JSON输出
 */
export function setHqForceJsonOutput(force) {
    hqForceJsonOutput = force;
    console.log(`状态更新: 高质量翻译强制JSON输出 -> ${force ? '启用' : '禁用'}`);
}

/**
 * 设置高质量翻译流式调用选项
 * @param {boolean} useStream - 是否启用流式调用
 */
export function setHqUseStream(useStream) {
    hqUseStream = useStream;
    console.log(`状态更新: 高质量翻译流式调用 -> ${useStream ? '启用' : '禁用'}`);
}

export function setDefaultStrokeSettings(enabled, color, width) {
    defaultStrokeEnabled = enabled;
    defaultStrokeColor = color;
    defaultStrokeWidth = width;
}

export function setStrokeEnabled(enabled) {
    strokeEnabled = enabled;
    console.log(`状态更新: strokeEnabled -> ${strokeEnabled}`);
}

export function setStrokeColor(color) {
    strokeColor = color;
    console.log(`状态更新: strokeColor -> ${strokeColor}`);
}

export function setStrokeWidth(width) {
    strokeWidth = parseInt(width);
    if (isNaN(strokeWidth) || strokeWidth < 0) {
        strokeWidth = 0;
    }
    console.log(`状态更新: strokeWidth -> ${strokeWidth}`);
}

/**
 * 设置校对功能启用状态
 * @param {boolean} enabled - 是否启用
 */
export function setProofreadingEnabled(enabled) {
    isProofreadingEnabled = enabled;
}

/**
 * 设置AI校对轮次配置
 * @param {Array} rounds - 校对轮次配置数组
 */
export function setProofreadingRounds(rounds) {
    proofreadingRounds.length = 0; // 清空数组
    if (rounds && Array.isArray(rounds)) {
        proofreadingRounds.push(...rounds); // 添加新元素
    }
    console.log(`状态更新: AI校对轮次配置已更新，共 ${proofreadingRounds.length} 个轮次`);
}

/**
 * 设置AI校对取消思考方法
 * @param {string} method - 取消思考方法('gemini'或'volcano')
 */
export function setProofreadingNoThinkingMethod(method) {
    proofreadingNoThinkingMethod = method;
    console.log(`状态更新: AI校对取消思考方法 -> ${method}`);
}

// --- 书籍/章节状态设置函数 ---

/**
 * 设置当前书籍ID
 * @param {string|null} bookId - 书籍ID
 */
export function setCurrentBookId(bookId) {
    currentBookId = bookId;
    console.log(`状态更新: 当前书籍ID -> ${bookId}`);
}

/**
 * 设置当前章节ID
 * @param {string|null} chapterId - 章节ID
 */
export function setCurrentChapterId(chapterId) {
    currentChapterId = chapterId;
    console.log(`状态更新: 当前章节ID -> ${chapterId}`);
}

/**
 * 设置当前书籍标题
 * @param {string} title - 书籍标题
 */
export function setCurrentBookTitle(title) {
    currentBookTitle = title;
}

/**
 * 设置当前章节标题
 * @param {string} title - 章节标题
 */
export function setCurrentChapterTitle(title) {
    currentChapterTitle = title;
}

/**
 * 设置书籍和章节上下文
 * @param {string|null} bookId - 书籍ID
 * @param {string|null} chapterId - 章节ID
 * @param {string} bookTitle - 书籍标题
 * @param {string} chapterTitle - 章节标题
 */
export function setBookChapterContext(bookId, chapterId, bookTitle = '', chapterTitle = '') {
    currentBookId = bookId;
    currentChapterId = chapterId;
    currentBookTitle = bookTitle;
    currentChapterTitle = chapterTitle;
    console.log(`状态更新: 书籍/章节上下文 -> 书籍: ${bookTitle}(${bookId}), 章节: ${chapterTitle}(${chapterId})`);
}

/**
 * 清除书籍/章节上下文
 */
export function clearBookChapterContext() {
    currentBookId = null;
    currentChapterId = null;
    currentBookTitle = '';
    currentChapterTitle = '';
    console.log('状态更新: 已清除书籍/章节上下文');
}
