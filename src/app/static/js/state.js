// src/app/static/js/state.js

<<<<<<< HEAD
// 导入常量模块
import * as constants from './constants.js';

// --- 核心状态变量 ---
export let images = [];
export let currentImageIndex = -1;
export let editModeActive = false;
export let selectedBubbleIndex = -1;
export let bubbleSettings = [];
export let initialBubbleSettings = [];
export let currentSessionName = null;

// --- 标注模式状态 ---
export let isLabelingModeActive = false;
export let manualBubbleCoords = [];
export let selectedManualBoxIndex = -1;
export let hasUnsavedChanges = false;

// --- 批量处理状态 ---
export let isBatchTranslationInProgress = false;

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

// --- OCR 引擎状态 ---
export let ocrEngine = 'auto';
export let baiduOcrVersion = 'standard';
export let aiVisionProvider = 'siliconflow';
export let aiVisionModelName = '';
export let aiVisionOcrPrompt = constants.DEFAULT_AI_VISION_OCR_PROMPT;
export let defaultAiVisionOcrJsonPrompt = constants.DEFAULT_AI_VISION_OCR_JSON_PROMPT;
export let isAiVisionOcrJsonMode = false;

// --- 新增 RPD 状态 ---
export let rpdLimitTranslation = constants.DEFAULT_RPD_TRANSLATION; // 从前端常量获取默认值
export let rpdLimitAiVisionOcr = constants.DEFAULT_RPD_AI_VISION_OCR;
// --------------------
=======
// --- 核心状态变量 ---
export let images = []; // 存储所有图片信息的数组
export let currentImageIndex = -1; // 当前显示的图片索引
export let editModeActive = false; // 编辑模式是否激活
export let selectedBubbleIndex = -1; // 编辑模式下选中的气泡索引
export let bubbleSettings = []; // 当前图片的独立气泡设置 (编辑模式用)
export let initialBubbleSettings = []; // 进入编辑模式时的初始气泡设置备份

// --- 提示词状态 ---
export let currentPromptContent = ""; // 当前漫画翻译提示词内容
export let defaultPromptContent = ""; // 默认漫画翻译提示词内容
export let savedPromptNames = []; // 已保存的漫画翻译提示词名称列表

export let currentTextboxPromptContent = ""; // 当前文本框提示词内容
export let defaultTextboxPromptContent = ""; // 默认文本框提示词内容
export let savedTextboxPromptNames = []; // 已保存的文本框提示词名称列表
export let useTextboxPrompt = false; // 是否启用文本框提示词

// --- 默认设置 (可以从 state 获取，避免硬编码) ---
// 注意：这些值应该在 main.js 初始化时从 DOM 读取一次
export let defaultFontSize = 25;
export let defaultFontFamily = 'fonts/STSONG.TTF'; // 初始值，会被 main.js 更新
export let defaultLayoutDirection = 'vertical';
export let defaultTextColor = '#000000';
export let defaultFillColor = '#FFFFFF' // 白色的填充颜色;
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915

// --- 更新状态的函数 ---

/**
<<<<<<< HEAD
 * 添加图片到状态数组
 * @param {object} imageObject - 要添加的图片对象
 */
export function addImage(imageObject) {
    const newImage = {
        ...imageObject,
        savedManualCoords: null,
        hasUnsavedChanges: false,
        fillColor: defaultFillColor,
        originalUseInpainting: undefined,
        originalUseLama: undefined,
        inpaintingStrength: constants.DEFAULT_INPAINTING_STRENGTH,
        blendEdges: true,
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
        hasUnsavedChanges: img.hasUnsavedChanges || false
    }));
    setHasUnsavedChanges(false);
=======
 * 更新图片数组
 * @param {Array} newImages - 新的图片数组
 */
export function setImages(newImages) {
    images = newImages;
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
}

/**
 * 设置当前图片索引
 * @param {number} index - 新的索引
 */
export function setCurrentImageIndex(index) {
    currentImageIndex = index;
<<<<<<< HEAD
    const currentImage = getCurrentImage();
    setHasUnsavedChanges(currentImage?.hasUnsavedChanges || false);
=======
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
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
<<<<<<< HEAD
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
=======
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
 * 设置编辑模式状态
 * @param {boolean} isActive - 是否激活编辑模式
 */
export function setEditModeActive(isActive) {
    editModeActive = isActive;
<<<<<<< HEAD
    if(isActive) setHasUnsavedChanges(false);
=======
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
}

/**
 * 设置当前选中的气泡索引
 * @param {number} index - 选中的气泡索引
 */
export function setSelectedBubbleIndex(index) {
    selectedBubbleIndex = index;
}

/**
 * 设置当前图片的气泡设置数组
 * @param {Array} settings - 新的气泡设置数组
 */
export function setBubbleSettings(settings) {
    bubbleSettings = settings;
<<<<<<< HEAD
=======
    // 同时更新当前图片的 bubbleSettings 属性
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
    updateCurrentImageProperty('bubbleSettings', JSON.parse(JSON.stringify(settings)));
}

/**
 * 更新单个气泡的设置
 * @param {number} index - 气泡索引
 * @param {object} newSetting - 新的设置对象
 */
export function updateSingleBubbleSetting(index, newSetting) {
    if (index >= 0 && index < bubbleSettings.length) {
<<<<<<< HEAD
        bubbleSettings[index] = { ...bubbleSettings[index], ...newSetting };
=======
        bubbleSettings[index] = { ...bubbleSettings[index], ...newSetting }; // 合并更新
        // 同时更新当前图片的 bubbleSettings 属性
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
        updateCurrentImageProperty('bubbleSettings', JSON.parse(JSON.stringify(bubbleSettings)));
    } else {
        console.warn(`状态更新失败: 无法更新气泡 ${index} 的设置，索引无效`);
    }
}

/**
 * 设置初始气泡设置备份
 * @param {Array} settings - 初始设置数组
 */
export function setInitialBubbleSettings(settings) {
    initialBubbleSettings = settings;
}

/**
 * 设置漫画翻译提示词状态
 * @param {string} current - 当前内容
 * @param {string} defaultContent - 默认内容
 * @param {Array<string>} names - 已保存名称列表
 */
<<<<<<< HEAD
export function setPromptState(current, defaultContent, names, defaultJsonPromptContent = constants.DEFAULT_TRANSLATE_JSON_PROMPT) {
    currentPromptContent = current;
    defaultPromptContent = defaultContent;
    defaultTranslateJsonPrompt = defaultJsonPromptContent;
=======
export function setPromptState(current, defaultContent, names) {
    currentPromptContent = current;
    defaultPromptContent = defaultContent;
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
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
    defaultLayoutDirection = direction;
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

<<<<<<< HEAD
/**
 * 设置手动标注坐标，并更新未保存状态
 * @param {Array<Array<number>>} coords - 新的坐标数组
 * @param {boolean} [fromLoad=false] - 是否是从已保存状态加载，或者只是切换图片加载的默认状态
 */
export function setManualCoords(coords, fromLoad = false) {
    if (isLabelingModeActive && !fromLoad) {
        const oldCoordsString = JSON.stringify(manualBubbleCoords);
        const newCoordsString = JSON.stringify(coords);
        if (oldCoordsString !== newCoordsString) {
            setHasUnsavedChanges(true);
            const currentImage = getCurrentImage();
            if (currentImage) {
                currentImage.hasUnsavedChanges = true;
            }
        }
    }
    manualBubbleCoords = coords;
    console.log(`状态更新: 手动坐标数量 -> ${coords.length}${fromLoad ? ' (已加载/默认)' : ''}`);
}

/**
 * 设置是否有未保存的更改
 * @param {boolean} value
 */
export function setHasUnsavedChanges(value) {
    if (value && !isLabelingModeActive) {
        console.warn("尝试在非标注模式下设置 hasUnsavedChanges=true，已忽略。");
        return;
    }
    hasUnsavedChanges = value;
    console.log(`状态更新: hasUnsavedChanges -> ${value}`);
}

/**
 * 设置标注模式状态
 * @param {boolean} isActive
 */
export function setLabelingModeActive(isActive) {
    isLabelingModeActive = isActive;
    if (!isActive) {
        setHasUnsavedChanges(false);
    }
    console.log(`状态更新: 标注模式状态 -> ${isActive}`);
}

/**
 * 设置选中手动框索引
 * @param {number} index
 */
export function setSelectedManualBoxIndex(index) {
    selectedManualBoxIndex = index;
    console.log(`状态更新: 选中手动框索引 -> ${index}`);
}

/**
 * 新增：保存当前手动标注框到当前图片对象
 * @returns {boolean} 是否成功保存
 */
export function saveManualCoordsToImage() {
    const currentImage = getCurrentImage();
    if (currentImage && isLabelingModeActive) {
        currentImage.savedManualCoords = JSON.parse(JSON.stringify(manualBubbleCoords));
        currentImage.bubbleCoords = JSON.parse(JSON.stringify(manualBubbleCoords));
        console.log(`【保存时同步】更新了 bubbleCoords 以匹配 manualCoords，长度: ${currentImage.bubbleCoords.length}`);
        setHasUnsavedChanges(false);
        currentImage.hasUnsavedChanges = false;
        console.log(`已将 ${manualBubbleCoords.length} 个手动标注框保存到图片 ${currentImageIndex}`);
        return true;
    }
    console.warn("保存手动坐标失败：不在标注模式或没有当前图片。");
    return false;
}

/**
 * 新增：清除当前图片已保存的手动坐标
 */
export function clearSavedManualCoords() {
    const currentImage = getCurrentImage();
    if (currentImage) {
        currentImage.savedManualCoords = null;
        currentImage.hasUnsavedChanges = false;
        setHasUnsavedChanges(false);
        console.log(`已清除图片 ${currentImageIndex} 的已保存手动标注。`);
    }
}

export function deleteImage(index) {
    if (index >= 0 && index < images.length) {
        images.splice(index, 1);
        const wasCurrent = (currentImageIndex === index);

        if (currentImageIndex === index) {
            currentImageIndex = Math.min(index, images.length - 1);
            if (images.length === 0) {
                currentImageIndex = -1;
                setEditModeActive(false);
                setLabelingModeActive(false);
                setManualCoords([], true);
                setHasUnsavedChanges(false);
                setSelectedManualBoxIndex(-1);
            } else {
                const newCurrentImage = getCurrentImage();
                setHasUnsavedChanges(newCurrentImage?.hasUnsavedChanges || false);
                if(wasCurrent && isLabelingModeActive) {
                    setManualCoords(newCurrentImage?.savedManualCoords || [], true);
                    setSelectedManualBoxIndex(-1);
                }
            }
        } else if (currentImageIndex > index) {
            currentImageIndex--;
            const currentImg = getCurrentImage();
            setHasUnsavedChanges(currentImg?.hasUnsavedChanges || false);
=======
// 可以在这里添加更多状态管理函数，例如添加图片、删除图片等
/**
 * 添加图片到状态数组
 * @param {object} imageObject - 要添加的图片对象
 */
export function addImage(imageObject) {
    images.push(imageObject);
}

/**
 * 删除指定索引的图片
 * @param {number} index - 要删除的图片索引
 * @returns {boolean} 是否成功删除
 */
export function deleteImage(index) {
    if (index >= 0 && index < images.length) {
        images.splice(index, 1);
        // 如果删除的是当前图片或之前的图片，需要调整 currentImageIndex
        if (currentImageIndex === index) {
            currentImageIndex = Math.min(index, images.length - 1); // 尝试选中下一个或最后一个
            if (images.length === 0) currentImageIndex = -1; // 如果空了则设为-1
        } else if (currentImageIndex > index) {
            currentImageIndex--; // 如果删除的是前面的，当前索引减一
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
        }
        return true;
    }
    return false;
}

<<<<<<< HEAD
=======
/**
 * 清空所有图片
 */
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
export function clearImages() {
    images = [];
    currentImageIndex = -1;
    bubbleSettings = [];
    initialBubbleSettings = [];
    selectedBubbleIndex = -1;
<<<<<<< HEAD
    setEditModeActive(false);
    setLabelingModeActive(false);
    setManualCoords([], true);
    setHasUnsavedChanges(false);
    setSelectedManualBoxIndex(-1);
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
 * 设置OCR引擎
 * @param {string} engine - OCR引擎('auto', 'manga_ocr', 'paddle_ocr', 'baidu_ocr', 'ai_vision')
 */
export function setOcrEngine(engine) {
    ocrEngine = engine;
}

/**
 * 设置百度OCR版本
 * @param {string} version - 百度OCR版本('standard', 'high_precision')
 */
export function setBaiduOcrVersion(version) {
    baiduOcrVersion = version;
}

/**
 * 设置AI视觉OCR提供商
 * @param {string} provider - 服务提供商(如'siliconflow')
 */
export function setAiVisionProvider(provider) {
    aiVisionProvider = provider;
}

/**
 * 设置AI视觉OCR模型名称
 * @param {string} modelName - 模型名称
 */
export function setAiVisionModelName(modelName) {
    aiVisionModelName = modelName;
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
 * @param {boolean} useJson - 是否使用JSON模式
 * @param {string} [content=null] - 如果提供，则设置当前内容；否则根据模式使用默认值
 */
export function setTranslatePromptMode(useJson, content = null) {
    isTranslateJsonMode = useJson;
    if (content !== null) {
        currentPromptContent = content;
    } else {
        currentPromptContent = useJson ? defaultTranslateJsonPrompt : defaultPromptContent;
    }
    console.log(`状态更新: 漫画翻译JSON模式 -> ${isTranslateJsonMode}, 当前提示词已更新`);
}

/**
 * 设置AI视觉OCR提示词模式和内容
 * @param {boolean} useJson - 是否使用JSON模式
 * @param {string} [content=null] - 如果提供，则设置当前内容；否则根据模式使用默认值
 */
export function setAiVisionOcrPromptMode(useJson, content = null) {
    isAiVisionOcrJsonMode = useJson;
    if (content !== null) {
        aiVisionOcrPrompt = content;
    } else {
        aiVisionOcrPrompt = useJson ? defaultAiVisionOcrJsonPrompt : constants.DEFAULT_AI_VISION_OCR_PROMPT;
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
    console.log(`状态更新: 批量翻译进度状态 -> ${isInProgress ? '进行中' : '已完成'}`);
}

/**
 * 设置翻译服务的RPD限制
 * @param {number} limit - 每分钟请求数 (0表示无限制)
 */
export function setRpdLimitTranslation(limit) {
    const newLimit = parseInt(limit);
    rpdLimitTranslation = isNaN(newLimit) || newLimit < 0 ? 0 : newLimit;
    console.log(`状态更新: 翻译服务 RPD -> ${rpdLimitTranslation}`);
}

/**
 * 设置AI视觉OCR服务的RPD限制
 * @param {number} limit - 每分钟请求数 (0表示无限制)
 */
export function setRpdLimitAiVisionOcr(limit) {
    const newLimit = parseInt(limit);
    rpdLimitAiVisionOcr = isNaN(newLimit) || newLimit < 0 ? 0 : newLimit;
    console.log(`状态更新: AI视觉OCR RPD -> ${rpdLimitAiVisionOcr}`);
=======
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
}
