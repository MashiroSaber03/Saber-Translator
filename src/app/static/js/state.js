// src/app/static/js/state.js

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

// --- 更新状态的函数 ---

/**
 * 更新图片数组
 * @param {Array} newImages - 新的图片数组
 */
export function setImages(newImages) {
    images = newImages;
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

/**
 * 设置当前图片的气泡设置数组
 * @param {Array} settings - 新的气泡设置数组
 */
export function setBubbleSettings(settings) {
    bubbleSettings = settings;
    // 同时更新当前图片的 bubbleSettings 属性
    updateCurrentImageProperty('bubbleSettings', JSON.parse(JSON.stringify(settings)));
}

/**
 * 更新单个气泡的设置
 * @param {number} index - 气泡索引
 * @param {object} newSetting - 新的设置对象
 */
export function updateSingleBubbleSetting(index, newSetting) {
    if (index >= 0 && index < bubbleSettings.length) {
        bubbleSettings[index] = { ...bubbleSettings[index], ...newSetting }; // 合并更新
        // 同时更新当前图片的 bubbleSettings 属性
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
export function setPromptState(current, defaultContent, names) {
    currentPromptContent = current;
    defaultPromptContent = defaultContent;
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
        }
        return true;
    }
    return false;
}

/**
 * 清空所有图片
 */
export function clearImages() {
    images = [];
    currentImageIndex = -1;
    bubbleSettings = [];
    initialBubbleSettings = [];
    selectedBubbleIndex = -1;
}
