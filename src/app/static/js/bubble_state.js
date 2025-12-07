/**
 * bubble_state.js - 统一的气泡状态管理模块
 * 
 * 与后端 BubbleState 数据类对应，用于前后端统一管理气泡渲染参数。
 * 所有气泡相关的操作都应该通过这个模块进行。
 */

import * as constants from './constants.js';

/**
 * 创建默认的气泡状态对象
 * 
 * @param {Object} overrides - 要覆盖的属性
 * @returns {Object} 气泡状态对象
 */
export function createBubbleState(overrides = {}) {
    return {
        // 文本内容
        originalText: "",
        translatedText: "",
        textboxText: "",
        
        // 坐标信息
        coords: [0, 0, 0, 0],
        polygon: [],
        
        // 渲染参数
        fontSize: constants.DEFAULT_FONT_SIZE || 25,
        fontFamily: constants.DEFAULT_FONT_RELATIVE_PATH || "fonts/STSONG.TTF",
        textDirection: constants.DEFAULT_TEXT_DIRECTION || "vertical",
        autoTextDirection: constants.DEFAULT_TEXT_DIRECTION || "vertical",  // 自动检测的排版方向
        textColor: constants.DEFAULT_TEXT_COLOR || "#000000",
        fillColor: constants.DEFAULT_FILL_COLOR || "#FFFFFF",
        rotationAngle: constants.DEFAULT_ROTATION_ANGLE || 0,
        position: { x: 0, y: 0 },
        
        // 描边参数
        strokeEnabled: constants.DEFAULT_STROKE_ENABLED !== undefined ? constants.DEFAULT_STROKE_ENABLED : true,
        strokeColor: constants.DEFAULT_STROKE_COLOR || "#FFFFFF",
        strokeWidth: constants.DEFAULT_STROKE_WIDTH || 3,
        
        // 修复参数
        inpaintMethod: "solid",
        
        // 覆盖默认值
        ...overrides
    };
}

/**
 * 从后端响应创建气泡状态数组
 * 
 * @param {Object} response - 后端 API 响应
 * @param {Object} globalDefaults - 全局默认设置
 * @returns {Array} BubbleState 对象数组
 */
export function createBubbleStatesFromResponse(response, globalDefaults = {}) {
    const { 
        bubble_coords = [], 
        bubble_states = [],
        original_texts = [],
        bubble_texts = [],
        textbox_texts = [],
        bubble_angles = []
    } = response;
    
    // 如果后端返回了 bubble_states，合并全局默认值
    if (bubble_states && bubble_states.length > 0) {
        return bubble_states.map((stateData, i) => {
            // 合并：先用后端数据，但 inpaintMethod 优先使用前端传入的值（用户当前选择）
            return createBubbleState({
                // 先用后端返回的数据
                ...stateData,
                // 确保 text 字段使用正确的名称
                translatedText: stateData.translatedText || stateData.text || bubble_texts[i] || "",
                originalText: stateData.originalText || original_texts[i] || "",
                textboxText: stateData.textboxText || textbox_texts[i] || "",
                // inpaintMethod 优先使用前端传入的值（用户当前选择的修复方式）
                // 因为后端返回的只是默认值 "solid"，不是用户选择的值
                inpaintMethod: globalDefaults.inpaintMethod || stateData.inpaintMethod || 'solid',
            });
        });
    }
    
    // 否则从分散的数据创建
    return bubble_coords.map((coords, i) => {
        // 【重要】如果没有后端返回的 autoTextDirection，根据宽高比计算
        let autoDir = globalDefaults.textDirection || constants.DEFAULT_TEXT_DIRECTION;
        if (coords && coords.length >= 4) {
            const [x1, y1, x2, y2] = coords;
            autoDir = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal';
        }
        
        return createBubbleState({
            coords: coords,
            originalText: original_texts[i] || "",
            translatedText: bubble_texts[i] || "",
            textboxText: textbox_texts[i] || "",
            rotationAngle: bubble_angles[i] || 0,
            // 应用全局默认值
            fontSize: globalDefaults.fontSize || constants.DEFAULT_FONT_SIZE,
            fontFamily: globalDefaults.fontFamily || constants.DEFAULT_FONT_RELATIVE_PATH,
            textDirection: globalDefaults.textDirection || constants.DEFAULT_TEXT_DIRECTION,
            autoTextDirection: autoDir,  // 自动计算的排版方向
            textColor: globalDefaults.textColor || constants.DEFAULT_TEXT_COLOR,
            fillColor: globalDefaults.fillColor || constants.DEFAULT_FILL_COLOR,
            inpaintMethod: globalDefaults.inpaintMethod || 'solid',
            strokeEnabled: globalDefaults.strokeEnabled !== undefined ? globalDefaults.strokeEnabled : constants.DEFAULT_STROKE_ENABLED,
            strokeColor: globalDefaults.strokeColor || constants.DEFAULT_STROKE_COLOR,
            strokeWidth: globalDefaults.strokeWidth || constants.DEFAULT_STROKE_WIDTH,
        });
    });
}

/**
 * 将气泡状态数组转换为 API 请求格式
 * 
 * @param {Array} bubbleStates - BubbleState 对象数组
 * @returns {Array} 用于 API 请求的对象数组
 */
export function bubbleStatesToApiRequest(bubbleStates) {
    return bubbleStates.map(state => ({
        // 文本内容
        originalText: state.originalText || "",
        translatedText: state.translatedText || state.text || "",
        textboxText: state.textboxText || "",
        
        // 坐标信息
        coords: state.coords,
        polygon: state.polygon || [],
        
        // 渲染参数
        fontSize: state.fontSize,
        fontFamily: state.fontFamily,
        textDirection: state.textDirection,
        textColor: state.textColor,
        fillColor: state.fillColor,
        rotationAngle: state.rotationAngle,
        position: state.position || { x: 0, y: 0 },
        
        // 描边参数
        strokeEnabled: state.strokeEnabled,
        strokeColor: state.strokeColor,
        strokeWidth: state.strokeWidth,
        
        // 修复参数
        inpaintMethod: state.inpaintMethod || "solid",
    }));
}

/**
 * 从气泡状态数组提取文本列表
 * 
 * @param {Array} bubbleStates - BubbleState 对象数组
 * @returns {Array} 译文文本数组
 */
export function getTextsFromStates(bubbleStates) {
    return bubbleStates.map(state => state.translatedText || state.text || "");
}

/**
 * 从气泡状态数组提取坐标列表
 * 
 * @param {Array} bubbleStates - BubbleState 对象数组
 * @returns {Array} 坐标数组
 */
export function getCoordsFromStates(bubbleStates) {
    return bubbleStates.map(state => state.coords);
}

/**
 * 更新单个气泡状态
 * 
 * @param {Array} bubbleStates - BubbleState 对象数组
 * @param {number} index - 要更新的气泡索引
 * @param {Object} updates - 要更新的属性
 * @returns {Array} 更新后的 BubbleState 数组
 */
export function updateBubbleState(bubbleStates, index, updates) {
    if (index < 0 || index >= bubbleStates.length) {
        console.warn(`无效的气泡索引: ${index}`);
        return bubbleStates;
    }
    
    const newStates = [...bubbleStates];
    newStates[index] = {
        ...newStates[index],
        ...updates
    };
    return newStates;
}

/**
 * 批量更新所有气泡状态的指定属性
 * 
 * @param {Array} bubbleStates - BubbleState 对象数组
 * @param {Object} updates - 要更新的属性
 * @returns {Array} 更新后的 BubbleState 数组
 */
export function updateAllBubbleStates(bubbleStates, updates) {
    return bubbleStates.map(state => ({
        ...state,
        ...updates
    }));
}

/**
 * 深拷贝气泡状态数组
 * 
 * @param {Array} bubbleStates - BubbleState 对象数组
 * @returns {Array} 深拷贝后的数组
 */
export function cloneBubbleStates(bubbleStates) {
    return JSON.parse(JSON.stringify(bubbleStates));
}

/**
 * 验证气泡状态是否有效
 * 
 * @param {Object} state - 气泡状态对象
 * @returns {boolean} 是否有效
 */
export function isValidBubbleState(state) {
    if (!state || typeof state !== 'object') {
        return false;
    }
    
    // 检查必需的坐标
    if (!state.coords || !Array.isArray(state.coords) || state.coords.length !== 4) {
        return false;
    }
    
    return true;
}

// 导出默认对象
export default {
    createBubbleState,
    createBubbleStatesFromResponse,
    bubbleStatesToApiRequest,
    getTextsFromStates,
    getCoordsFromStates,
    updateBubbleState,
    updateAllBubbleStates,
    cloneBubbleStates,
    isValidBubbleState,
};
