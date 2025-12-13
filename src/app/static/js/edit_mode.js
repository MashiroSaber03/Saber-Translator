import * as state from './state.js';
import * as ui from './ui.js';
import * as api from './api.js';
import * as constants from './constants.js';
import * as main from './main.js';
import { DualImageViewer, PanelDividerController, SidePanelResizer } from './image_viewer.js';
import * as bubbleStateModule from './bubble_state.js';

// 新版编辑模式状态
let dualViewer = null;
let panelDivider = null;
let bottomResizer = null;
let viewMode = 'dual'; // 'dual', 'original', 'translated'
let layoutMode = 'horizontal'; // 'horizontal' (左右布局) | 'vertical' (上下布局)
let customFontPresets = []; // 用户自定义字号预设

// 布局模式存储键
const LAYOUT_MODE_KEY = 'edit_mode_layout';

// ============ 气泡操作状态变量 ============
let isDrawingMode = false;        // 是否处于绘制模式（点击"添加"按钮后激活）
let isDrawingBox = false;         // 是否正在绘制新框
let isDraggingBox = false;        // 是否正在拖拽框
let isResizingBox = false;        // 是否正在调整框大小
let isMiddleButtonDown = false;   // 中键是否按下
let selectedBubbleIndices = [];   // 多选的气泡索引数组
let drawStartX = 0, drawStartY = 0;  // 绘制起始点
let dragStartX = 0, dragStartY = 0;  // 拖拽起始点
let dragBoxInitialX = 0, dragBoxInitialY = 0; // 框的初始位置
let resizeStartX = 0, resizeStartY = 0;  // 调整大小起始点
let resizeHandleType = '';           // 调整手柄类型
let resizeInitialCoords = null;      // 调整前的原始坐标
let currentDrawingRect = null;       // 当前绘制的临时矩形
let activeViewport = null;           // 当前操作的视口
let isRepairingBubble = false;       // 是否正在修复气泡（防抖）

// ============ 旋转状态变量 ============
let isRotatingBox = false;           // 是否正在旋转框
let rotateStartAngle = 0;            // 旋转开始时的角度
let rotateInitialAngle = 0;          // 框的初始旋转角度
let rotateCenterX = 0, rotateCenterY = 0;  // 旋转中心点

// ============ 笔刷状态变量 ============
let brushMode = null;             // 当前笔刷模式: 'repair' | 'restore' | null
let isBrushKeyDown = false;       // 笔刷快捷键是否按下 (R 或 U)
let isBrushPainting = false;      // 是否正在涂抹
let brushSize = 30;               // 笔刷大小（像素）
let brushMinSize = 5;             // 最小笔刷大小
let brushMaxSize = 200;           // 最大笔刷大小
let brushPath = [];               // 笔刷涂抹路径
let brushCanvas = null;           // 笔刷临时画布
let brushCtx = null;              // 笔刷画布上下文

const EDIT_MODE_EVENT_NS = '.editModeUi';
const editModeBoundSelectors = new Set();

function bindEditModeEvent(selector, events, handler) {
    const $elements = $(selector);
    if (!$elements.length) {
        return;
    }
    const eventList = events.split(' ').map(ev => `${ev}${EDIT_MODE_EVENT_NS}`).join(' ');
    $elements.off(eventList);
    $elements.on(eventList, handler);
    editModeBoundSelectors.add(selector);
}

/**
 * 切换编辑模式
 */
export function toggleEditMode() {
    state.setEditModeActive(!state.editModeActive); // 更新状态

    if (state.editModeActive) {
        // --- 进入编辑模式 ---
        const currentImage = state.getCurrentImage();
        if (!currentImage) {
            ui.showGeneralMessage("请先上传图片", "warning");
            state.setEditModeActive(false); // 切换失败，恢复状态
            return;
        }
        
        // 初始化空数组（允许无气泡进入编辑模式）
        if (!currentImage.bubbleCoords) currentImage.bubbleCoords = [];
        if (!currentImage.bubbleTexts) currentImage.bubbleTexts = [];
        if (!currentImage.originalTexts) currentImage.originalTexts = [];
        if (!currentImage.bubbleAngles) currentImage.bubbleAngles = [];

        ui.toggleEditModeUI(true); // 更新 UI
        initBubbleStates(); // 初始化或加载气泡状态
        // 保存初始状态备份
        state.setInitialBubbleStates(JSON.parse(JSON.stringify(state.bubbleStates)));
        console.log("已保存初始气泡状态:", state.initialBubbleStates);

        // 初始化新版编辑模式
        initNewEditMode();
        
        // 确保有干净背景
        ensureCleanBackground();

    } else {
        // --- 退出编辑模式 ---
        ui.toggleEditModeUI(false);
        ui.clearGeneralMessageById("rendering_loading_message");
        cleanupNewEditMode();
        
        // 保存 state.bubbleStates 到 currentImage（switchImage 会从 bubbleStates[0] 读取设置）
        const currentImage = state.getCurrentImage();
        if (currentImage) {
            if (state.bubbleStates.length > 0) {
                currentImage.bubbleStates = JSON.parse(JSON.stringify(state.bubbleStates));
                currentImage.bubbleTexts = state.bubbleStates.map(s => s.translatedText || s.text || "");
                console.log("退出编辑模式，已保存气泡状态");
            } else if (Array.isArray(currentImage.bubbleStates) && currentImage.bubbleStates.length > 0) {
                // 用户删除了所有气泡，需要同步更新为空数组（保持"已处理过"的语义）
                currentImage.bubbleStates = [];
                currentImage.bubbleTexts = [];
                console.log("退出编辑模式，用户已删除所有气泡，bubbleStates 更新为空数组");
            }
            // 如果 currentImage.bubbleStates 原本就是 null/undefined，不改变（保持"从未处理过"的语义）
        }
        
        // 清理状态
        state.setSelectedBubbleIndex(-1);
        state.setInitialBubbleStates([]);
        
        // 刷新主页面显示
        main.switchImage(state.currentImageIndex);
    }
}

/**
 * 初始化或加载当前图片的气泡状态
 * 简化版：直接从 currentImage.bubbleStates 加载，如果不存在则创建默认状态
 * @param {Array} autoDirections - 可选，自动检测的排版方向数组 ('v' 或 'h')
 */
export function initBubbleStates(autoDirections = null) {
    const currentImage = state.getCurrentImage();
    if (!currentImage) {
        console.warn("无法初始化气泡状态：无有效图像");
        state.setBubbleStates([]);
        return;
    }
    
    // 如果没有气泡坐标，初始化为空数组（允许无气泡进入编辑模式）
    if (!currentImage.bubbleCoords || currentImage.bubbleCoords.length === 0) {
        console.log("当前图片没有气泡坐标，初始化为空状态");
        const shouldPreserveNullBubbleStates = currentImage.bubbleStates === null || currentImage.bubbleStates === undefined;
        state.setBubbleStates([]);
        if (shouldPreserveNullBubbleStates) {
            currentImage.bubbleStates = null;
        }
        return;
    }

    // 如果已有保存的状态且数量匹配，直接加载
    if (currentImage.bubbleStates && currentImage.bubbleStates.length === currentImage.bubbleCoords.length) {
        console.log("从图像加载已保存的气泡状态");
        state.setBubbleStates(JSON.parse(JSON.stringify(currentImage.bubbleStates)));
        return;
    }

    // 否则创建默认状态
    console.log("创建新的默认气泡状态");
    const defaults = getDefaultBubbleSettings();
    const newStates = currentImage.bubbleCoords.map((coords, i) => {
        // 【重要】计算每个气泡的自动排版方向
        let autoDir;
        if (autoDirections && i < autoDirections.length) {
            autoDir = autoDirections[i] === 'v' ? 'vertical' : 'horizontal';
        } else {
            // 没有自动检测结果时，根据宽高比判断
            const [x1, y1, x2, y2] = coords;
            autoDir = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal';
        }
        
        return bubbleStateModule.createBubbleState({
            translatedText: currentImage.bubbleTexts?.[i] || "",
            coords: coords,
            ...defaults,
            autoTextDirection: autoDir,  // 保存自动检测的方向
            rotationAngle: currentImage.bubbleAngles?.[i] || 0
        });
    });
    state.setBubbleStates(newStates);
    
    // 同时保存到图片对象
    currentImage.bubbleStates = JSON.parse(JSON.stringify(newStates));
}

/**
 * 获取当前默认的气泡设置（从全局 UI 读取）
 */
function getDefaultBubbleSettings() {
    const layoutDir = $('#layoutDirection').val();
    return {
        fontSize: parseInt($('#fontSize').val()) || state.defaultFontSize,
        fontFamily: $('#fontFamily').val() || state.defaultFontFamily,
        textDirection: layoutDir === 'auto' ? 'vertical' : layoutDir,
        textColor: $('#textColor').val() || state.defaultTextColor,
        fillColor: $('#fillColor').val() || state.defaultFillColor,
        inpaintMethod: 'solid',
        position: { x: 0, y: 0 },
        strokeEnabled: state.strokeEnabled,
        strokeColor: state.strokeColor,
        strokeWidth: state.strokeWidth
    };
}

/**
 * 选择一个气泡进行编辑
 * @param {number} index - 要选择的气泡索引
 */
export function selectBubble(index) {
    if (index < 0 || index >= state.bubbleStates.length) {
        console.warn(`选择气泡失败：无效索引 ${index}`);
        return;
    }
    state.setSelectedBubbleIndex(index);
    
    // 使用新版编辑模式的选择函数
    selectBubbleNew(index);
}

/**
 * 选择上一个气泡
 */
export function selectPrevBubble() {
    if (state.selectedBubbleIndex > 0) {
        selectBubble(state.selectedBubbleIndex - 1);
    }
}

/**
 * 选择下一个气泡
 */
export function selectNextBubble() {
    if (state.selectedBubbleIndex < state.bubbleStates.length - 1) {
        selectBubble(state.selectedBubbleIndex + 1);
    }
}

/**
 * 更新当前选中气泡的文本内容并立即触发渲染
 * @param {string} newText - 新的文本内容
 */
export function updateBubbleText(newText) {
    const index = state.selectedBubbleIndex;
    if (index < 0) return;
    
    state.updateSingleBubbleState(index, { translatedText: newText });
    renderBubblePreview(index); // 立即触发渲染预览
}

/**
 * 渲染单个气泡的预览（通过重新渲染整个图像实现）
 * 简化版：直接调用 reRenderFullImage，不做复杂判断
 * @param {number} bubbleIndex - 要预览的气泡索引（仅用于日志）
 */
export function renderBubblePreview(bubbleIndex) {
    if (bubbleIndex < 0 || bubbleIndex >= state.bubbleStates.length) return;
    console.log(`渲染预览 - 气泡 ${bubbleIndex}`);
    reRenderFullImage();
}

/**
 * 重新渲染整个图像
 * @param {boolean} [fromAutoToManual=false] - (保留) 是否是从自动字号切换到手动字号,用于后端特殊处理
 * @param {boolean} [silentMode=false] - 是否静默模式，不更新界面显示
 * @param {boolean} [useAutoFontSize=false] - 是否为每个气泡自动计算字号
 * @returns {Promise<void>} - 在渲染成功时 resolve，失败时 reject
 */
export function reRenderFullImage(fromAutoToManual = false, silentMode = false, useAutoFontSize = false) {
    return new Promise(async (resolve, reject) => { // 改为 async 以便使用 await
        const imageIndex = state.currentImageIndex;
        if (imageIndex < 0 || imageIndex >= state.images.length) {
            if (!silentMode) {
                ui.showGeneralMessage("无法重新渲染，当前没有有效图片", "error");
            }
            reject(new Error("无法重新渲染：无有效图片"));
            return;
        }
        const currentImage = state.images[imageIndex];
        if (!currentImage || (!currentImage.translatedDataURL && !currentImage.originalDataURL)) {
            console.error("无法重新渲染：缺少必要数据");
            if (!silentMode) {
                ui.showGeneralMessage("无法重新渲染，缺少图像或气泡数据", "error");
            }
            reject(new Error("无法重新渲染：缺少必要数据"));
            return;
        }
        if (!currentImage.bubbleCoords || currentImage.bubbleCoords.length === 0) {
            console.log("reRenderFullImage: 当前图片没有气泡坐标，跳过重新渲染。");
            resolve();
            return;
        }

        const renderToken = Symbol('render');
        currentImage._activeRenderToken = renderToken;
        const liveBubbleStates = state.editModeActive && state.bubbleStates
            ? JSON.parse(JSON.stringify(state.bubbleStates))
            : null;

        // 使用固定消息ID，确保相同操作只显示一条消息
        const loadingMessageId = "rendering_loading_message";
        if (!silentMode) {
            ui.showGeneralMessage("重新渲染中，请不要在重渲染时快速切换图片", "info", false, 0, loadingMessageId);
        }

        let preFilledBackgroundBase64 = null; // 用于存储前端预填充后的背景
        let backendShouldInpaint = false; // 后端是否需要做任何背景修复

        try {
            // 1. 确定最原始的干净背景 (original 或 cleanImageData)
            let pristineBackgroundSrc;
            if (currentImage.cleanImageData) {
                pristineBackgroundSrc = 'data:image/png;base64,' + currentImage.cleanImageData;
                console.log("reRenderFullImage: 使用 cleanImageData 作为原始干净背景。");
            } else if (currentImage.originalDataURL) {
                pristineBackgroundSrc = currentImage.originalDataURL;
                console.log("reRenderFullImage: 使用 originalDataURL 作为原始干净背景。");
                backendShouldInpaint = true; // 如果只有原图，后端可能需要修复（除非所有气泡都有独立颜色）
            } else {
                throw new Error("无法找到原始干净背景用于填充。");
            }

            const pristineBgImage = await main.loadImage(pristineBackgroundSrc); // main.js 需要导出 loadImage
            const canvas = document.createElement('canvas');
            canvas.width = pristineBgImage.naturalWidth;
            canvas.height = pristineBgImage.naturalHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(pristineBgImage, 0, 0);

            // 2. 背景处理：不再自动填充 fillColor
            // - 纯色填充只在用户点击"修复"按钮时执行，会更新 cleanImageData
            // - 这里直接使用当前背景（cleanImageData 或原图），不自动填充
            
            if (currentImage.cleanImageData) {
                // 有已修复的背景（LAMA 或纯色填充的结果），直接使用
                console.log("reRenderFullImage: 使用已有的 cleanImageData 作为背景。");
                preFilledBackgroundBase64 = currentImage.cleanImageData;
            } else {
                // 没有 cleanImageData，使用当前 canvas（原图）
                console.log("reRenderFullImage: 没有 cleanImageData，使用原图作为背景。");
                preFilledBackgroundBase64 = canvas.toDataURL('image/png').split(',')[1];
            }

            backendShouldInpaint = false;

        } catch (error) {
            console.error("前端预填充背景时出错:", error);
            // 如果预填充失败，回退到让后端处理，但这样就无法实现独立气泡颜色
            // 或者直接报错并停止
            ui.showGeneralMessage("预处理背景失败，无法应用独立填充色。", "error");
            ui.clearGeneralMessageById(loadingMessageId);
            reject(error);
            return;
        }

        // 简化：直接使用 state.bubbleStates 或 currentImage.bubbleStates
        const bubbleStates = liveBubbleStates || currentImage.bubbleStates || [];
        
        if (bubbleStates.length !== currentImage.bubbleCoords.length) {
            console.error("气泡状态数量与坐标不匹配");
            ui.clearGeneralMessageById(loadingMessageId);
            reject(new Error("气泡状态数量不匹配"));
            return;
        }

        // 直接从 bubbleStates 构建 API 请求数据，不需要中间转换
        const layoutDir = $('#layoutDirection').val();
        const isAutoLayout = layoutDir === 'auto';
        
        // 【重要】当用户选择 'auto' 时，使用每个气泡的 autoTextDirection
        // 否则使用用户选择的全局方向
        const getEffectiveDirection = (s, index) => {
            if (isAutoLayout) {
                // 自动模式：优先使用气泡自己检测到的方向
                // 回退逻辑：autoTextDirection → textDirection → 根据宽高比计算 → 默认 'vertical'
                if (s.autoTextDirection && s.autoTextDirection !== '') {
                    return s.autoTextDirection;
                }
                if (s.textDirection && s.textDirection !== '' && s.textDirection !== 'auto') {
                    return s.textDirection;
                }
                // 最后根据坐标宽高比计算
                if (currentImage.bubbleCoords && currentImage.bubbleCoords[index]) {
                    const [x1, y1, x2, y2] = currentImage.bubbleCoords[index];
                    return (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal';
                }
                return 'vertical';
            } else {
                // 手动模式：使用用户选择的全局方向
                return layoutDir;
            }
        };
        
        const bubbleStatesForApi = bubbleStates.map((s, i) => ({
            translatedText: s.translatedText || s.text || "",
            coords: currentImage.bubbleCoords[i],
            fontSize: s.fontSize || state.defaultFontSize,
            fontFamily: s.fontFamily || state.defaultFontFamily,
            textDirection: getEffectiveDirection(s, i),  // 传入索引以支持宽高比回退计算
            textColor: s.textColor || state.defaultTextColor,
            rotationAngle: s.rotationAngle || 0,
            position: s.position || { x: 0, y: 0 },
            strokeEnabled: s.strokeEnabled !== undefined ? s.strokeEnabled : state.strokeEnabled,
            strokeColor: s.strokeColor || state.strokeColor,
            strokeWidth: s.strokeWidth !== undefined ? s.strokeWidth : state.strokeWidth,
        }));

        const currentTexts = bubbleStatesForApi.map(s => s.translatedText);

        const data = {
            clean_image: preFilledBackgroundBase64,
            bubble_texts: currentTexts,
            bubble_coords: currentImage.bubbleCoords,
            fontSize: $('#fontSize').val(),
            fontFamily: $('#fontFamily').val() || state.defaultFontFamily,
            textDirection: layoutDir === 'auto' ? 'vertical' : layoutDir,
            textColor: $('#textColor').val() || state.defaultTextColor,
            rotationAngle: 0,
            bubble_states: bubbleStatesForApi,
            use_individual_styles: true,
            use_inpainting: false,
            use_lama: false,
            fillColor: null,
            is_font_style_change: true,
            strokeEnabled: state.strokeEnabled,
            strokeColor: state.strokeColor,
            strokeWidth: state.strokeWidth,
            autoFontSize: useAutoFontSize  // 传递自动字号参数
        };

        api.reRenderImageApi(data)
            .then(response => {
                if (!silentMode) {
                    ui.clearGeneralMessageById(loadingMessageId);
                }
                if (response.rendered_image) {
                    const latestImage = state.images[imageIndex];
                    if (!latestImage || latestImage !== currentImage || latestImage._activeRenderToken !== renderToken) {
                        resolve();
                        return;
                    }

                    const translatedDataUrl = 'data:image/png;base64,' + response.rendered_image;
                    state.updateImagePropertyByIndex(imageIndex, 'translatedDataURL', translatedDataUrl);

                    // 注意：不在每次渲染时更新 cleanImageData
                    // cleanImageData 应该保持为"原始干净背景"（无填充色），只在用户手动修复背景时才更新
                    // 否则移动气泡框时，旧位置的填充色会被固化到 cleanImageData 中无法清除

                    state.updateImagePropertyByIndex(imageIndex, 'bubbleTexts', currentTexts);
                    
                    // 更新气泡状态
                    if (response.bubble_states && Array.isArray(response.bubble_states)) {
                        const latestImageForStyles = state.images[imageIndex];
                        if (latestImageForStyles && !latestImageForStyles.bubbleStates) {
                            // 如果前端没有 bubbleStates（首次渲染），则使用后端返回的值
                            latestImageForStyles.bubbleStates = response.bubble_states;
                            console.log(`[reRenderFullImage] 首次设置 ${response.bubble_states.length} 个气泡状态`);
                        } else if (latestImageForStyles && latestImageForStyles.bubbleStates && useAutoFontSize) {
                            // 如果使用自动字号，将后端计算的 fontSize 合并到前端状态
                            response.bubble_states.forEach((serverState, i) => {
                                if (latestImageForStyles.bubbleStates[i] && serverState.fontSize) {
                                    latestImageForStyles.bubbleStates[i].fontSize = serverState.fontSize;
                                }
                            });
                            // 同步到编辑模式的实时状态
                            if (state.editModeActive && state.bubbleStates) {
                                response.bubble_states.forEach((serverState, i) => {
                                    if (state.bubbleStates[i] && serverState.fontSize) {
                                        state.bubbleStates[i].fontSize = serverState.fontSize;
                                    }
                                });
                            }
                            console.log(`[reRenderFullImage] 自动字号: 已更新 ${response.bubble_states.length} 个气泡的字号`);
                            // 刷新编辑面板显示当前选中气泡的字号
                            if (state.editModeActive && state.selectedBubbleIndex >= 0) {
                                selectBubbleNew(state.selectedBubbleIndex);
                            }
                        }
                    }
                    
                    if (!silentMode && state.currentImageIndex === imageIndex) {
                        const translatedURL = translatedDataUrl;
                        ui.updateTranslatedImage(translatedURL);
                        updateTranslatedImageNew(translatedURL);
                        
                        const translatedImgEl = $('#translatedImageDisplay')[0];
                        const finalizeUpdate = () => {
                            ui.updateBubbleHighlight(state.selectedBubbleIndex);
                            resolve();
                        };
                        if (translatedImgEl && translatedImgEl.complete) {
                            finalizeUpdate();
                        } else {
                            $('#translatedImageDisplay').one('load', finalizeUpdate);
                        }
                    } else {
                        resolve();
                    }
                } else {
                    throw new Error("渲染 API 未返回图像数据");
                }
            })
            .catch(error => {
                ui.clearGeneralMessageById(loadingMessageId);
                if (!silentMode && state.currentImageIndex === imageIndex) {
                    ui.showGeneralMessage(`重新渲染失败: ${error.message}`, "error");
                    ui.updateBubbleHighlight(state.selectedBubbleIndex);
                }
                reject(error);
            });
    });
}


/**
 * 将当前选中气泡的样式应用到所有气泡
 */
export function applySettingsToAllBubbles() {
    const index = state.selectedBubbleIndex;
    if (index < 0) return;

    const currentSetting = state.bubbleStates[index];
    const newSettings = state.bubbleStates.map(setting => ({
        ...setting, // 保留 text, position, inpaintMethod, fillColor（修复相关参数独立保留）
        fontSize: currentSetting.fontSize,
        fontFamily: currentSetting.fontFamily,
        textDirection: currentSetting.textDirection,
        textColor: currentSetting.textColor,
        rotationAngle: currentSetting.rotationAngle,
        strokeEnabled: currentSetting.strokeEnabled,
        strokeColor: currentSetting.strokeColor,
        strokeWidth: currentSetting.strokeWidth
    }));
    state.setBubbleStates(newSettings); // 更新状态
    reRenderFullImage(); // 重新渲染整个图像
    ui.showGeneralMessage("样式已应用到所有气泡", "success", false, 2000);
}

/**
 * 重置当前选中气泡的设置为初始状态
 */
export function resetCurrentBubble() {
    const index = state.selectedBubbleIndex;
    if (index < 0 || !state.initialBubbleStates || index >= state.initialBubbleStates.length) {
        ui.showGeneralMessage("无法重置：未找到初始状态", "warning");
        return;
    }
    const initialState = state.initialBubbleStates[index];
    // 深拷贝恢复状态
    state.updateSingleBubbleState(index, JSON.parse(JSON.stringify(initialState)));
    // 更新新版编辑区显示
    selectBubbleNew(index);
    reRenderFullImage(); // 重新渲染
    ui.showGeneralMessage(`气泡 ${index + 1} 已重置`, "info", false, 2000);
}

/**
 * 调整当前选中气泡的位置
 * @param {'moveUp' | 'moveDown' | 'moveLeft' | 'moveRight'} direction - 移动方向
 */
export function adjustPosition(direction) {
    const index = state.selectedBubbleIndex;
    if (index < 0) return;

    const currentSetting = state.bubbleStates[index];
    const position = { ...(currentSetting.position || { x: 0, y: 0 }) }; // 创建副本或默认值
    const step = 2;
    const limit = getPositionLimit(); // 位置偏移限制

    switch (direction) {
        case 'moveUp':    position.y = Math.max(position.y - step, -limit); break;
        case 'moveDown':  position.y = Math.min(position.y + step, limit); break;
        case 'moveLeft':  position.x = Math.max(position.x - step, -limit); break;
        case 'moveRight': position.x = Math.min(position.x + step, limit); break;
    }

    state.updateSingleBubbleState(index, { position: position });
    // 更新新版编辑区的位置显示
    $('#positionXValue').text(position.x);
    $('#positionYValue').text(position.y);
    triggerDelayedPreview(index); // 延迟渲染
}

/**
 * 重置当前选中气泡的位置
 */
export function resetPosition() {
    const index = state.selectedBubbleIndex;
    if (index < 0) return;
    state.updateSingleBubbleState(index, { position: { x: 0, y: 0 } });
    $('#positionXValue').text(0);
    $('#positionYValue').text(0);
    reRenderFullImage(); // 立即渲染
}

// --- 辅助函数 ---

/**
 * 确保有干净的背景图，如果没有则尝试生成（仅用于纯色填充模式）
 */
function ensureCleanBackground() {
    const currentImage = state.getCurrentImage();
    if (!currentImage || currentImage.cleanImageData || currentImage._tempCleanImage) {
        return;
    }
    const repairMethod = $('#useInpainting').val();
    if (repairMethod !== 'false') return;

    console.log("尝试为纯色填充模式创建临时干净背景");
    if (currentImage.bubbleCoords && currentImage.translatedDataURL) {
        try {
            const img = new Image();
            img.onload = function() {
                try {
                    const canvas = document.createElement('canvas');
                    canvas.width = img.naturalWidth;
                    canvas.height = img.naturalHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    ctx.fillStyle = $('#fillColor').val() || state.defaultFillColor;
                    for (const [x1, y1, x2, y2] of currentImage.bubbleCoords) {
                        ctx.fillRect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
                    }
                    const tempCleanImage = canvas.toDataURL('image/png').split(',')[1];
                    currentImage._tempCleanImage = tempCleanImage;
                    console.log("成功创建临时干净背景 (纯色填充)");
                } catch (e) { console.error("创建临时干净背景 Canvas 操作失败:", e); }
            };
            img.onerror = () => console.error("加载图像以创建临时背景失败");
            img.src = currentImage.translatedDataURL;
        } catch (e) { console.error("创建临时干净背景失败:", e); }
    } else { console.warn("缺少数据无法创建临时干净背景"); }
}

function getPositionLimit() {
    const imgEl = document.getElementById('originalImageDisplay');
    const width = imgEl?.naturalWidth || 1000;
    const height = imgEl?.naturalHeight || 1000;
    const maxDimension = Math.max(width, height);
    return Math.max(50, Math.round(maxDimension * 0.1));
}

function extractBubbleImageDataURL(imageSrc, coords) {
    return new Promise((resolve, reject) => {
        if (!imageSrc || !coords || coords.length < 4) {
            reject(new Error("无法裁剪气泡：缺少图像或坐标"));
            return;
        }
        const img = new Image();
        img.onload = () => {
            try {
                const [x1, y1, x2, y2] = coords;
                const canvas = document.createElement('canvas');
                canvas.width = Math.max(1, x2 - x1);
                canvas.height = Math.max(1, y2 - y1);
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, x1, y1, canvas.width, canvas.height, 0, 0, canvas.width, canvas.height);
                resolve(canvas.toDataURL('image/png'));
            } catch (err) {
                reject(err);
            }
        };
        img.onerror = () => reject(new Error("加载原图失败，无法裁剪气泡"));
        img.src = imageSrc;
    });
}

// 用于位置和旋转调整的延迟渲染计时器
let previewTimer = null;
let isRenderingPreview = false; // 渲染状态锁，防止竞态条件
/**
 * 触发带延迟的预览渲染
 * @param {number} bubbleIndex - 气泡索引
 */
function triggerDelayedPreview(bubbleIndex) {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(async () => {
        if (isRenderingPreview) {
            console.log(`跳过渲染气泡 ${bubbleIndex}，上一次渲染仍在进行中`);
            return;
        }
        console.log(`准备渲染气泡 ${bubbleIndex} 的预览`);
        isRenderingPreview = true;
        try {
            await renderBubblePreview(bubbleIndex);
        } finally {
            isRenderingPreview = false;
        }
    }, 150); // 150ms 延迟，更快的实时响应
}

/**
 * 退出编辑模式
 */
export function exitEditMode() {
    if (state.editModeActive) {
        toggleEditMode(); // 这会处理所有清理工作，然后设置 editModeActive = false
    }
}

/**
 * 退出编辑模式，但不触发重渲染
 * 用于切换图片等场景，避免不必要的重渲染
 */
export function exitEditModeWithoutRender() {
    if (!state.editModeActive) return;
    
    ui.toggleEditModeUI(false);
    
    // 保存气泡状态（switchImage 会从 bubbleStates[0] 读取设置）
    const currentImage = state.getCurrentImage();
    if (currentImage) {
        if (state.bubbleStates.length > 0) {
            currentImage.bubbleStates = JSON.parse(JSON.stringify(state.bubbleStates));
            currentImage.bubbleTexts = state.bubbleStates.map(s => s.translatedText || s.text || "");
        } else if (Array.isArray(currentImage.bubbleStates) && currentImage.bubbleStates.length > 0) {
            // 用户删除了所有气泡，需要同步更新为空数组（保持"已处理过"的语义）
            currentImage.bubbleStates = [];
            currentImage.bubbleTexts = [];
            console.log("退出编辑模式（无重渲染），用户已删除所有气泡");
        }
    }
    
    cleanupNewEditMode();
    state.setEditModeActive(false);
    state.setSelectedBubbleIndex(-1);
    state.setInitialBubbleStates([]);
    console.log("已退出编辑模式（无重渲染）");
}


// ============================================================
// ============ 新版编辑模式功能 ============
// ============================================================

/**
 * 初始化新版编辑模式
 */
export function initNewEditMode() {
    const currentImage = state.getCurrentImage();
    if (!currentImage) return;
    
    console.log("初始化新版编辑模式...");
    
    // 初始化双图浏览器
    if (!dualViewer) {
        dualViewer = new DualImageViewer({
            syncEnabled: true,
            onScaleChange: (scale) => {
                $('#zoomLevel').text(Math.round(scale * 100) + '%');
            }
        });
        dualViewer.init();
    }
    
    // 初始化面板分隔条
    if (!panelDivider) {
        panelDivider = new PanelDividerController('panelDivider', 'originalPanel', 'translatedPanel');
    }
    
    // 初始化右侧面板宽度调整
    if (!bottomResizer) {
        bottomResizer = new SidePanelResizer('editPanelResizer', 'editPanelContainer', {
            minWidth: 480,
            maxWidth: window.innerWidth * 0.7
        });
    }
    
    // 加载自定义字号预设
    loadCustomFontPresets();
    
    // 渲染字号预设按钮
    renderFontPresets();
    
    // 加载并应用保存的布局模式
    loadLayoutMode();
    
    // 加载图片到双图浏览器
    loadImagesToViewer();
    
    // 初始化字体选择器
    initFontSelector();
    
    // 绑定新版编辑模式事件
    bindNewEditModeEvents();
    
    // 初始化50音键盘
    initKanaKeyboard();
    
    // 更新导航器
    updateImageNavigator();
    updateBubbleNavigator();
    
    // 选择第一个气泡
    if (state.bubbleStates.length > 0) {
        selectBubbleNew(0);
    }
    
    console.log("新版编辑模式初始化完成");
}

/**
 * 清理新版编辑模式资源
 */
export function cleanupNewEditMode() {
    // 清理气泡操作状态
    cleanupBubbleOperationState();
    unbindBubbleOperationEvents();
    exitDrawingMode();
    
    // 关闭50音键盘
    $('#kanaKeyboard').hide();
    $('#toggleKanaKeyboard').removeClass('active');
    
    if (dualViewer) {
        dualViewer.destroy();
        dualViewer = null;
    }
    panelDivider = null;
    bottomResizer = null;
    
    // 解绑事件
    unbindNewEditModeEvents();
}

/**
 * 加载图片到双图浏览器
 * @param {boolean} resetView - 是否重置视图（fitToScreen），默认true
 */
function loadImagesToViewer(resetView = true) {
    const currentImage = state.getCurrentImage();
    if (!currentImage) return;
    
    const originalImg = $('#originalImageDisplay');
    const translatedImg = $('#translatedImageDisplayNew');
    const originalWrapper = $('#originalCanvasWrapper');
    const translatedWrapper = $('#translatedCanvasWrapper');
    
    // 切换时先隐藏图片，避免看到偏移过程
    if (resetView) {
        originalWrapper.css('opacity', '0');
        translatedWrapper.css('opacity', '0');
        // 立即重置 transform 到中心位置
        if (dualViewer) {
            dualViewer.resetTransform();
        }
    }
    
    // 图片加载完成后的处理
    const onImageLoaded = () => {
        if (dualViewer && resetView) {
            dualViewer.fitToScreen();
        }
        updateBubbleHighlightsNew();
        // 显示图片
        originalWrapper.css('opacity', '1');
        translatedWrapper.css('opacity', '1');
    };
    
    // 设置原图
    if (currentImage.originalDataURL) {
        const img = originalImg[0];
        if (img.src === currentImage.originalDataURL) {
            onImageLoaded();
        } else {
            img.onload = onImageLoaded;
            originalImg.attr('src', currentImage.originalDataURL);
        }
    }
    
    // 设置翻译图
    if (currentImage.translatedDataURL) {
        translatedImg.attr('src', currentImage.translatedDataURL);
    } else if (currentImage.originalDataURL) {
        translatedImg.attr('src', currentImage.originalDataURL);
    }
}

/**
 * 加载自定义字号预设
 */
function loadCustomFontPresets() {
    try {
        const saved = localStorage.getItem(constants.FONT_SIZE_CUSTOM_PRESETS_KEY);
        if (saved) {
            customFontPresets = JSON.parse(saved);
        }
    } catch (e) {
        console.warn('加载自定义字号预设失败:', e);
        customFontPresets = [];
    }
}

/**
 * 保存自定义字号预设
 */
function saveCustomFontPresets() {
    try {
        localStorage.setItem(constants.FONT_SIZE_CUSTOM_PRESETS_KEY, JSON.stringify(customFontPresets));
    } catch (e) {
        console.warn('保存自定义字号预设失败:', e);
    }
}

/**
 * 渲染字号预设按钮
 */
function renderFontPresets() {
    const container = $('#fontSizePresets');
    container.empty();
    
    // 默认预设
    constants.FONT_SIZE_PRESETS.forEach(size => {
        const btn = $(`<button class="font-preset-btn" data-size="${size}">${size}</button>`);
        btn.on('click', () => applyFontSizePreset(size));
        container.append(btn);
    });
    
    // 自定义预设
    customFontPresets.forEach(size => {
        const btn = $(`<button class="font-preset-btn custom" data-size="${size}" title="自定义预设，右键删除">${size}</button>`);
        btn.on('click', () => applyFontSizePreset(size));
        btn.on('contextmenu', (e) => {
            e.preventDefault();
            removeCustomFontPreset(size);
        });
        container.append(btn);
    });
    
    // 添加按钮
    const addBtn = $('<button class="add-preset-btn" title="添加当前字号为预设">+</button>');
    addBtn.on('click', addCurrentFontAsPreset);
    container.append(addBtn);
}

/**
 * 应用字号预设
 */
function applyFontSizePreset(size) {
    $('#fontSizeSlider').val(size);
    $('#fontSizeInput').val(size);
    updateSliderBackground(size);
    
    // 更新当前气泡设置
    const index = state.selectedBubbleIndex;
    if (index >= 0) {
        state.updateSingleBubbleState(index, { fontSize: size });
        triggerDelayedPreview(index);
    }
    
    // 高亮当前选中的预设
    $('.font-preset-btn').removeClass('active');
    $(`.font-preset-btn[data-size="${size}"]`).addClass('active');
}

/**
 * 添加当前字号为预设
 */
function addCurrentFontAsPreset() {
    const size = parseInt($('#fontSizeInput').val());
    if (isNaN(size) || size < 10 || size > 100) {
        ui.showGeneralMessage('请输入有效的字号 (10-100)', 'warning');
        return;
    }
    
    // 检查是否已存在
    if (constants.FONT_SIZE_PRESETS.includes(size) || customFontPresets.includes(size)) {
        ui.showGeneralMessage('该字号预设已存在', 'info');
        return;
    }
    
    customFontPresets.push(size);
    customFontPresets.sort((a, b) => a - b);
    saveCustomFontPresets();
    renderFontPresets();
    ui.showGeneralMessage(`已添加字号预设: ${size}`, 'success', false, 2000);
}

/**
 * 删除自定义字号预设
 */
function removeCustomFontPreset(size) {
    const idx = customFontPresets.indexOf(size);
    if (idx > -1) {
        customFontPresets.splice(idx, 1);
        saveCustomFontPresets();
        renderFontPresets();
        ui.showGeneralMessage(`已删除字号预设: ${size}`, 'info', false, 2000);
    }
}

/**
 * 更新滑块背景
 */
function updateSliderBackground(value) {
    const slider = $('#fontSizeSlider');
    const min = parseInt(slider.attr('min'));
    const max = parseInt(slider.attr('max'));
    const percent = ((value - min) / (max - min)) * 100;
    slider.css('background', `linear-gradient(to right, #3498db ${percent}%, #dee2e6 ${percent}%)`);
}

/**
 * 初始化字体选择器
 */
function initFontSelector() {
    const fontSelect = $('#fontFamilyNew');
    const originalSelect = $('#fontFamily');
    
    // 如果 fontFamilyNew 选项已经足够多（已被 loadFontList 初始化），只需设置选中值
    // 否则从 fontFamily 复制选项
    if (fontSelect.find('option:not([data-custom])').length === 0 && originalSelect.length) {
        originalSelect.find('option:not([data-custom])').each(function() {
            fontSelect.append($(this).clone());
        });
    }
    
    // 设置当前选中的字体
    const currentFont = state.bubbleStates.length > 0 
        ? state.bubbleStates[0].fontFamily 
        : originalSelect.val();
    
    if (currentFont && fontSelect.find(`option[value="${currentFont}"]`).length > 0) {
        fontSelect.val(currentFont);
    } else if (originalSelect.val()) {
        // 如果当前字体不在选项中，使用全局选择器的值
        fontSelect.val(originalSelect.val());
    }
}

/**
 * 绑定新版编辑模式事件
 */
function bindNewEditModeEvents() {
    editModeBoundSelectors.clear();

    bindEditModeEvent('#prevImageBtn', 'click', () => navigateImage(-1));
    bindEditModeEvent('#nextImageBtn', 'click', () => navigateImage(1));
    bindEditModeEvent('#toggleThumbnails', 'click', toggleThumbnailsPanel);
    bindEditModeEvent('#imageIndicator', 'click', toggleThumbnailsPanel);

    bindEditModeEvent('#prevBubbleBtn', 'click', () => selectPrevBubbleNew());
    bindEditModeEvent('#nextBubbleBtn', 'click', () => selectNextBubbleNew());

    bindEditModeEvent('#layoutToggle', 'click', toggleLayoutMode);
    bindEditModeEvent('#viewModeToggle', 'click', toggleViewMode);
    bindEditModeEvent('#syncViewToggle', 'click', toggleSyncView);
    bindEditModeEvent('#fitToScreen', 'click', () => dualViewer?.fitToScreen());
    bindEditModeEvent('#zoomIn', 'click', () => dualViewer?.zoomIn());
    bindEditModeEvent('#zoomOut', 'click', () => dualViewer?.zoomOut());
    bindEditModeEvent('#resetZoom', 'click', () => dualViewer?.resetZoom());

    bindEditModeEvent('.panel-toggle', 'click', function() {
        const panel = $(this).data('panel');
        togglePanel(panel);
    });

    bindEditModeEvent('#applyAndNext', 'click', applyAndNextImage);
    bindEditModeEvent('#exitEditModeBtn', 'click', () => toggleEditMode());

    bindEditModeEvent('#translatedTextEditor', 'input', handleTranslatedTextChange);
    bindEditModeEvent('#originalTextEditor', 'input', handleOriginalTextChange);
    bindEditModeEvent('#applyTextBtn', 'click', applyCurrentText);

    bindEditModeEvent('.copy-btn', 'click', function() {
        const targetId = $(this).data('target');
        const text = $(`#${targetId}`).val();
        navigator.clipboard.writeText(text).then(() => {
            ui.showGeneralMessage('已复制到剪贴板', 'success', false, 1500);
        });
    });

    // === Office风格工具栏事件 ===
    
    // 字号输入框变化 - 分离 change 和 input 事件
    // change 事件（失去焦点或回车）：校验并修正值
    bindEditModeEvent('#fontSizeInput', 'change', function() {
        const size = Math.min(100, Math.max(10, parseInt($(this).val()) || 24));
        $(this).val(size);
        $('#fontSizeSlider').val(size);
        handleFontSizeChange(size);
    });
    // input 事件：实时预览但不修改输入框内容，避免用户输入时跳转
    bindEditModeEvent('#fontSizeInput', 'input', function() {
        const val = parseInt($(this).val());
        if (!isNaN(val) && val >= 10 && val <= 100) {
            $('#fontSizeSlider').val(val);
            handleFontSizeChange(val);
        }
    });
    
    // 字号增减按钮
    bindEditModeEvent('#fontSizeUp', 'click', function() {
        const currentSize = parseInt($('#fontSizeInput').val()) || 24;
        const newSize = Math.min(100, currentSize + 2);
        $('#fontSizeInput').val(newSize);
        $('#fontSizeSlider').val(newSize);
        handleFontSizeChange(newSize);
    });
    
    bindEditModeEvent('#fontSizeDown', 'click', function() {
        const currentSize = parseInt($('#fontSizeInput').val()) || 24;
        const newSize = Math.max(10, currentSize - 2);
        $('#fontSizeInput').val(newSize);
        $('#fontSizeSlider').val(newSize);
        handleFontSizeChange(newSize);
    });
    
    // 隐藏的滑块保持兼容
    bindEditModeEvent('#fontSizeSlider', 'input', function() {
        const size = parseInt($(this).val());
        $('#fontSizeInput').val(size);
        handleFontSizeChange(size);
    });

    // 字体变化
    bindEditModeEvent('#fontFamilyNew', 'change', handleStyleChange);
    
    // 排版方向按钮
    bindEditModeEvent('#directionVerticalBtn', 'click', function() {
        $('#textDirectionNew').val('vertical').trigger('change');
        $('#directionVerticalBtn').attr('data-active', 'true').addClass('active');
        $('#directionHorizontalBtn').attr('data-active', 'false').removeClass('active');
    });
    
    bindEditModeEvent('#directionHorizontalBtn', 'click', function() {
        $('#textDirectionNew').val('horizontal').trigger('change');
        $('#directionHorizontalBtn').attr('data-active', 'true').addClass('active');
        $('#directionVerticalBtn').attr('data-active', 'false').removeClass('active');
    });
    
    bindEditModeEvent('#textDirectionNew', 'change', handleStyleChange);
    
    // 颜色选择器按钮
    bindEditModeEvent('#textColorBtn', 'click', function() {
        $('#textColorNew').click();
    });
    
    bindEditModeEvent('#fillColorBtn', 'click', function() {
        $('#fillColorNew').click();
    });
    
    bindEditModeEvent('#strokeColorBtn', 'click', function() {
        $('#strokeColorNew').click();
    });
    
    // 颜色变化 - 更新指示器
    bindEditModeEvent('#textColorNew', 'input change', function() {
        $('#textColorIndicator').css('background', $(this).val());
        handleStyleChange();
    });
    
    bindEditModeEvent('#fillColorNew', 'input change', function() {
        $('#fillColorIndicator').css('background', $(this).val());
        handleStyleChange();
    });
    
    // 背景修复方式选择器
    bindEditModeEvent('#bubbleInpaintMethodNew', 'change', function() {
        const method = $(this).val();
        // 根据方法显示/隐藏颜色选择器
        if (method === 'solid') {
            $('#solidColorOptionsNew').removeClass('hidden');
        } else {
            $('#solidColorOptionsNew').addClass('hidden');
        }
        handleStyleChange();
    });
    
    bindEditModeEvent('#strokeColorNew', 'input change', function() {
        $('#strokeColorIndicator').css('background', $(this).val());
        handleStyleChange();
    });
    
    // 描边开关按钮
    bindEditModeEvent('#strokeToggleBtn', 'click', function() {
        const isEnabled = $('#strokeEnabledNew').is(':checked');
        const newState = !isEnabled;
        $('#strokeEnabledNew').prop('checked', newState);
        $(this).attr('data-active', newState ? 'true' : 'false');
        $(this).toggleClass('active', newState);
        $('.toolbar-stroke-options').toggleClass('hidden', !newState);
        handleStyleChange();
    });

    bindEditModeEvent('#strokeEnabledNew', 'change', function() {
        const enabled = $(this).is(':checked');
        $('#strokeToggleBtn').attr('data-active', enabled ? 'true' : 'false');
        $('#strokeToggleBtn').toggleClass('active', enabled);
        $('.toolbar-stroke-options').toggleClass('hidden', !enabled);
        handleStyleChange();
    });
    
    bindEditModeEvent('#strokeWidthNew', 'input change', handleStyleChange);
    
    // 旋转控制
    bindEditModeEvent('#rotationAngleNew', 'input change', handleStyleChange);
    
    bindEditModeEvent('#rotateLeftBtn', 'click', function() {
        const current = parseInt($('#rotationAngleNew').val()) || 0;
        const newAngle = Math.max(-180, current - 15);
        $('#rotationAngleNew').val(newAngle);
        handleStyleChange();
    });
    
    bindEditModeEvent('#rotateRightBtn', 'click', function() {
        const current = parseInt($('#rotationAngleNew').val()) || 0;
        const newAngle = Math.min(180, current + 15);
        $('#rotationAngleNew').val(newAngle);
        handleStyleChange();
    });
    
    bindEditModeEvent('#rotateResetBtn', 'click', function() {
        $('#rotationAngleNew').val(0);
        handleStyleChange();
    });

    // 位置调整
    bindEditModeEvent('#moveUpNew', 'click', () => adjustPositionNew('moveUp'));
    bindEditModeEvent('#moveDownNew', 'click', () => adjustPositionNew('moveDown'));
    bindEditModeEvent('#moveLeftNew', 'click', () => adjustPositionNew('moveLeft'));
    bindEditModeEvent('#moveRightNew', 'click', () => adjustPositionNew('moveRight'));
    bindEditModeEvent('#resetPositionNew', 'click', resetPositionNew);

    // 操作按钮
    bindEditModeEvent('#applyBubbleEditNew', 'click', () => reRenderFullImage());
    bindEditModeEvent('#applyToAllBubblesNew', 'click', applySettingsToAllBubbles);
    bindEditModeEvent('#resetBubbleEditNew', 'click', resetCurrentBubble);

    bindEditModeEvent('#reOcrBtn', 'click', reOcrCurrentBubble);
    bindEditModeEvent('#reTranslateBtn', 'click', reTranslateCurrentBubble);

    // === 气泡操作按钮事件 ===
    bindEditModeEvent('#addBubbleBtn', 'click', function() {
        enterDrawingMode();
    });
    bindEditModeEvent('#deleteBubbleBtn', 'click', deleteSelectedBubble);
    bindEditModeEvent('#repairBubbleBtn', 'click', repairSelectedBubble);
    
    // === 从标注模式迁移的按钮事件 ===
    bindEditModeEvent('#autoDetectBtn', 'click', autoDetectBubbles);
    bindEditModeEvent('#detectAllBtn', 'click', detectAllImages);
    bindEditModeEvent('#translateWithBubblesBtn', 'click', translateWithCurrentBubbles);
    
    // 绑定气泡操作事件（拖拽、调整大小、中键绘制）
    bindBubbleOperationEvents();

    $(document).off('keydown.editMode').on('keydown.editMode', handleEditModeKeydown);
    $(document).off('keyup.editMode').on('keyup.editMode', handleEditModeKeyup);
}

/**
 * 解绑新版编辑模式事件
 */
function unbindNewEditModeEvents() {
    editModeBoundSelectors.forEach(selector => {
        $(selector).off(EDIT_MODE_EVENT_NS);
    });
    editModeBoundSelectors.clear();
    $(document).off('keydown.editMode');
    $(document).off('keyup.editMode');
    
    // 确保退出笔刷模式
    if (brushMode) {
        exitBrushMode();
    }
}

/**
 * 处理编辑模式键盘快捷键
 */
function handleEditModeKeydown(e) {
    if (!state.editModeActive) return;
    
    const key = e.key.toLowerCase();
    const $target = $(e.target);
    
    // 笔刷快捷键 R/U 和导航快捷键 A/D 只在 textarea 中禁用（用户可能想输入文字）
    // 在其他所有元素（包括 select、input[type=number]、input[type=color] 等）中都允许触发
    if (key === 'r' || key === 'u' || key === 'a' || key === 'd') {
        if ($target.is('textarea')) return;
        // 让其他输入元素失去焦点，以便快捷键正常工作
        if ($target.is('input, select, button')) {
            $target.blur();
        }
    } else {
        // 其他快捷键在输入框中不处理
        if ($target.is('input, textarea, select')) return;
    }
    
    switch(e.key) {
        // 注意：已移除 ArrowLeft/ArrowRight 切换气泡的快捷键
        case 'a':
        case 'A':
            if (!brushMode) {
                navigateImage(-1);
                e.preventDefault();
            }
            break;
        case 'd':
        case 'D':
            if (!brushMode) {
                navigateImage(1);
                e.preventDefault();
            }
            break;
        case 'Enter':
            if (e.ctrlKey && !brushMode) {
                applyAndNextImage();
                e.preventDefault();
            }
            break;
        case 'Delete':
        case 'Backspace':
            // 删除选中的气泡框
            if (!brushMode && state.selectedBubbleIndex >= 0) {
                deleteSelectedBubble();
                e.preventDefault();
            }
            break;
        case 'r':
        case 'R':
            // R 键进入修复笔刷模式
            if (!isBrushKeyDown) {
                enterBrushMode('repair');
                e.preventDefault();
            }
            break;
        case 'u':
        case 'U':
            // U 键进入还原笔刷模式
            if (!isBrushKeyDown) {
                enterBrushMode('restore');
                e.preventDefault();
            }
            break;
    }
}

/**
 * 处理编辑模式键盘抬起
 */
function handleEditModeKeyup(e) {
    if (!state.editModeActive) return;
    
    const key = e.key.toLowerCase();
    // 只要 R 或 U 键抬起，就重置笔刷状态（无论当前是什么模式）
    if (key === 'r' || key === 'u') {
        // 强制重置笔刷状态
        isBrushKeyDown = false;
        if (brushMode) {
            exitBrushMode();
        }
        e.preventDefault();
    }
}

/**
 * 选择气泡（新版）
 */
export function selectBubbleNew(index) {
    const currentImage = state.getCurrentImage();
    
    // 验证并同步数组长度一致性（Bug 7 防护）
    if (!validateBubbleArrays(currentImage)) {
        console.log('检测到数组不一致，正在同步...');
        syncBubbleArraysLength(currentImage);
    }
    
    if (!state.bubbleStates || index < 0 || index >= state.bubbleStates.length) return;
    
    state.setSelectedBubbleIndex(index);
    
    // 更新导航器
    updateBubbleNavigator();
    
    // 更新文本编辑区
    updateTextEditors(index);
    
    // 更新样式设置
    updateStyleSettings(index);
    
    // 更新高亮
    updateBubbleHighlightsNew();
    
    // 更新删除和修复按钮状态
    const hasSelection = index >= 0;
    $('#deleteBubbleBtn').prop('disabled', !hasSelection);
    $('#repairBubbleBtn').prop('disabled', !hasSelection);
    
    // 不再自动滚动到气泡位置，避免切换气泡时图片位置频繁变化
    // scrollToBubble(index);
}

/**
 * 选择上一个气泡（新版）
 */
function selectPrevBubbleNew() {
    if (state.selectedBubbleIndex > 0) {
        selectBubbleNew(state.selectedBubbleIndex - 1);
    }
}

/**
 * 选择下一个气泡（新版）
 */
function selectNextBubbleNew() {
    if (state.selectedBubbleIndex < state.bubbleStates.length - 1) {
        selectBubbleNew(state.selectedBubbleIndex + 1);
    }
}

/**
 * 更新气泡导航器
 */
function updateBubbleNavigator() {
    const total = state.bubbleStates.length;
    const current = state.selectedBubbleIndex + 1;
    
    $('#currentBubbleNum').text(current);
    $('#totalBubbleNum').text(total);
    
    $('#prevBubbleBtn').prop('disabled', state.selectedBubbleIndex <= 0);
    $('#nextBubbleBtn').prop('disabled', state.selectedBubbleIndex >= total - 1);
}

// ============ 图片导航功能 ============

/**
 * 更新图片导航器
 */
function updateImageNavigator() {
    const total = state.images.length;
    const current = state.currentImageIndex + 1;
    
    $('#currentImageNum').text(current);
    $('#totalImageNum').text(total);
    
    $('#prevImageBtn').prop('disabled', state.currentImageIndex <= 0);
    $('#nextImageBtn').prop('disabled', state.currentImageIndex >= total - 1);
}

/**
 * 导航到上一张或下一张图片
 * @param {number} direction - -1表示上一张，1表示下一张
 */
function navigateImage(direction) {
    // 如果在笔刷模式下，先退出笔刷
    if (brushMode) {
        exitBrushMode();
    }
    
    const newIndex = state.currentImageIndex + direction;
    
    if (newIndex < 0 || newIndex >= state.images.length) {
        return;
    }
    
    // 保存当前气泡设置
    saveCurrentBubbleStates();
    
    // 清理多选状态
    clearMultiSelection();
    
    // 切换图片
    state.setCurrentImageIndex(newIndex);
    
    // 重新初始化编辑模式（加载新图片的气泡）
    initBubbleStates();
    
    // 更新图片显示（不重置视图位置，保持用户当前的缩放和位置）
    loadImagesToViewer(false);
    
    // 更新导航器
    updateImageNavigator();
    updateBubbleNavigator();
    
    // 更新缩略图高亮
    updateThumbnailHighlight();
    
    // 选择第一个气泡
    if (state.bubbleStates.length > 0) {
        selectBubbleNew(0);
    }
    
    ui.showGeneralMessage(`已切换到图片 ${newIndex + 1}/${state.images.length}`, 'info', false, 1500);
}

/**
 * 保存当前气泡状态到图片对象
 */
function saveCurrentBubbleStates() {
    const currentImage = state.getCurrentImage();
    if (currentImage && state.bubbleStates.length > 0) {
        currentImage.bubbleStates = JSON.parse(JSON.stringify(state.bubbleStates));
        currentImage.bubbleTexts = state.bubbleStates.map(s => s.translatedText || "");
    }
}

/**
 * 加载保存的布局模式
 */
function loadLayoutMode() {
    try {
        const saved = localStorage.getItem(LAYOUT_MODE_KEY);
        if (saved && (saved === 'horizontal' || saved === 'vertical')) {
            layoutMode = saved;
        }
    } catch (e) {
        console.warn('加载布局模式失败:', e);
    }
    applyLayoutMode();
}

/**
 * 保存布局模式
 */
function saveLayoutMode() {
    try {
        localStorage.setItem(LAYOUT_MODE_KEY, layoutMode);
    } catch (e) {
        console.warn('保存布局模式失败:', e);
    }
}

/**
 * 切换布局模式
 */
function toggleLayoutMode() {
    layoutMode = layoutMode === 'horizontal' ? 'vertical' : 'horizontal';
    applyLayoutMode();
    saveLayoutMode();
    
    // 切换后适应屏幕
    setTimeout(() => {
        dualViewer?.fitToScreen();
    }, 300);
    
    ui.showGeneralMessage(
        layoutMode === 'horizontal' ? '已切换到左右布局' : '已切换到上下布局', 
        'info', false, 1500
    );
}

/**
 * 应用布局模式
 */
function applyLayoutMode() {
    const $workspace = $('#editWorkspace');
    const $layoutBtn = $('#layoutToggle');
    const $iconH = $layoutBtn.find('.layout-icon-horizontal');
    const $iconV = $layoutBtn.find('.layout-icon-vertical');
    const $panel = $('#editPanelContainer');
    
    // 重置面板样式，避免之前的布局样式影响新布局
    $panel.css({
        'flex': '',
        'width': '',
        'height': '',
        'min-width': '',
        'max-width': '',
        'min-height': '',
        'max-height': ''
    });
    
    if (layoutMode === 'vertical') {
        $workspace.addClass('layout-vertical');
        $layoutBtn.addClass('active');
        $iconH.hide();
        $iconV.show();
    } else {
        $workspace.removeClass('layout-vertical');
        $layoutBtn.removeClass('active');
        $iconH.show();
        $iconV.hide();
    }
    
    // 更新缩略图面板状态类（用于上下布局时的空间留白）
    updateThumbnailsVisibleClass();
}

/**
 * 更新缩略图可见状态类
 */
function updateThumbnailsVisibleClass() {
    const $workspace = $('#editWorkspace');
    const panel = $('#editThumbnailsPanel');
    
    if (panel.is(':visible')) {
        $workspace.addClass('thumbnails-visible');
    } else {
        $workspace.removeClass('thumbnails-visible');
    }
}

/**
 * 切换缩略图面板显示
 */
function toggleThumbnailsPanel() {
    const panel = $('#editThumbnailsPanel');
    const isVisible = panel.is(':visible');
    
    if (isVisible) {
        panel.slideUp(200, () => {
            updateThumbnailsVisibleClass();
        });
        $('#toggleThumbnails').removeClass('active');
    } else {
        // 先渲染缩略图
        renderEditThumbnails();
        panel.slideDown(200, () => {
            updateThumbnailsVisibleClass();
        });
        $('#toggleThumbnails').addClass('active');
    }
}

/**
 * 渲染编辑模式缩略图
 */
function renderEditThumbnails() {
    const container = $('#editThumbnailsScroll');
    container.empty();
    
    state.images.forEach((imageData, index) => {
        const item = $(`
            <div class="edit-thumbnail-item ${index === state.currentImageIndex ? 'active' : ''}" data-index="${index}">
                <img src="${imageData.originalDataURL || imageData.translatedDataURL}" alt="图片 ${index + 1}">
                <span class="thumb-index">${index + 1}</span>
            </div>
        `);
        
        item.on('click', function() {
            const targetIndex = parseInt($(this).data('index'));
            if (targetIndex !== state.currentImageIndex) {
                // 计算移动方向
                const direction = targetIndex - state.currentImageIndex;
                // 直接设置新索引
                saveCurrentBubbleStates();
                state.setCurrentImageIndex(targetIndex);
                initBubbleStates();
                loadImagesToViewer(false);
                updateImageNavigator();
                updateBubbleNavigator();
                updateThumbnailHighlight();
                if (state.bubbleStates.length > 0) {
                    selectBubbleNew(0);
                }
                ui.showGeneralMessage(`已切换到图片 ${targetIndex + 1}/${state.images.length}`, 'info', false, 1500);
            }
        });
        
        container.append(item);
    });
    
    // 滚动到当前图片
    scrollToCurrentThumbnail();
}

/**
 * 更新缩略图高亮
 */
function updateThumbnailHighlight() {
    $('.edit-thumbnail-item').removeClass('active');
    $(`.edit-thumbnail-item[data-index="${state.currentImageIndex}"]`).addClass('active');
}

/**
 * 滚动到当前缩略图
 */
function scrollToCurrentThumbnail() {
    const container = $('#editThumbnailsScroll');
    const currentThumb = $(`.edit-thumbnail-item[data-index="${state.currentImageIndex}"]`);
    
    if (currentThumb.length) {
        const containerWidth = container.width();
        const thumbLeft = currentThumb.position().left;
        const thumbWidth = currentThumb.outerWidth();
        
        if (thumbLeft < 0 || thumbLeft + thumbWidth > containerWidth) {
            container.scrollLeft(container.scrollLeft() + thumbLeft - containerWidth / 2 + thumbWidth / 2);
        }
    }
}

/**
 * 更新文本编辑区
 */
function updateTextEditors(index) {
    const currentImage = state.getCurrentImage();
    if (!currentImage) return;
    
    const setting = state.bubbleStates[index];
    
    // 译文
    $('#translatedTextEditor').val(setting?.translatedText || '');
    
    // 原文 (OCR结果)
    const originalText = currentImage.originalTexts?.[index] || '';
    $('#originalTextEditor').val(originalText);
}

/**
 * 更新样式设置
 */
function updateStyleSettings(index) {
    const setting = state.bubbleStates[index];
    if (!setting) return;
    
    // 获取字号（直接使用 fontSize，自动字号已在首次翻译时计算好）
    let fontSize = 24; // 默认值
    if (typeof setting.fontSize === 'number' && setting.fontSize > 0) {
        fontSize = setting.fontSize;
    } else if (typeof setting.fontSize === 'string' && !isNaN(parseInt(setting.fontSize)) && setting.fontSize !== 'auto') {
        fontSize = parseInt(setting.fontSize);
    }
    console.log(`[updateStyleSettings] 气泡 ${index} 字号: ${fontSize}`);
    
    $('#fontSizeSlider').val(fontSize);
    $('#fontSizeInput').val(fontSize);
    
    // 更新预设按钮高亮
    $('.font-preset-btn').removeClass('active');
    $(`.font-preset-btn[data-size="${fontSize}"]`).addClass('active');
    
    // 字体 - 确保值存在于选项中
    const fontToSet = setting.fontFamily || state.defaultFontFamily;
    const fontSelect = $('#fontFamilyNew');
    if (fontSelect.find(`option[value="${fontToSet}"]`).length > 0) {
        fontSelect.val(fontToSet);
    } else {
        // 回退到全局选择器的值或默认值
        fontSelect.val($('#fontFamily').val() || state.defaultFontFamily);
    }
    
    // 排版方向 - 更新按钮状态
    const direction = setting.textDirection || state.defaultLayoutDirection;
    $('#textDirectionNew').val(direction);
    if (direction === 'vertical') {
        $('#directionVerticalBtn').attr('data-active', 'true').addClass('active');
        $('#directionHorizontalBtn').attr('data-active', 'false').removeClass('active');
    } else {
        $('#directionHorizontalBtn').attr('data-active', 'true').addClass('active');
        $('#directionVerticalBtn').attr('data-active', 'false').removeClass('active');
    }
    
    // 文字颜色 - 更新颜色指示器
    const textColor = setting.textColor || state.defaultTextColor;
    $('#textColorNew').val(textColor);
    $('#textColorIndicator').css('background', textColor);
    
    // 背景修复方式
    let inpaintMethod = setting.inpaintMethod || 'solid';
    // 兼容旧值：将 'lama' 转换为 'litelama'（旧版 LAMA 对应通用版）
    if (inpaintMethod === 'lama') {
        inpaintMethod = 'litelama';
    }
    $('#bubbleInpaintMethodNew').val(inpaintMethod);
    
    // 根据修复方式显示/隐藏颜色选择器
    if (inpaintMethod === 'solid') {
        $('#solidColorOptionsNew').removeClass('hidden');
    } else {
        $('#solidColorOptionsNew').addClass('hidden');
    }
    
    // 填充颜色 - 更新颜色指示器
    const fillColor = setting.fillColor || state.defaultFillColor;
    $('#fillColorNew').val(fillColor);
    $('#fillColorIndicator').css('background', fillColor);
    
    // 旋转角度
    $('#rotationAngleNew').val(setting.rotationAngle || 0);
    
    // 描边 - 更新按钮和颜色指示器
    const strokeEnabled = setting.strokeEnabled !== undefined ? setting.strokeEnabled : state.strokeEnabled;
    $('#strokeEnabledNew').prop('checked', strokeEnabled);
    $('#strokeToggleBtn').attr('data-active', strokeEnabled ? 'true' : 'false');
    $('#strokeToggleBtn').toggleClass('active', strokeEnabled);
    $('.toolbar-stroke-options').toggleClass('hidden', !strokeEnabled);
    
    const strokeColor = setting.strokeColor || state.strokeColor;
    $('#strokeColorNew').val(strokeColor);
    $('#strokeColorIndicator').css('background', strokeColor);
    $('#strokeWidthNew').val(setting.strokeWidth !== undefined ? setting.strokeWidth : state.strokeWidth);
    
    // 位置
    const position = setting.position || { x: 0, y: 0 };
    $('#positionXValue').text(position.x);
    $('#positionYValue').text(position.y);
}

/**
 * 更新气泡高亮（新版）
 */
function updateBubbleHighlightsNew() {
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.bubbleCoords) return;
    
    // 清除旧高亮
    $('#originalHighlights, #translatedHighlights').empty();
    
    const originalImg = $('#originalImageDisplay')[0];
    const translatedImg = $('#translatedImageDisplayNew')[0];
    
    if (!originalImg?.naturalWidth || !translatedImg?.naturalWidth) {
        // 图片未加载，延迟重试
        setTimeout(() => updateBubbleHighlightsNew(), 200);
        return;
    }
    
    // 为每个气泡创建高亮框
    currentImage.bubbleCoords.forEach((coords, index) => {
        const [x1, y1, x2, y2] = coords;
        const isSelected = index === state.selectedBubbleIndex;
        
        // 获取气泡的旋转角度（从 bubbleStates 或 bubbleAngles）
        let rotationAngle = 0;
        if (state.bubbleStates && state.bubbleStates[index]) {
            rotationAngle = state.bubbleStates[index].rotationAngle || 0;
        } else if (currentImage.bubbleAngles && currentImage.bubbleAngles[index]) {
            rotationAngle = currentImage.bubbleAngles[index] || 0;
        }
        
        // 原图高亮
        const origHighlight = createHighlightBox(x1, y1, x2, y2, index, isSelected, rotationAngle);
        $('#originalHighlights').append(origHighlight);
        
        // 翻译图高亮
        const transHighlight = createHighlightBox(x1, y1, x2, y2, index, isSelected, rotationAngle);
        $('#translatedHighlights').append(transHighlight);
    });
    
    // 绑定高亮框点击事件
    $('.bubble-highlight-box').off('click').on('click', function(e) {
        e.stopPropagation();
        const bubbleIndex = parseInt($(this).data('index'));
        selectBubbleNew(bubbleIndex);
    });
    
    // 绑定旋转手柄事件
    bindRotateHandleEvents();
}

/**
 * 创建高亮框元素
 * @param {number} x1 - 左上角x
 * @param {number} y1 - 左上角y
 * @param {number} x2 - 右下角x
 * @param {number} y2 - 右下角y
 * @param {number} index - 气泡索引
 * @param {boolean} isSelected - 是否选中
 * @param {number} rotationAngle - 旋转角度（度）
 */
function createHighlightBox(x1, y1, x2, y2, index, isSelected, rotationAngle = 0) {
    const width = x2 - x1;
    const height = y2 - y1;
    const centerX = x1 + width / 2;
    const centerY = y1 + height / 2;
    
    const box = $('<div class="bubble-highlight-box"></div>');
    
    // 如果有旋转角度，使用 transform-origin 和 transform 实现旋转
    if (rotationAngle !== 0) {
        // 将框的位置设置为中心点，然后用 transform 旋转
        box.css({
            left: (centerX - width / 2) + 'px',
            top: (centerY - height / 2) + 'px',
            width: width + 'px',
            height: height + 'px',
            transformOrigin: 'center center',
            transform: `rotate(${rotationAngle}deg)`
        });
    } else {
        box.css({
            left: x1 + 'px',
            top: y1 + 'px',
            width: width + 'px',
            height: height + 'px'
        });
    }
    
    box.attr('data-index', index);
    box.attr('data-coords', JSON.stringify([x1, y1, x2, y2]));
    box.attr('data-rotation', rotationAngle);
    if (isSelected) {
        box.addClass('active');
    }
    
    // 添加索引标签
    box.append(`<span class="bubble-index">#${index + 1}</span>`);
    
    // 添加四角调整手柄
    const handles = ['top-left', 'top-right', 'bottom-left', 'bottom-right'];
    handles.forEach(handleType => {
        const handle = $('<div class="resize-handle"></div>');
        handle.addClass(handleType);
        handle.attr('data-handle', handleType);
        handle.attr('data-parent-index', index);
        box.append(handle);
    });
    
    // 添加旋转手柄（上边框中点垂直延伸出的圆点）
    const rotateHandle = $('<div class="rotate-handle"></div>');
    rotateHandle.attr('data-parent-index', index);
    box.append(rotateHandle);
    
    // 添加旋转连接线
    const rotateLine = $('<div class="rotate-line"></div>');
    box.append(rotateLine);
    
    return box;
}

/**
 * 滚动到气泡位置
 */
function scrollToBubble(index) {
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.bubbleCoords || !currentImage.bubbleCoords[index]) return;
    
    const coords = currentImage.bubbleCoords[index];
    const img = $('#originalImageDisplay')[0];
    
    if (dualViewer && img?.naturalWidth) {
        dualViewer.scrollToBubble(coords, img.naturalWidth, img.naturalHeight);
    }
}

/**
 * 切换视图模式
 */
function toggleViewMode() {
    const modes = ['dual', 'original', 'translated'];
    const currentIdx = modes.indexOf(viewMode);
    viewMode = modes[(currentIdx + 1) % modes.length];
    
    applyViewMode();
    
    ui.showGeneralMessage(`视图模式: ${viewMode === 'dual' ? '双图对照' : viewMode === 'original' ? '仅原图' : '仅翻译图'}`, 'info', false, 1500);
}

/**
 * 应用视图模式
 */
function applyViewMode() {
    const originalPanel = $('#originalPanel');
    const translatedPanel = $('#translatedPanel');
    const divider = $('#panelDivider');
    
    switch(viewMode) {
        case 'dual':
            originalPanel.show().removeClass('collapsed');
            translatedPanel.show().removeClass('collapsed');
            divider.show();
            break;
        case 'original':
            originalPanel.show().removeClass('collapsed');
            translatedPanel.hide();
            divider.hide();
            break;
        case 'translated':
            originalPanel.hide();
            translatedPanel.show().removeClass('collapsed');
            divider.hide();
            break;
    }
    
    // 更新按钮状态
    $('#viewModeToggle').toggleClass('single-mode', viewMode !== 'dual');
}

/**
 * 切换同步状态
 */
function toggleSyncView() {
    if (dualViewer) {
        const isSync = dualViewer.toggleSync();
        $('#syncViewToggle').toggleClass('active', isSync);
        ui.showGeneralMessage(`双图同步: ${isSync ? '开启' : '关闭'}`, 'info', false, 1500);
    }
}

/**
 * 切换面板折叠
 */
function togglePanel(panelName) {
    const panel = $(`#${panelName}Panel`);
    const btn = panel.find('.panel-toggle');
    
    if (panel.hasClass('collapsed')) {
        panel.removeClass('collapsed');
        btn.text('−');
    } else {
        panel.addClass('collapsed');
        btn.text('+');
    }
}

/**
 * 处理译文变化（实时预览）
 */
function handleTranslatedTextChange() {
    const text = $('#translatedTextEditor').val();
    const index = state.selectedBubbleIndex;
    if (index >= 0) {
        state.updateSingleBubbleState(index, { translatedText: text });
        triggerDelayedPreview(index); // 实时预览
    }
}

/**
 * 处理原文变化
 */
function handleOriginalTextChange() {
    const text = $('#originalTextEditor').val();
    const index = state.selectedBubbleIndex;
    const currentImage = state.getCurrentImage();
    if (index >= 0 && currentImage) {
        if (!currentImage.originalTexts) {
            currentImage.originalTexts = [];
        }
        currentImage.originalTexts[index] = text;
    }
}

/**
 * 应用当前文本
 */
function applyCurrentText() {
    const index = state.selectedBubbleIndex;
    if (index < 0) return;
    
    const text = $('#translatedTextEditor').val();
    state.updateSingleBubbleState(index, { translatedText: text });
    
    triggerDelayedPreview(index);
    ui.showGeneralMessage('文本已应用', 'success', false, 1500);
}

/**
 * 应用当前文本并切换到下一张图片
 */
function applyAndNextImage() {
    applyCurrentText();
    
    if (state.currentImageIndex < state.images.length - 1) {
        setTimeout(() => {
            navigateImage(1);
        }, 100);
    } else {
        ui.showGeneralMessage('已是最后一张图片', 'info', false, 1500);
    }
}

/**
 * 处理字号变化
 */
function handleFontSizeChange(size) {
    const index = state.selectedBubbleIndex;
    if (index >= 0) {
        // 更新气泡字号
        state.updateSingleBubbleState(index, { fontSize: size });
        
        // 更新预设按钮高亮
        $('.font-preset-btn').removeClass('active');
        $(`.font-preset-btn[data-size="${size}"]`).addClass('active');
        
        triggerDelayedPreview(index);
    }
}

/**
 * 处理样式变化
 */
function handleStyleChange() {
    const index = state.selectedBubbleIndex;
    if (index < 0) return;
    
    // 检查字体值，如果是 'custom-font' 则跳过（由 events.js 处理文件选择）
    const fontValue = $('#fontFamilyNew').val();
    if (fontValue === 'custom-font') {
        return;
    }
    
    const setting = {
        fontFamily: fontValue,
        textColor: $('#textColorNew').val(),
        textDirection: $('#textDirectionNew').val(),
        rotationAngle: parseInt($('#rotationAngleNew').val()) || 0,
        inpaintMethod: $('#bubbleInpaintMethodNew').val(),
        fillColor: $('#fillColorNew').val(),
        strokeEnabled: $('#strokeEnabledNew').is(':checked'),
        strokeColor: $('#strokeColorNew').val(),
        strokeWidth: parseInt($('#strokeWidthNew').val()) || 3
    };
    
    state.updateSingleBubbleState(index, setting);
    triggerDelayedPreview(index);
}

/**
 * 位置调整（新版）
 */
function adjustPositionNew(direction) {
    const index = state.selectedBubbleIndex;
    if (index < 0) return;
    
    const currentSetting = state.bubbleStates[index];
    const position = { ...(currentSetting.position || { x: 0, y: 0 }) };
    const step = 2;
    const limit = getPositionLimit();
    
    switch(direction) {
        case 'moveUp': position.y = Math.max(position.y - step, -limit); break;
        case 'moveDown': position.y = Math.min(position.y + step, limit); break;
        case 'moveLeft': position.x = Math.max(position.x - step, -limit); break;
        case 'moveRight': position.x = Math.min(position.x + step, limit); break;
    }
    
    state.updateSingleBubbleState(index, { position: position });
    $('#positionXValue').text(position.x);
    $('#positionYValue').text(position.y);
    triggerDelayedPreview(index);
}

/**
 * 重置位置（新版）
 */
function resetPositionNew() {
    const index = state.selectedBubbleIndex;
    if (index < 0) return;
    
    state.updateSingleBubbleState(index, { position: { x: 0, y: 0 } });
    $('#positionXValue').text(0);
    $('#positionYValue').text(0);
    reRenderFullImage();
}

/**
 * 重新OCR当前气泡
 * 使用侧边栏配置的OCR引擎重新识别当前气泡
 */
async function reOcrCurrentBubble() {
    const index = state.selectedBubbleIndex;
    const currentImage = state.getCurrentImage();
    if (index < 0 || !currentImage || !currentImage.bubbleCoords?.[index]) {
        ui.showGeneralMessage('请先选择一个气泡', 'warning');
        return;
    }
    
    // 获取气泡坐标
    const coords = currentImage.bubbleCoords[index];
    if (!coords || coords.length < 4) {
        ui.showGeneralMessage('气泡坐标无效', 'error');
        return;
    }
    
    // 获取侧边栏配置的OCR引擎
    const ocrEngine = $('#ocrEngine').val() || 'manga_ocr';
    
    // 禁用按钮并显示加载状态
    const $btn = $('#reOcrBtn');
    const originalText = $btn.text();
    $btn.prop('disabled', true).text('⏳');
    ui.showGeneralMessage('正在重新OCR...', 'info', false, 0);
    
    try {
        const bubbleImageDataUrl = await extractBubbleImageDataURL(currentImage.originalDataURL, coords);
        const bubbleImageBase64 = bubbleImageDataUrl.split(',')[1];
        const response = await api.ocrSingleBubbleApi({
            bubble_image: bubbleImageBase64,
            bubble_coords: coords,
            ocr_engine: ocrEngine,
            source_language: $('#sourceLanguage').val() || 'japanese',
            // 百度OCR参数
            baidu_ocr_api_key: $('#baiduApiKey').val() || '',
            baidu_ocr_secret_key: $('#baiduSecretKey').val() || '',
            baidu_version: $('#baiduVersion').val() || 'standard',
            baidu_source_language: $('#baiduSourceLanguage').val() || 'auto_detect',
            // AI视觉OCR参数
            ai_vision_provider: $('#aiVisionProvider').val() || 'siliconflow',
            ai_vision_api_key: $('#aiVisionApiKey').val() || '',
            ai_vision_model_name: $('#aiVisionModelName').val() || '',
            ai_vision_ocr_prompt: $('#aiVisionOcrPrompt').val() || '',
            custom_ai_vision_base_url: $('#customAiVisionBaseUrl').val() || ''
        });
        
        if (response.success && response.text !== undefined) {
            // 更新原文编辑器
            $('#originalTextEditor').val(response.text);
            // 保存到 originalTexts
            if (!currentImage.originalTexts) {
                currentImage.originalTexts = [];
            }
            currentImage.originalTexts[index] = response.text;
            ui.showGeneralMessage('OCR完成', 'success', false, 2000);
        } else {
            ui.showGeneralMessage(response.error || 'OCR失败', 'error');
        }
    } catch (error) {
        console.error('重新OCR失败:', error);
        ui.showGeneralMessage('OCR请求失败: ' + (error.message || error), 'error');
    } finally {
        // 恢复按钮状态
        $btn.prop('disabled', false).text(originalText);
    }
}

/**
 * 重新翻译当前气泡
 * 使用侧边栏配置的翻译设置重新翻译当前气泡的原文
 */
async function reTranslateCurrentBubble() {
    const index = state.selectedBubbleIndex;
    const currentImage = state.getCurrentImage();
    if (index < 0 || !currentImage) {
        ui.showGeneralMessage('请先选择一个气泡', 'warning');
        return;
    }
    
    const originalText = $('#originalTextEditor').val();
    if (!originalText || originalText.trim() === '') {
        ui.showGeneralMessage('没有原文可翻译', 'warning');
        return;
    }
    
    // 获取侧边栏配置的翻译参数
    const modelProvider = $('#modelProvider').val();
    const apiKey = $('#apiKey').val();
    const modelName = $('#modelName').val();
    const targetLanguage = $('#targetLanguage').val() || 'zh';
    const customBaseUrl = $('#customBaseUrl').val() || '';
    
    // 直接从侧边栏的提示词文本框获取内容
    const promptContent = $('#promptContent').val() || '';
    
    // 禁用按钮并显示加载状态
    const $btn = $('#reTranslateBtn');
    const originalBtnText = $btn.text();
    $btn.prop('disabled', true).text('⏳');
    ui.showGeneralMessage('正在翻译...', 'info', false, 0);
    
    console.log('翻译参数:', { 
        modelProvider, 
        modelName, 
        targetLanguage, 
        promptContentLength: promptContent.length,
        promptContentPreview: promptContent.substring(0, 50) + '...',
        customBaseUrl: customBaseUrl ? '有' : '无'
    });
    
    try {
        
        // 调用翻译API
        const translatePayload = {
            original_text: originalText.trim(),
            target_language: targetLanguage,
            model_provider: modelProvider,
            model_name: modelName
        };
        if (apiKey) {
            translatePayload.api_key = apiKey;
        }
        if (promptContent && promptContent.trim()) {
            translatePayload.prompt_content = promptContent.trim();
        }
        if (customBaseUrl) {
            translatePayload.custom_base_url = customBaseUrl;
        }
        
        const response = await api.translateSingleTextApi(translatePayload);
        
        if (response.translated_text) {
            // 更新译文编辑器
            $('#translatedTextEditor').val(response.translated_text);
            // 更新气泡状态
            state.updateSingleBubbleState(index, { translatedText: response.translated_text });
            // 触发实时预览
            triggerDelayedPreview(index);
            ui.showGeneralMessage('翻译完成', 'success', false, 2000);
        } else {
            ui.showGeneralMessage(response.error || '翻译失败', 'error');
        }
    } catch (error) {
        console.error('重新翻译失败:', error);
        ui.showGeneralMessage('翻译请求失败: ' + (error.message || error), 'error');
    } finally {
        // 恢复按钮状态
        $btn.prop('disabled', false).text(originalBtnText);
    }
}

/**
 * 更新翻译图显示（新版）
 */
export function updateTranslatedImageNew(dataURL) {
    $('#translatedImageDisplayNew').attr('src', dataURL);
    // 同时更新旧的显示（保持兼容）
    $('#translatedImageDisplay').attr('src', dataURL);
}

// ============================================================
// ============ 编辑模式内置标注功能 ============
// ============================================================

// ============ 气泡操作功能（融合原标注模式） ============

/**
 * 进入绘制模式（点击添加按钮后）
 */
function enterDrawingMode() {
    isDrawingMode = true;
    $('#editWorkspace').addClass('drawing-mode');
    $('#addBubbleBtn').addClass('active');
    ui.showGeneralMessage('绘制模式：在图片上拖拽绘制气泡框，或按 Esc 取消', 'info', false, 2500);
}

/**
 * 退出绘制模式
 */
function exitDrawingMode() {
    isDrawingMode = false;
    $('#editWorkspace').removeClass('drawing-mode');
    $('#addBubbleBtn').removeClass('active');
    cleanupBubbleOperationState();
}

/**
 * 绑定气泡操作事件（始终生效）
 */
function bindBubbleOperationEvents() {
    const $viewports = $('#originalViewport, #translatedViewport');
    
    // 在视口上绑定鼠标事件
    $viewports.on('mousedown.bubbleOp', handleBubbleMouseDown);
    
    // 全局鼠标移动和释放
    $(document).on('mousemove.bubbleOp', handleBubbleMouseMove);
    $(document).on('mouseup.bubbleOp', handleBubbleMouseUp);
}

/**
 * 解绑气泡操作事件
 */
function unbindBubbleOperationEvents() {
    $('#originalViewport, #translatedViewport').off('.bubbleOp');
    $(document).off('.bubbleOp');
}

/**
 * 清理气泡操作状态
 */
function cleanupBubbleOperationState() {
    isDrawingBox = false;
    isDraggingBox = false;
    isResizingBox = false;
    isMiddleButtonDown = false;
    isRotatingBox = false;  // 清理旋转状态
    if (currentDrawingRect) {
        currentDrawingRect.remove();
        currentDrawingRect = null;
    }
    activeViewport = null;
    // 清理多选状态
    selectedBubbleIndices = [];
    $('body').removeClass('middle-button-drawing');
    $('body').removeClass('rotating-box');  // 清理旋转光标
    // 解绑旋转事件
    $(document).off('mousemove.rotate');
    $(document).off('mouseup.rotate');
}

/**
 * 处理鼠标按下事件
 */
function handleBubbleMouseDown(e) {
    // 笔刷模式下不处理气泡操作
    if (brushMode) return;
    
    const $target = $(e.target);
    const $viewport = $(e.currentTarget);
    activeViewport = $viewport;
    
    // 中键 (button === 1) - 绘制新框
    if (e.button === 1) {
        e.preventDefault();
        isMiddleButtonDown = true;
        // 设置十字光标
        $('body').addClass('middle-button-drawing');
        
        const pos = getMousePositionInImage(e, $viewport);
        const $img = $viewport.find('img');
        if (pos && $img.length) {
            const imgWidth = $img[0].naturalWidth || $img.width();
            const imgHeight = $img[0].naturalHeight || $img.height();
            if (pos.x >= 0 && pos.x <= imgWidth && pos.y >= 0 && pos.y <= imgHeight) {
                startDrawing(e, $viewport);
            }
        }
        return;
    }
    
    // 左键 (button === 0)
    if (e.button !== 0) return;
    
    // 检查是否点击了调整手柄
    if ($target.hasClass('resize-handle')) {
        e.preventDefault();
        e.stopPropagation();
        startResizing(e, $target);
        return;
    }
    
    // 检查是否点击了高亮框
    const $box = $target.hasClass('bubble-highlight-box') ? $target : $target.closest('.bubble-highlight-box');
    if ($box.length) {
        e.preventDefault();
        e.stopPropagation();
        const index = parseInt($box.data('index'));
        
        // Shift+点击 - 多选/取消选择
        if (e.shiftKey) {
            toggleBubbleSelection(index);
            return;
        }
        
        // 如果点击的是当前选中的框，开始拖拽
        if (index === state.selectedBubbleIndex) {
            startDragging(e, $box);
        } else {
            // 否则清除多选，只选择这个框
            clearMultiSelection();
            selectBubbleNew(index);
        }
        return;
    }
    
    // 点击空白处，清除多选
    if (!e.shiftKey) {
        clearMultiSelection();
    }
    
    // 绘制模式下，在空白处绘制新框
    if (isDrawingMode) {
        if ($target.hasClass('bubble-highlight-layer') || $target.closest('.image-canvas-wrapper').length) {
            const pos = getMousePositionInImage(e, $viewport);
            const $img = $viewport.find('img');
            if (pos && $img.length) {
                const imgWidth = $img[0].naturalWidth || $img.width();
                const imgHeight = $img[0].naturalHeight || $img.height();
                if (pos.x >= 0 && pos.x <= imgWidth && pos.y >= 0 && pos.y <= imgHeight) {
                    e.preventDefault();
                    startDrawing(e, $viewport);
                }
            }
        }
    }
}

/**
 * 切换气泡的多选状态
 */
function toggleBubbleSelection(index) {
    const idx = selectedBubbleIndices.indexOf(index);
    if (idx >= 0) {
        // 已选中，取消选择
        selectedBubbleIndices.splice(idx, 1);
    } else {
        // 未选中，添加到多选
        selectedBubbleIndices.push(index);
    }
    // 更新主选择为最后点击的
    selectBubbleNew(index);
    // 刷新多选高亮
    updateMultiSelectionHighlight();
}

/**
 * 清除多选
 */
function clearMultiSelection() {
    selectedBubbleIndices = [];
    updateMultiSelectionHighlight();
}

/**
 * 更新多选高亮显示
 */
function updateMultiSelectionHighlight() {
    // 移除所有多选高亮
    $('.bubble-highlight-box').removeClass('multi-selected');
    // 添加多选高亮
    selectedBubbleIndices.forEach(idx => {
        $(`.bubble-highlight-box[data-index="${idx}"]`).addClass('multi-selected');
    });
}

/**
 * 处理鼠标移动事件
 */
function handleBubbleMouseMove(e) {
    if (isDrawingBox) {
        updateDrawing(e);
    } else if (isDraggingBox) {
        updateDragging(e);
    } else if (isResizingBox) {
        updateResizing(e);
    }
}

/**
 * 处理鼠标释放事件
 */
function handleBubbleMouseUp(e) {
    // 中键释放时恢复光标
    if (e.button === 1 || isMiddleButtonDown) {
        isMiddleButtonDown = false;
        $('body').removeClass('middle-button-drawing');
    }
    
    if (isDrawingBox) {
        finishDrawing(e);
        // 如果是绘制模式，绘制完成后退出
        if (isDrawingMode) {
            exitDrawingMode();
        }
    } else if (isDraggingBox) {
        finishDragging(e);
    } else if (isResizingBox) {
        finishResizing(e);
    }
}

// ============ 绘制新框功能 ============

/**
 * 获取鼠标在图片原生坐标系中的位置
 */
function getMousePositionInImage(e, $viewport) {
    const $wrapper = $viewport.find('.image-canvas-wrapper');
    const $img = $viewport.find('img');
    
    if (!$wrapper.length || !$img.length) return null;
    
    // 使用 getBoundingClientRect 获取精确的视觉位置
    const wrapperRect = $wrapper[0].getBoundingClientRect();
    const scale = dualViewer?.getScale() || 1;
    
    // 计算鼠标相对于 wrapper 的位置，然后转换为图片原生坐标
    const x = (e.clientX - wrapperRect.left) / scale;
    const y = (e.clientY - wrapperRect.top) / scale;
    
    return { x, y };
}

/**
 * 开始绘制新框
 */
function startDrawing(e, $viewport) {
    const pos = getMousePositionInImage(e, $viewport);
    if (!pos) return;
    
    isDrawingBox = true;
    drawStartX = pos.x;
    drawStartY = pos.y;
    
    // 创建临时绘制矩形
    const $highlightLayer = $viewport.find('.bubble-highlight-layer');
    currentDrawingRect = $('<div class="drawing-rect-edit"></div>');
    currentDrawingRect.css({
        left: drawStartX + 'px',
        top: drawStartY + 'px',
        width: '0px',
        height: '0px'
    });
    
    $highlightLayer.append(currentDrawingRect);
    console.log('开始绘制新框:', drawStartX.toFixed(1), drawStartY.toFixed(1));
}

/**
 * 更新绘制中的框
 */
function updateDrawing(e) {
    if (!currentDrawingRect || !activeViewport) return;
    
    const pos = getMousePositionInImage(e, activeViewport);
    if (!pos) return;
    
    const width = Math.abs(pos.x - drawStartX);
    const height = Math.abs(pos.y - drawStartY);
    const left = Math.min(drawStartX, pos.x);
    const top = Math.min(drawStartY, pos.y);
    
    currentDrawingRect.css({
        left: left + 'px',
        top: top + 'px',
        width: width + 'px',
        height: height + 'px'
    });
}

/**
 * 完成绘制新框
 */
function finishDrawing(e) {
    isDrawingBox = false;
    
    if (!currentDrawingRect || !activeViewport) {
        cleanupBubbleOperationState();
        return;
    }
    
    const pos = getMousePositionInImage(e, activeViewport);
    
    // 移除临时绘制矩形
    currentDrawingRect.remove();
    currentDrawingRect = null;
    
    if (!pos) return;
    
    // 计算框的坐标（原生图像坐标）并进行边界检查
    const $img = activeViewport.find('img');
    const imgWidth = $img.length ? ($img[0].naturalWidth || $img.width()) : 2000;
    const imgHeight = $img.length ? ($img[0].naturalHeight || $img.height()) : 2000;
    
    const x1 = Math.max(0, Math.round(Math.min(drawStartX, pos.x)));
    const y1 = Math.max(0, Math.round(Math.min(drawStartY, pos.y)));
    const x2 = Math.min(imgWidth, Math.round(Math.max(drawStartX, pos.x)));
    const y2 = Math.min(imgHeight, Math.round(Math.max(drawStartY, pos.y)));
    
    // 检查框大小是否有效
    if (x2 - x1 < 10 || y2 - y1 < 10) {
        console.log('绘制的框太小，已忽略');
        return;
    }
    
    console.log('完成绘制新框:', [x1, y1, x2, y2]);
    
    // 添加新的气泡坐标
    addNewBubble([x1, y1, x2, y2]);
}

// ============ 拖拽移动功能 ============

/**
 * 开始拖拽框
 */
function startDragging(e, $box) {
    isDraggingBox = true;
    
    const index = parseInt($box.attr('data-index'));
    
    // 先选中这个框
    if (index !== state.selectedBubbleIndex) {
        selectBubbleNew(index);
    }
    
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    
    const coords = JSON.parse($box.attr('data-coords'));
    dragBoxInitialX = coords[0];
    dragBoxInitialY = coords[1];
    
    $box.css('cursor', 'grabbing');
}

/**
 * 更新拖拽位置
 */
function updateDragging(e) {
    const scale = dualViewer?.getScale() || 1;
    const deltaX = (e.clientX - dragStartX) / scale;
    const deltaY = (e.clientY - dragStartY) / scale;
    
    // 实时更新所有同步的高亮框显示
    const index = state.selectedBubbleIndex;
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.bubbleCoords || !currentImage.bubbleCoords[index]) return;
    
    const [x1, y1, x2, y2] = currentImage.bubbleCoords[index];
    const width = x2 - x1;
    const height = y2 - y1;
    
    const newX1 = dragBoxInitialX + deltaX;
    const newY1 = dragBoxInitialY + deltaY;
    
    // 更新两个视口中的高亮框位置
    $(`.bubble-highlight-box[data-index="${index}"]`).css({
        left: newX1 + 'px',
        top: newY1 + 'px'
    });
}

/**
 * 完成拖拽
 */
function finishDragging(e) {
    isDraggingBox = false;
    
    const scale = dualViewer?.getScale() || 1;
    const deltaX = (e.clientX - dragStartX) / scale;
    const deltaY = (e.clientY - dragStartY) / scale;
    
    // 更新气泡坐标
    const index = state.selectedBubbleIndex;
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.bubbleCoords || !currentImage.bubbleCoords[index]) return;
    
    const [x1, y1, x2, y2] = currentImage.bubbleCoords[index];
    const width = x2 - x1;
    const height = y2 - y1;
    
    const newX1 = Math.round(dragBoxInitialX + deltaX);
    const newY1 = Math.round(dragBoxInitialY + deltaY);
    const newX2 = newX1 + width;
    const newY2 = newY1 + height;
    
    // 边界检查 - 确保框不超出图片范围，同时处理框尺寸大于图片的极端情况
    const img = $('#originalImageDisplay')[0];
    const imgWidth = img?.naturalWidth || 2000;
    const imgHeight = img?.naturalHeight || 2000;
    
    // 确保框宽高不超过图片尺寸
    const safeWidth = Math.min(width, imgWidth);
    const safeHeight = Math.min(height, imgHeight);
    
    const finalX1 = Math.max(0, Math.min(newX1, imgWidth - safeWidth));
    const finalY1 = Math.max(0, Math.min(newY1, imgHeight - safeHeight));
    const finalX2 = finalX1 + safeWidth;
    const finalY2 = finalY1 + safeHeight;
    
    // 更新坐标
    updateBubbleCoords(index, [finalX1, finalY1, finalX2, finalY2]);
    
    // 光标样式由 CSS 控制，无需手动设置
}

// ============ 调整大小功能 ============

/**
 * 开始调整大小
 */
function startResizing(e, $handle) {
    isResizingBox = true;
    
    resizeHandleType = $handle.attr('data-handle');
    const index = parseInt($handle.attr('data-parent-index'));
    
    // 选中这个框
    if (index !== state.selectedBubbleIndex) {
        selectBubbleNew(index);
    }
    
    resizeStartX = e.clientX;
    resizeStartY = e.clientY;
    
    const currentImage = state.getCurrentImage();
    if (currentImage && currentImage.bubbleCoords && currentImage.bubbleCoords[index]) {
        resizeInitialCoords = [...currentImage.bubbleCoords[index]];
    }
    
    $('body').css('cursor', $handle.css('cursor'));
}

/**
 * 更新调整大小
 */
function updateResizing(e) {
    if (!resizeInitialCoords) return;
    
    const scale = dualViewer?.getScale() || 1;
    const deltaX = (e.clientX - resizeStartX) / scale;
    const deltaY = (e.clientY - resizeStartY) / scale;
    
    let [x1, y1, x2, y2] = resizeInitialCoords;
    
    // 根据手柄类型调整坐标
    if (resizeHandleType.includes('left')) x1 += deltaX;
    if (resizeHandleType.includes('right')) x2 += deltaX;
    if (resizeHandleType.includes('top')) y1 += deltaY;
    if (resizeHandleType.includes('bottom')) y2 += deltaY;
    
    // 确保有效性
    if (x1 > x2) [x1, x2] = [x2, x1];
    if (y1 > y2) [y1, y2] = [y2, y1];
    
    const width = x2 - x1;
    const height = y2 - y1;
    
    if (width < 10 || height < 10) return;
    
    // 实时更新显示
    const index = state.selectedBubbleIndex;
    $(`.bubble-highlight-box[data-index="${index}"]`).css({
        left: x1 + 'px',
        top: y1 + 'px',
        width: width + 'px',
        height: height + 'px'
    });
}

/**
 * 完成调整大小
 */
function finishResizing(e) {
    isResizingBox = false;
    
    if (!resizeInitialCoords) return;
    
    const scale = dualViewer?.getScale() || 1;
    const deltaX = (e.clientX - resizeStartX) / scale;
    const deltaY = (e.clientY - resizeStartY) / scale;
    
    let [x1, y1, x2, y2] = resizeInitialCoords;
    
    // 根据手柄类型调整坐标
    if (resizeHandleType.includes('left')) x1 += deltaX;
    if (resizeHandleType.includes('right')) x2 += deltaX;
    if (resizeHandleType.includes('top')) y1 += deltaY;
    if (resizeHandleType.includes('bottom')) y2 += deltaY;
    
    // 确保有效性
    if (x1 > x2) [x1, x2] = [x2, x1];
    if (y1 > y2) [y1, y2] = [y2, y1];
    
    // 获取图片尺寸进行边界检查
    const img = $('#originalImageDisplay')[0];
    const imgWidth = img?.naturalWidth || 2000;
    const imgHeight = img?.naturalHeight || 2000;
    
    // 边界约束
    x1 = Math.max(0, Math.round(x1));
    y1 = Math.max(0, Math.round(y1));
    x2 = Math.min(imgWidth, Math.round(x2));
    y2 = Math.min(imgHeight, Math.round(y2));
    
    const width = x2 - x1;
    const height = y2 - y1;
    
    if (width < 10 || height < 10) {
        ui.showGeneralMessage('框尺寸过小，已撤销', 'warning', false, 1500);
        updateBubbleHighlightsNew();
        return;
    }
    
    // 更新坐标
    const index = state.selectedBubbleIndex;
    updateBubbleCoords(index, [x1, y1, x2, y2]);
    
    resizeInitialCoords = null;
    $('body').css('cursor', 'default');
}

// ============ 添加/删除气泡功能 ============

/**
 * 验证气泡相关数组长度一致性
 * @returns {boolean} 是否一致
 */
function validateBubbleArrays(currentImage) {
    if (!currentImage) return true;
    
    const coordsLen = currentImage.bubbleCoords?.length || 0;
    const textsLen = currentImage.bubbleTexts?.length || 0;
    const originalLen = currentImage.originalTexts?.length || 0;
    const settingsLen = state.bubbleStates?.length || 0;
    
    if (coordsLen !== textsLen || coordsLen !== originalLen || coordsLen !== settingsLen) {
        console.warn(`气泡数组长度不一致: coords=${coordsLen}, texts=${textsLen}, original=${originalLen}, settings=${settingsLen}`);
        return false;
    }
    return true;
}

/**
 * 同步气泡数组长度（修复不一致问题）
 */
function syncBubbleArraysLength(currentImage) {
    if (!currentImage) return;
    
    const targetLen = currentImage.bubbleCoords?.length || 0;
    
    // 确保 bubbleTexts 长度一致
    if (!currentImage.bubbleTexts) currentImage.bubbleTexts = [];
    while (currentImage.bubbleTexts.length < targetLen) {
        currentImage.bubbleTexts.push('');
    }
    currentImage.bubbleTexts.length = targetLen;
    
    // 确保 originalTexts 长度一致
    if (!currentImage.originalTexts) currentImage.originalTexts = [];
    while (currentImage.originalTexts.length < targetLen) {
        currentImage.originalTexts.push('');
    }
    currentImage.originalTexts.length = targetLen;
    
    // 确保 bubbleAngles 长度一致
    if (!currentImage.bubbleAngles) currentImage.bubbleAngles = [];
    while (currentImage.bubbleAngles.length < targetLen) {
        currentImage.bubbleAngles.push(0);  // 默认角度为0
    }
    currentImage.bubbleAngles.length = targetLen;
    
    // 确保 bubbleStates 长度一致
    while (state.bubbleStates.length < targetLen) {
        const idx = state.bubbleStates.length;
        const detectedAngle = (currentImage.bubbleAngles && currentImage.bubbleAngles[idx]) || 0;
        // 计算自动排版方向
        let autoDir = 'vertical';
        if (currentImage.bubbleCoords && currentImage.bubbleCoords[idx] && currentImage.bubbleCoords[idx].length >= 4) {
            const [x1, y1, x2, y2] = currentImage.bubbleCoords[idx];
            autoDir = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal';
        }
        state.bubbleStates.push({
            translatedText: '',
            fontSize: state.defaultFontSize || 24,
            fontFamily: state.defaultFontFamily,
            textDirection: 'vertical',
            autoTextDirection: autoDir,  // 自动检测的排版方向
            position: { x: 0, y: 0 },
            textColor: state.defaultTextColor,
            rotationAngle: detectedAngle,
            fillColor: state.defaultFillColor,
            strokeEnabled: state.strokeEnabled,
            strokeColor: state.strokeColor,
            strokeWidth: state.strokeWidth
        });
    }
    state.bubbleStates.length = targetLen;
}

/**
 * 添加新气泡
 */
function addNewBubble(coords) {
    const currentImage = state.getCurrentImage();
    if (!currentImage) return;
    
    // 初始化数组
    if (!currentImage.bubbleCoords) currentImage.bubbleCoords = [];
    if (!currentImage.bubbleTexts) currentImage.bubbleTexts = [];
    if (!currentImage.originalTexts) currentImage.originalTexts = [];
    if (!currentImage.bubbleAngles) currentImage.bubbleAngles = [];
    
    // 添加坐标
    currentImage.bubbleCoords.push(coords);
    currentImage.bubbleTexts.push('');
    currentImage.originalTexts.push('');
    currentImage.bubbleAngles.push(0);  // 新气泡默认角度为0
    
    // 获取当前的填充色设置
    const fillColor = currentImage.fillColor || state.defaultFillColor || '#FFFFFF';
    
    // 添加对应的气泡设置
    // 处理自动排版：如果选择"auto"，回退到默认的 'vertical'
    const newBubbleLayoutDirection = $('#layoutDirection').val();
    const newBubbleTextDirection = newBubbleLayoutDirection === 'auto' ? 'vertical' : (newBubbleLayoutDirection || 'vertical');
    // 根据坐标计算自动排版方向
    const [x1, y1, x2, y2] = coords;
    const autoDir = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal';
    const newSetting = {
        translatedText: '',
        fontSize: parseInt($('#fontSize').val()) || 24,
        fontFamily: $('#fontFamily').val() || state.defaultFontFamily,
        textDirection: newBubbleTextDirection,
        autoTextDirection: autoDir,  // 根据宽高比计算的自动方向
        position: { x: 0, y: 0 },
        textColor: $('#textColor').val() || state.defaultTextColor,
        rotationAngle: 0,
        inpaintMethod: $('#bubbleInpaintMethodNew').val() || 'solid',
        fillColor: fillColor,
        strokeEnabled: state.strokeEnabled,
        strokeColor: state.strokeColor,
        strokeWidth: state.strokeWidth
    };
    
    // 使用状态管理函数更新 bubbleStates
    const newState = bubbleStateModule.createBubbleState(newSetting);
    state.setBubbleStates([...state.bubbleStates, newState]);
    
    // 选中新添加的气泡
    const newIndex = currentImage.bubbleCoords.length - 1;
    selectBubbleNew(newIndex);
    
    // 更新导航器
    updateBubbleNavigator();
    
    ui.showGeneralMessage(`已添加气泡框 #${newIndex + 1}，可点击"修复"按钮清理背景`, 'success', false, 2000);
}

/**
 * 删除选中的气泡框（支持批量删除）
 */
export async function deleteSelectedBubble() {
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.bubbleCoords) {
        ui.showGeneralMessage('无法删除：气泡数据无效', 'error');
        return;
    }
    
    // 确定要删除的索引列表（多选优先，否则使用当前选中）
    let indicesToDelete = [];
    if (selectedBubbleIndices.length > 0) {
        indicesToDelete = [...selectedBubbleIndices];
    } else if (state.selectedBubbleIndex >= 0) {
        indicesToDelete = [state.selectedBubbleIndex];
    }
    
    if (indicesToDelete.length === 0) {
        ui.showGeneralMessage('请先选中要删除的气泡框', 'warning');
        return;
    }
    
    // 按索引从大到小排序，避免删除时索引错位
    indicesToDelete.sort((a, b) => b - a);
    
    // 收集所有要删除的坐标（在删除前收集，避免索引变化）
    const coordsToRestore = [];
    for (const index of indicesToDelete) {
        if (index >= 0 && index < currentImage.bubbleCoords.length) {
            coordsToRestore.push([...currentImage.bubbleCoords[index]]);
        }
    }
    
    // 逐个删除数据
    for (const index of indicesToDelete) {
        if (index < 0 || index >= currentImage.bubbleCoords.length) continue;
        
        // 从数组中移除
        currentImage.bubbleCoords.splice(index, 1);
        if (currentImage.bubbleTexts) currentImage.bubbleTexts.splice(index, 1);
        if (currentImage.originalTexts) currentImage.originalTexts.splice(index, 1);
        if (currentImage.bubbleAngles) currentImage.bubbleAngles.splice(index, 1);
        
        // 更新 bubbleStates
        const newStates = [...state.bubbleStates];
        newStates.splice(index, 1);
        state.setBubbleStates(newStates);
        
        // 同步更新 initialBubbleStates
        if (state.initialBubbleStates && state.initialBubbleStates.length > index) {
            const newInitialStates = [...state.initialBubbleStates];
            newInitialStates.splice(index, 1);
            state.setInitialBubbleStates(newInitialStates);
        }
    }
    
    // 批量恢复 cleanImageData 中被删除区域的原图（确保背景正确）
    await restoreCleanImageAreaBatch(currentImage, coordsToRestore);
    
    // 整体重新渲染（彻底清除所有可能超出文本框的残留文字）
    if (state.bubbleStates.length > 0) {
        await reRenderFullImage(); // 需要更新界面显示，不能用 silentMode
    } else {
        // 如果没有气泡了，直接用 cleanImageData 或原图作为结果
        if (currentImage.cleanImageData) {
            currentImage.translatedDataURL = 'data:image/png;base64,' + currentImage.cleanImageData;
        } else {
            currentImage.translatedDataURL = currentImage.originalDataURL;
        }
        updateTranslatedImageNew(currentImage.translatedDataURL);
    }
    
    // 清除多选
    clearMultiSelection();
    
    // 调整选中索引
    let newSelectedIndex = -1;
    if (state.bubbleStates.length > 0) {
        newSelectedIndex = Math.min(indicesToDelete[indicesToDelete.length - 1], state.bubbleStates.length - 1);
        newSelectedIndex = Math.max(0, newSelectedIndex);
    }
    
    if (newSelectedIndex >= 0) {
        selectBubbleNew(newSelectedIndex);
    } else {
        state.setSelectedBubbleIndex(-1);
        updateBubbleHighlightsNew();
    }
    
    // 更新导航器和删除按钮状态
    updateBubbleNavigator();
    $('#deleteBubbleBtn').prop('disabled', state.selectedBubbleIndex < 0);
    
    const deletedCount = indicesToDelete.length;
    ui.showGeneralMessage(`已删除 ${deletedCount} 个气泡框，剩余 ${state.bubbleStates.length} 个`, 'success', false, 1500);
}

/**
 * 批量恢复 cleanImageData 中的多个区域（性能优化：只加载一次图片）
 * @param {object} currentImage - 当前图片对象
 * @param {Array} coordsList - 要恢复的坐标数组 [[x1,y1,x2,y2], ...]
 */
function restoreCleanImageAreaBatch(currentImage, coordsList) {
    // 如果没有要恢复的区域，直接返回
    if (!coordsList || coordsList.length === 0) {
        return Promise.resolve();
    }
    
    // 如果没有 cleanImageData，无需恢复（全局渲染时会使用 originalDataURL）
    if (!currentImage.cleanImageData || !currentImage.originalDataURL) {
        return Promise.resolve();
    }
    
    const originalImg = new Image();
    const cleanImg = new Image();
    
    // 只加载一次图片
    const loadPromises = [
        new Promise(resolve => {
            originalImg.onload = resolve;
            originalImg.onerror = resolve;
            originalImg.src = currentImage.originalDataURL;
        }),
        new Promise(resolve => {
            cleanImg.onload = resolve;
            cleanImg.onerror = resolve;
            cleanImg.src = 'data:image/png;base64,' + currentImage.cleanImageData;
        })
    ];
    
    return Promise.all(loadPromises).then(() => {
        // 创建一次 Canvas
        const canvas = document.createElement('canvas');
        canvas.width = cleanImg.naturalWidth;
        canvas.height = cleanImg.naturalHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(cleanImg, 0, 0);
        
        // 批量恢复所有区域
        for (const coords of coordsList) {
            const [x1, y1, x2, y2] = coords;
            const sWidth = (x2 - x1) + 1;
            const sHeight = (y2 - y1) + 1;
            
            if (sWidth > 0 && sHeight > 0) {
                ctx.drawImage(originalImg, x1, y1, sWidth, sHeight, x1, y1, sWidth, sHeight);
            }
        }
        
        // 一次性更新 cleanImageData
        currentImage.cleanImageData = canvas.toDataURL('image/png').split(',')[1];
        console.log(`已批量恢复 cleanImageData ${coordsList.length} 个区域`);
    }).catch(err => {
        console.error('批量恢复 cleanImageData 区域失败:', err);
    });
}

/**
 * 更新气泡坐标
 */
function updateBubbleCoords(index, newCoords) {
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.bubbleCoords) return;
    
    currentImage.bubbleCoords[index] = newCoords;
    
    // 同步更新 bubbleStates 中的坐标（保持状态一致性）
    if (state.bubbleStates && state.bubbleStates[index]) {
        state.updateSingleBubbleState(index, { coords: newCoords });
    }
    
    // 刷新高亮框显示
    updateBubbleHighlightsNew();
    
    console.log(`气泡框 #${index + 1} 坐标已更新:`, newCoords);
    
    // 延迟重新渲染翻译图，使译文跟随框位置更新
    triggerDelayedPreview(index);
}

/**
 * 导出绘制模式状态（供外部使用）
 */
export function isDrawingModeActive() {
    return isDrawingMode;
}

/**
 * 修复选中气泡的背景
 */
async function repairSelectedBubble() {
    // 防抖检查：如果已经在修复中，忽略重复调用（包括键盘快捷键触发）
    if (isRepairingBubble) {
        console.log('修复操作正在进行中，忽略重复调用');
        return;
    }
    
    const index = state.selectedBubbleIndex;
    if (index < 0) {
        ui.showGeneralMessage('请先选中要修复的气泡框', 'warning');
        return;
    }
    
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.bubbleCoords || !currentImage.bubbleCoords[index]) {
        ui.showGeneralMessage('无法获取气泡坐标', 'error');
        return;
    }
    
    const coords = currentImage.bubbleCoords[index];
    
    // 获取当前气泡的修复方法设置和旋转角度
    const bubbleSetting = state.bubbleStates[index] || {};
    const inpaintMethod = bubbleSetting.inpaintMethod || 'solid';
    const fillColor = bubbleSetting.fillColor || currentImage.fillColor || state.defaultFillColor || '#FFFFFF';
    const rotationAngle = bubbleSetting.rotationAngle || 0;
    
    // 设置防抖标志并禁用按钮
    isRepairingBubble = true;
    const $btn = $('#repairBubbleBtn');
    $btn.prop('disabled', true);
    
    try {
        // 根据气泡设置的修复方法决定使用 LAMA 还是纯色填充
        const isLamaMethod = inpaintMethod === 'lama_mpe' || inpaintMethod === 'litelama' || inpaintMethod === 'lama';
        if (isLamaMethod) {
            ui.showLoading(`正在对气泡 #${index + 1} 进行 LAMA 背景修复...`);
            
            // 使用当前的干净背景或原图作为基础（保留之前的修复效果）
            let baseImageData;
            if (currentImage.cleanImageData) {
                baseImageData = currentImage.cleanImageData;
            } else if (currentImage.originalDataURL) {
                baseImageData = currentImage.originalDataURL.split(',')[1];
            } else {
                throw new Error('无法获取基础图像');
            }
            
            // 调用后端 LAMA 修复 API（传递角度用于生成旋转多边形）
            // 根据 inpaintMethod 传递具体的 LAMA 模型类型
            // 旧值 'lama' 对应 litelama（通用版）
            const lamaModel = (inpaintMethod === 'litelama' || inpaintMethod === 'lama') ? 'litelama' : 'lama_mpe';
            const response = await api.inpaintSingleBubbleApi({
                image_data: baseImageData,
                bubble_coords: coords,
                bubble_angle: rotationAngle,
                method: 'lama',
                lama_model: lamaModel
            });
            
            if (response.success && response.inpainted_image) {
                // 更新干净背景（保留其他气泡的修复效果）
                currentImage.cleanImageData = response.inpainted_image;
                currentImage._lama_inpainted = true;
                
                ui.hideLoading();
                ui.showGeneralMessage(`气泡框 #${index + 1} LAMA背景修复完成，正在重新渲染...`, 'success', false, 1500);
                
                // 重新渲染所有气泡的译文（在干净背景上）
                await reRenderFullImage();
                return;
            } else {
                throw new Error(response.error || 'LAMA修复返回无效数据');
            }
        }
        
        // 使用纯色填充（传递角度）
        await fillBubbleAreaWithColor(currentImage, coords, fillColor, rotationAngle);
        ui.showGeneralMessage(`气泡框 #${index + 1} 背景已修复，正在重新渲染...`, 'success', false, 1500);
        
        // 重新渲染所有气泡的译文
        await reRenderFullImage();
        
    } catch (error) {
        console.error('背景修复失败:', error);
        ui.hideLoading();
        
        // 使用纯色填充作为备选方案
        ui.showGeneralMessage('LAMA修复失败，使用纯色填充', 'warning', false, 2000);
        await fillBubbleAreaWithColor(currentImage, coords, fillColor);
        // 重新渲染
        await reRenderFullImage();
    } finally {
        // 重置防抖状态并恢复按钮
        isRepairingBubble = false;
        $btn.prop('disabled', false);
    }
}

/**
 * 使用纯色填充气泡区域（只更新cleanImageData，不直接更新translatedDataURL）
 * @param {Object} currentImage - 当前图片对象
 * @param {Array} coords - 坐标 [x1, y1, x2, y2]
 * @param {string} fillColor - 填充颜色
 * @param {number} rotationAngle - 旋转角度（度），默认0
 */
async function fillBubbleAreaWithColor(currentImage, coords, fillColor, rotationAngle = 0) {
    const [x1, y1, x2, y2] = coords;
    
    // 获取当前的干净背景或原图作为基础（保留之前的修复效果）
    let baseSrc;
    if (currentImage.cleanImageData) {
        baseSrc = 'data:image/png;base64,' + currentImage.cleanImageData;
    } else if (currentImage.originalDataURL) {
        baseSrc = currentImage.originalDataURL;
    } else {
        console.error('无法找到基础图像用于填充');
        return;
    }
    
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            const ctx = canvas.getContext('2d');
            
            // 绘制基础图像
            ctx.drawImage(img, 0, 0);
            
            // 用填充色填充指定气泡区域
            ctx.fillStyle = fillColor;
            
            if (Math.abs(rotationAngle) < 0.1) {
                // 无旋转，使用简单矩形
                ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
            } else {
                // 有旋转，绘制旋转后的多边形
                const cx = (x1 + x2) / 2;
                const cy = (y1 + y2) / 2;
                const hw = (x2 - x1) / 2;
                const hh = (y2 - y1) / 2;
                const rad = rotationAngle * Math.PI / 180;
                const cos_a = Math.cos(rad);
                const sin_a = Math.sin(rad);
                
                // 计算旋转后的四个角点
                const corners = [
                    [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
                ].map(([dx, dy]) => [
                    cx + dx * cos_a - dy * sin_a,
                    cy + dx * sin_a + dy * cos_a
                ]);
                
                ctx.beginPath();
                ctx.moveTo(corners[0][0], corners[0][1]);
                for (let i = 1; i < corners.length; i++) {
                    ctx.lineTo(corners[i][0], corners[i][1]);
                }
                ctx.closePath();
                ctx.fill();
            }
            
            // 只更新干净背景数据（保留其他气泡的修复效果）
            // 不直接更新 translatedDataURL，由调用者通过 reRenderFullImage 重新渲染所有译文
            const newCleanData = canvas.toDataURL('image/png').split(',')[1];
            currentImage.cleanImageData = newCleanData;
            
            console.log('已对气泡区域进行纯色填充:', coords, fillColor, '角度:', rotationAngle);
            resolve();
        };
        img.onerror = () => {
            console.error('加载基础图像失败');
            resolve();
        };
        img.src = baseSrc;
    });
}

// ============ 笔刷功能 ============

/**
 * 进入笔刷模式
 * @param {string} mode - 'repair' 修复笔刷 | 'restore' 还原笔刷
 */
function enterBrushMode(mode) {
    if (brushMode === mode) return;
    
    brushMode = mode;
    isBrushKeyDown = true;
    brushPath = [];
    
    // 禁用图像拖拽和缩放
    if (dualViewer) {
        if (dualViewer.originalViewer) dualViewer.originalViewer.isDragging = false;
        if (dualViewer.translatedViewer) dualViewer.translatedViewer.isDragging = false;
    }
    
    // 添加笔刷模式样式
    $('#editWorkspace').addClass('brush-mode-active');
    $('#editWorkspace').attr('data-brush-mode', mode);
    
    // 更新工具栏图标状态
    if (mode === 'repair') {
        $('#repairBrushBtn').addClass('active');
        $('#restoreBrushBtn').removeClass('active');
    } else {
        $('#restoreBrushBtn').addClass('active');
        $('#repairBrushBtn').removeClass('active');
    }
    
    // 显示笔刷大小
    $('#brushSizeDisplay').text(brushSize + 'px').show();
    
    // 更新笔刷光标
    updateBrushCursor();
    
    // 绑定笔刷事件
    bindBrushEvents();
    
    console.log(`进入${mode === 'repair' ? '修复' : '还原'}笔刷模式，笔刷大小: ${brushSize}px`);
}

/**
 * 退出笔刷模式
 */
function exitBrushMode() {
    // 如果正在涂抹，先完成涂抹
    if (isBrushPainting) {
        finishBrushPainting();
    }
    
    // 无论当前状态如何，都重置所有笔刷相关变量
    const wasActive = brushMode !== null;
    brushMode = null;
    isBrushKeyDown = false;
    isBrushPainting = false;
    brushPath = [];
    
    // 如果笔刷之前是激活状态，清理 UI
    if (!wasActive) return;
    
    // 移除笔刷模式样式
    $('#editWorkspace').removeClass('brush-mode-active');
    $('#editWorkspace').removeAttr('data-brush-mode');
    
    // 移除工具栏图标状态
    $('#repairBrushBtn').removeClass('active');
    $('#restoreBrushBtn').removeClass('active');
    
    // 隐藏笔刷大小显示
    $('#brushSizeDisplay').hide();
    
    // 移除笔刷光标
    removeBrushCursor();
    
    // 解绑笔刷事件
    unbindBrushEvents();
    
    console.log('退出笔刷模式');
}

/**
 * 绑定笔刷事件
 */
function bindBrushEvents() {
    const $viewports = $('#originalViewport, #translatedViewport');
    
    $viewports.on('mousedown.brush', handleBrushMouseDown);
    $viewports.on('wheel.brush', handleBrushWheel);
    $(document).on('mousemove.brush', handleBrushMouseMove);
    $(document).on('mouseup.brush', handleBrushMouseUp);
}

/**
 * 解绑笔刷事件
 */
function unbindBrushEvents() {
    const $viewports = $('#originalViewport, #translatedViewport');
    
    $viewports.off('mousedown.brush');
    $viewports.off('wheel.brush');
    $(document).off('mousemove.brush');
    $(document).off('mouseup.brush');
}

/**
 * 更新笔刷光标
 */
function updateBrushCursor() {
    // 移除旧光标
    $('.brush-cursor').remove();
    
    if (!brushMode) return;
    
    // 创建圆形笔刷光标
    const color = brushMode === 'repair' ? 'rgba(76, 175, 80, 0.6)' : 'rgba(33, 150, 243, 0.6)';
    const borderColor = brushMode === 'repair' ? '#4CAF50' : '#2196F3';
    
    const $cursor = $('<div class="brush-cursor"></div>').css({
        position: 'fixed',
        width: brushSize + 'px',
        height: brushSize + 'px',
        borderRadius: '50%',
        border: `2px solid ${borderColor}`,
        backgroundColor: color,
        pointerEvents: 'none',
        zIndex: 99999,
        transform: 'translate(-50%, -50%)',
        display: 'none'
    });
    
    $('body').append($cursor);
}

/**
 * 移除笔刷光标
 */
function removeBrushCursor() {
    $('.brush-cursor').remove();
}

/**
 * 处理笔刷鼠标按下
 */
function handleBrushMouseDown(e) {
    if (!brushMode || e.button !== 0) return;
    
    e.preventDefault();
    e.stopPropagation();
    
    isBrushPainting = true;
    activeViewport = e.currentTarget;
    brushPath = [];
    
    // 获取相对于图片的坐标
    const pos = getBrushPositionInImage(e);
    if (pos) {
        brushPath.push(pos);
        // 创建临时画布用于显示涂抹效果
        createBrushCanvas();
        drawBrushStroke(pos);
    }
}

/**
 * 处理笔刷鼠标移动
 */
function handleBrushMouseMove(e) {
    // 更新光标位置
    const $cursor = $('.brush-cursor');
    if ($cursor.length && brushMode) {
        $cursor.css({
            left: e.clientX + 'px',
            top: e.clientY + 'px',
            display: 'block'
        });
    }
    
    if (!isBrushPainting || !brushMode) return;
    
    const pos = getBrushPositionInImage(e);
    if (pos) {
        brushPath.push(pos);
        drawBrushStroke(pos);
    }
}

/**
 * 处理笔刷鼠标抬起
 */
function handleBrushMouseUp(e) {
    if (!isBrushPainting) return;
    
    finishBrushPainting();
}

/**
 * 处理笔刷滚轮（调整笔刷大小）
 */
function handleBrushWheel(e) {
    if (!brushMode) return;
    
    e.preventDefault();
    e.stopPropagation();
    
    const delta = e.originalEvent.deltaY > 0 ? -5 : 5;
    brushSize = Math.max(brushMinSize, Math.min(brushMaxSize, brushSize + delta));
    
    // 更新光标大小
    $('.brush-cursor').css({
        width: brushSize + 'px',
        height: brushSize + 'px'
    });
    
    // 更新显示
    $('#brushSizeDisplay').text(brushSize + 'px');
}

/**
 * 获取笔刷在图片中的位置
 */
function getBrushPositionInImage(e) {
    const viewport = activeViewport || e.currentTarget;
    if (!viewport) return null;
    
    const $wrapper = $(viewport).find('.image-canvas-wrapper');
    const $img = $wrapper.find('img');
    
    if (!$img.length || !$img[0].naturalWidth) return null;
    
    const rect = $wrapper[0].getBoundingClientRect();
    const transform = $wrapper.css('transform');
    let scale = 1, translateX = 0, translateY = 0;
    
    if (transform && transform !== 'none') {
        const matrix = new DOMMatrix(transform);
        scale = matrix.a;
        translateX = matrix.e;
        translateY = matrix.f;
    }
    
    // 计算相对于图片的坐标
    const imgX = (e.clientX - rect.left) / scale;
    const imgY = (e.clientY - rect.top) / scale;
    
    // 确保坐标在图片范围内
    const imgWidth = $img[0].naturalWidth;
    const imgHeight = $img[0].naturalHeight;
    
    if (imgX < 0 || imgY < 0 || imgX > imgWidth || imgY > imgHeight) {
        return null;
    }
    
    return { x: imgX, y: imgY, scale: scale };
}

/**
 * 创建笔刷临时画布
 */
function createBrushCanvas() {
    // 移除旧画布
    $('#brushOverlayCanvas').remove();
    
    const $viewport = $(activeViewport);
    const $wrapper = $viewport.find('.image-canvas-wrapper');
    const $img = $wrapper.find('img');
    
    if (!$img.length) return;
    
    const canvas = document.createElement('canvas');
    canvas.id = 'brushOverlayCanvas';
    canvas.width = $img[0].naturalWidth;
    canvas.height = $img[0].naturalHeight;
    canvas.style.cssText = 'position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 100;';
    
    $wrapper.append(canvas);
    brushCanvas = canvas;
    brushCtx = canvas.getContext('2d');
}

/**
 * 绘制笔刷笔画
 */
function drawBrushStroke(pos) {
    if (!brushCtx || !pos) return;
    
    const color = brushMode === 'repair' ? 'rgba(76, 175, 80, 0.4)' : 'rgba(33, 150, 243, 0.4)';
    const radius = brushSize / 2 / (pos.scale || 1);
    
    brushCtx.beginPath();
    brushCtx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
    brushCtx.fillStyle = color;
    brushCtx.fill();
}

/**
 * 完成笔刷涂抹
 */
async function finishBrushPainting() {
    if (!isBrushPainting || brushPath.length === 0) {
        isBrushPainting = false;
        return;
    }
    
    isBrushPainting = false;
    
    const currentImage = state.getCurrentImage();
    if (!currentImage) {
        removeBrushCanvas();
        return;
    }
    
    // 获取涂抹区域的边界
    const bounds = getBrushPathBounds();
    if (!bounds) {
        removeBrushCanvas();
        return;
    }
    
    const mode = brushMode;
    
    try {
        if (mode === 'restore') {
            // 还原笔刷：将涂抹区域恢复为原图
            await restoreBrushArea(currentImage, bounds);
        } else if (mode === 'repair') {
            // 修复笔刷：使用填充色填充涂抹区域
            await repairBrushArea(currentImage, bounds);
        }
        
        // 重新渲染（不使用静默模式，确保更新UI）
        await reRenderFullImage(false, false);
        
        // 手动更新编辑模式下的翻译图显示
        const updatedImage = state.getCurrentImage();
        if (updatedImage && updatedImage.translatedDataURL) {
            updateTranslatedImageNew(updatedImage.translatedDataURL);
        }
        
    } catch (error) {
        console.error('笔刷操作失败:', error);
        ui.showGeneralMessage('笔刷操作失败', 'error');
    }
    
    removeBrushCanvas();
    brushPath = [];
}

/**
 * 获取笔刷路径的边界
 */
function getBrushPathBounds() {
    if (brushPath.length === 0) return null;
    
    const scale = brushPath[0].scale || 1;
    const radius = brushSize / 2 / scale;
    
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;
    
    for (const pos of brushPath) {
        minX = Math.min(minX, pos.x - radius);
        minY = Math.min(minY, pos.y - radius);
        maxX = Math.max(maxX, pos.x + radius);
        maxY = Math.max(maxY, pos.y + radius);
    }
    
    return {
        x1: Math.max(0, Math.floor(minX)),
        y1: Math.max(0, Math.floor(minY)),
        x2: Math.ceil(maxX),
        y2: Math.ceil(maxY),
        path: brushPath,
        radius: radius
    };
}

/**
 * 还原笔刷区域（恢复为原图）
 */
async function restoreBrushArea(currentImage, bounds) {
    if (!currentImage.originalDataURL) return;
    
    // 获取当前干净背景
    let cleanSrc;
    if (currentImage.cleanImageData) {
        cleanSrc = 'data:image/png;base64,' + currentImage.cleanImageData;
    } else {
        cleanSrc = currentImage.originalDataURL;
    }
    
    return new Promise((resolve) => {
        const cleanImg = new Image();
        const originalImg = new Image();
        let loadedCount = 0;
        
        const onLoad = () => {
            loadedCount++;
            if (loadedCount < 2) return;
            
            const canvas = document.createElement('canvas');
            canvas.width = cleanImg.naturalWidth;
            canvas.height = cleanImg.naturalHeight;
            const ctx = canvas.getContext('2d');
            
            // 先绘制当前干净背景
            ctx.drawImage(cleanImg, 0, 0);
            
            // 创建笔刷蒙版
            const maskCanvas = document.createElement('canvas');
            maskCanvas.width = canvas.width;
            maskCanvas.height = canvas.height;
            const maskCtx = maskCanvas.getContext('2d');
            
            // 绘制笔刷路径作为蒙版
            maskCtx.fillStyle = 'white';
            for (const pos of bounds.path) {
                maskCtx.beginPath();
                maskCtx.arc(pos.x, pos.y, bounds.radius, 0, Math.PI * 2);
                maskCtx.fill();
            }
            
            // 使用蒙版从原图恢复
            ctx.globalCompositeOperation = 'destination-out';
            ctx.drawImage(maskCanvas, 0, 0);
            ctx.globalCompositeOperation = 'destination-over';
            ctx.drawImage(originalImg, 0, 0);
            ctx.globalCompositeOperation = 'source-over';
            
            // 更新 cleanImageData
            currentImage.cleanImageData = canvas.toDataURL('image/png').split(',')[1];
            console.log('还原笔刷区域完成');
            resolve();
        };
        
        cleanImg.onload = onLoad;
        cleanImg.onerror = resolve;
        originalImg.onload = onLoad;
        originalImg.onerror = resolve;
        
        cleanImg.src = cleanSrc;
        originalImg.src = currentImage.originalDataURL;
    });
}

/**
 * 修复笔刷区域（根据填充方式选择纯色或 LAMA）
 */
async function repairBrushArea(currentImage, bounds) {
    // 获取当前填充方式
    const inpaintMethod = $('#bubbleInpaintMethodNew').val() || 'solid';
    
    // 判断是否使用 LAMA（lama_mpe 或 litelama）
    const isLamaMethod = inpaintMethod === 'lama_mpe' || inpaintMethod === 'litelama';
    if (isLamaMethod) {
        // 使用 LAMA 修复
        await repairBrushAreaWithLama(currentImage, bounds, inpaintMethod);
    } else {
        // 使用纯色填充
        await repairBrushAreaWithColor(currentImage, bounds);
    }
}

/**
 * 使用纯色填充修复笔刷区域
 */
async function repairBrushAreaWithColor(currentImage, bounds) {
    // 获取填充色
    const fillColor = $('#fillColorNew').val() || currentImage.fillColor || state.defaultFillColor || '#FFFFFF';
    
    // 获取当前干净背景
    let cleanSrc;
    if (currentImage.cleanImageData) {
        cleanSrc = 'data:image/png;base64,' + currentImage.cleanImageData;
    } else if (currentImage.originalDataURL) {
        cleanSrc = currentImage.originalDataURL;
    } else {
        return;
    }
    
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            const ctx = canvas.getContext('2d');
            
            // 绘制当前背景
            ctx.drawImage(img, 0, 0);
            
            // 用填充色绘制笔刷路径
            ctx.fillStyle = fillColor;
            for (const pos of bounds.path) {
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, bounds.radius, 0, Math.PI * 2);
                ctx.fill();
            }
            
            // 更新 cleanImageData
            currentImage.cleanImageData = canvas.toDataURL('image/png').split(',')[1];
            console.log('修复笔刷区域完成（纯色填充）');
            resolve();
        };
        img.onerror = resolve;
        img.src = cleanSrc;
    });
}

/**
 * 使用 LAMA 修复笔刷区域
 * 支持精确掩膜：根据用户的笔刷路径生成掩膜，而非使用外接矩形
 */
async function repairBrushAreaWithLama(currentImage, bounds, inpaintMethod = 'lama_mpe') {
    // 获取当前干净背景或原图
    let baseImageData;
    if (currentImage.cleanImageData) {
        baseImageData = currentImage.cleanImageData;
    } else if (currentImage.originalDataURL) {
        baseImageData = currentImage.originalDataURL.split(',')[1];
    } else {
        console.error('无法获取基础图像用于 LAMA 修复');
        return;
    }
    
    // 获取图像尺寸
    const img = $('#originalImageDisplay')[0];
    const imgWidth = img?.naturalWidth || 2000;
    const imgHeight = img?.naturalHeight || 2000;
    
    // 生成精确的笔刷掩膜（白色=需要修复的区域，黑色=保留的区域）
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = imgWidth;
    maskCanvas.height = imgHeight;
    const maskCtx = maskCanvas.getContext('2d');
    
    // 填充黑色背景（保留区域）
    maskCtx.fillStyle = 'black';
    maskCtx.fillRect(0, 0, imgWidth, imgHeight);
    
    // 用白色绘制笔刷路径（需要修复的区域）
    maskCtx.fillStyle = 'white';
    for (const pos of bounds.path) {
        maskCtx.beginPath();
        maskCtx.arc(pos.x, pos.y, bounds.radius, 0, Math.PI * 2);
        maskCtx.fill();
    }
    
    // 将掩膜转换为 base64
    const maskDataUrl = maskCanvas.toDataURL('image/png');
    const maskBase64 = maskDataUrl.split(',')[1];
    
    // 将笔刷路径边界转换为矩形坐标（用于边界检查）
    const coords = [bounds.x1, bounds.y1, bounds.x2, bounds.y2];
    
    // 确定 LAMA 模型类型
    const lamaModel = (inpaintMethod === 'litelama') ? 'litelama' : 'lama_mpe';
    
    try {
        ui.showLoading('LAMA 修复中...');
        
        const response = await api.inpaintSingleBubbleApi({
            image_data: baseImageData,
            bubble_coords: coords,
            mask_data: maskBase64,  // 传递精确掩膜
            method: 'lama',
            lama_model: lamaModel
        });
        
        ui.hideLoading();
        
        if (response.success && response.inpainted_image) {
            currentImage.cleanImageData = response.inpainted_image;
            console.log('修复笔刷区域完成（LAMA 修复，精确掩膜）');
        } else {
            throw new Error(response.error || 'LAMA 修复返回无效数据');
        }
    } catch (error) {
        ui.hideLoading();
        console.error('LAMA 修复失败，回退到纯色填充:', error);
        ui.showGeneralMessage('LAMA 修复失败，使用纯色填充', 'warning', false, 2000);
        // 回退到纯色填充
        await repairBrushAreaWithColor(currentImage, bounds);
    }
}

/**
 * 移除笔刷临时画布
 */
function removeBrushCanvas() {
    $('#brushOverlayCanvas').remove();
    brushCanvas = null;
    brushCtx = null;
}

/**
 * 导出笔刷模式状态
 */
export function isBrushModeActive() {
    return brushMode !== null;
}

// ============ 50音软键盘功能 ============

/**
 * 初始化50音软键盘
 */
export function initKanaKeyboard() {
    const $keyboard = $('#kanaKeyboard');
    const $toggleBtn = $('#toggleKanaKeyboard');
    const $closeBtn = $('#closeKanaKeyboard');
    
    if (!$keyboard.length) return;
    
    // 切换键盘显示/隐藏
    $toggleBtn.off('click').on('click', function() {
        const isVisible = $keyboard.is(':visible');
        if (isVisible) {
            $keyboard.slideUp(200);
            $(this).removeClass('active');
        } else {
            $keyboard.slideDown(200);
            $(this).addClass('active');
        }
    });
    
    // 关闭按钮
    $closeBtn.off('click').on('click', function() {
        $keyboard.slideUp(200);
        $toggleBtn.removeClass('active');
    });
    
    // 标签页切换
    $('.kana-tab').off('click').on('click', function() {
        const tab = $(this).data('tab');
        $('.kana-tab').removeClass('active');
        $(this).addClass('active');
        $('.kana-tab-content').removeClass('active');
        $(`.kana-tab-content[data-content="${tab}"]`).addClass('active');
    });
    
    // 假名按键点击事件
    $('.kana-key').off('click').on('click', function() {
        const $key = $(this);
        const mode = $('input[name="kanaMode"]:checked').val(); // 'hiragana' or 'katakana'
        const targetId = $('#kanaTargetSelect').val(); // 'original' or 'translated'
        
        let char = '';
        
        // 特殊字符（如标点符号）
        if ($key.hasClass('special-key')) {
            char = $key.data('char');
        } else {
            // 平假名或片假名
            const hiragana = $key.data('h');
            const katakana = $key.data('k');
            
            if (mode === 'hiragana' && hiragana) {
                char = hiragana;
            } else if (katakana) {
                char = katakana;
            } else if (hiragana) {
                // 如果没有片假名（如外来语专用），使用片假名显示
                char = hiragana;
            }
        }
        
        if (char) {
            insertCharToTarget(char, targetId);
            // 按键视觉反馈
            $key.addClass('pressed');
            setTimeout(() => $key.removeClass('pressed'), 100);
        }
    });
    
    // 退格键
    $('#kanaBackspace').off('click').on('click', function() {
        const targetId = $('#kanaTargetSelect').val();
        deleteCharFromTarget(targetId);
    });
    
    console.log('50音键盘初始化完成');
}

/**
 * 向目标文本框插入字符
 */
function insertCharToTarget(char, targetId) {
    const $target = targetId === 'original' ? $('#originalTextEditor') : $('#translatedTextEditor');
    
    if (!$target.length) return;
    
    const textarea = $target[0];
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const text = textarea.value;
    
    // 在光标位置插入字符
    textarea.value = text.substring(0, start) + char + text.substring(end);
    
    // 移动光标到插入字符后
    const newPos = start + char.length;
    textarea.selectionStart = newPos;
    textarea.selectionEnd = newPos;
    
    // 聚焦到文本框
    textarea.focus();
    
    // 触发input事件以便其他监听器能够响应
    $target.trigger('input');
}

/**
 * 从目标文本框删除一个字符
 */
function deleteCharFromTarget(targetId) {
    const $target = targetId === 'original' ? $('#originalTextEditor') : $('#translatedTextEditor');
    
    if (!$target.length) return;
    
    const textarea = $target[0];
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const text = textarea.value;
    
    if (start === end) {
        // 没有选中文本，删除光标前一个字符
        if (start > 0) {
            textarea.value = text.substring(0, start - 1) + text.substring(end);
            textarea.selectionStart = start - 1;
            textarea.selectionEnd = start - 1;
        }
    } else {
        // 有选中文本，删除选中部分
        textarea.value = text.substring(0, start) + text.substring(end);
        textarea.selectionStart = start;
        textarea.selectionEnd = start;
    }
    
    textarea.focus();
    $target.trigger('input');
}

/**
 * 显示/隐藏50音键盘
 */
export function toggleKanaKeyboard(show) {
    const $keyboard = $('#kanaKeyboard');
    const $toggleBtn = $('#toggleKanaKeyboard');
    
    if (show === undefined) {
        show = !$keyboard.is(':visible');
    }
    
    if (show) {
        $keyboard.slideDown(200);
        $toggleBtn.addClass('active');
    } else {
        $keyboard.slideUp(200);
        $toggleBtn.removeClass('active');
    }
}

// ============================================================
// ============ 旋转手柄功能 ============
// ============================================================

/**
 * 绑定旋转手柄事件
 */
function bindRotateHandleEvents() {
    // 旋转手柄 mousedown
    $('.rotate-handle').off('mousedown.rotate').on('mousedown.rotate', function(e) {
        if (e.button !== 0) return; // 只响应左键
        e.preventDefault();
        e.stopPropagation();
        
        const $handle = $(this);
        const index = parseInt($handle.attr('data-parent-index'));
        const $box = $handle.closest('.bubble-highlight-box');
        
        startRotating(e, $box, index);
    });
}

/**
 * 开始旋转
 */
function startRotating(e, $box, index) {
    isRotatingBox = true;
    
    // 选中这个气泡
    if (index !== state.selectedBubbleIndex) {
        selectBubbleNew(index);
    }
    
    // 获取框的中心点（相对于视口）
    const boxRect = $box[0].getBoundingClientRect();
    rotateCenterX = boxRect.left + boxRect.width / 2;
    rotateCenterY = boxRect.top + boxRect.height / 2;
    
    // 计算鼠标当前位置相对于中心的角度
    const dx = e.clientX - rotateCenterX;
    const dy = e.clientY - rotateCenterY;
    rotateStartAngle = Math.atan2(dy, dx) * 180 / Math.PI;
    
    // 获取当前旋转角度
    rotateInitialAngle = parseFloat($box.attr('data-rotation')) || 0;
    
    // 添加旋转光标
    $('body').addClass('rotating-box');
    
    // 绑定全局事件
    $(document).on('mousemove.rotate', handleRotateMouseMove);
    $(document).on('mouseup.rotate', handleRotateMouseUp);
    
    console.log(`开始旋转气泡 #${index + 1}，初始角度: ${rotateInitialAngle}°`);
}

/**
 * 处理旋转鼠标移动
 */
function handleRotateMouseMove(e) {
    if (!isRotatingBox) return;
    
    // 计算鼠标当前位置相对于中心的角度
    const dx = e.clientX - rotateCenterX;
    const dy = e.clientY - rotateCenterY;
    const currentAngle = Math.atan2(dy, dx) * 180 / Math.PI;
    
    // 计算旋转差值
    let deltaAngle = currentAngle - rotateStartAngle;
    
    // 计算新角度
    let newAngle = rotateInitialAngle + deltaAngle;
    
    // 限制角度范围 -180 到 180
    while (newAngle > 180) newAngle -= 360;
    while (newAngle < -180) newAngle += 360;
    
    // 按住 Shift 键时吸附到 15° 的倍数
    if (e.shiftKey) {
        newAngle = Math.round(newAngle / 15) * 15;
    }
    
    // 实时更新所有同索引的高亮框旋转
    const index = state.selectedBubbleIndex;
    $(`.bubble-highlight-box[data-index="${index}"]`).each(function() {
        $(this).css('transform', `rotate(${newAngle}deg)`);
        $(this).attr('data-rotation', newAngle);
    });
    
    // 更新旋转角度输入框
    $('#rotationAngleNew').val(Math.round(newAngle));
}

/**
 * 处理旋转鼠标释放
 */
function handleRotateMouseUp(e) {
    if (!isRotatingBox) return;
    
    isRotatingBox = false;
    $('body').removeClass('rotating-box');
    
    // 解绑全局事件
    $(document).off('mousemove.rotate');
    $(document).off('mouseup.rotate');
    
    // 获取最终角度
    const index = state.selectedBubbleIndex;
    const $box = $(`.bubble-highlight-box[data-index="${index}"]`).first();
    const finalAngle = parseFloat($box.attr('data-rotation')) || 0;
    
    // 更新气泡设置中的旋转角度
    if (index >= 0 && state.bubbleStates && state.bubbleStates[index]) {
        state.updateSingleBubbleState(index, { rotationAngle: finalAngle });
        console.log(`气泡 #${index + 1} 旋转角度已更新为: ${finalAngle}°`);
        
        // 触发重新渲染
        triggerDelayedPreview(index);
    }
}

// ============================================================
// ============ 从标注模式迁移的功能 ============
// ============================================================

/**
 * 自动检测当前图片的文本框
 * (从标注模式迁移)
 */
export async function autoDetectBubbles() {
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.originalDataURL) {
        ui.showGeneralMessage("没有有效的图片用于检测。", "warning");
        return;
    }

    ui.showLoading("正在自动检测文本框...");

    const imageData = currentImage.originalDataURL.split(',')[1];

    try {
        const response = await api.detectBoxesApi(imageData);
        ui.hideLoading();
        
        if (response.success && response.bubble_coords) {
            // 保存检测到的坐标和角度
            currentImage.bubbleCoords = response.bubble_coords;
            currentImage.bubbleAngles = response.bubble_angles || [];
            
            // 【重要】保存自动检测的排版方向
            const autoDirections = response.auto_directions || [];
            
            // 初始化空的文本数组
            if (!currentImage.bubbleTexts) currentImage.bubbleTexts = [];
            if (!currentImage.originalTexts) currentImage.originalTexts = [];
            
            // 确保数组长度一致
            while (currentImage.bubbleTexts.length < currentImage.bubbleCoords.length) {
                currentImage.bubbleTexts.push('');
            }
            while (currentImage.originalTexts.length < currentImage.bubbleCoords.length) {
                currentImage.originalTexts.push('');
            }
            
            // 重新初始化气泡状态，并传入自动方向
            initBubbleStates(autoDirections);
            
            // 更新高亮框显示
            updateBubbleHighlightsNew();
            
            // 更新导航器
            updateBubbleNavigator();
            
            // 选中第一个气泡
            if (state.bubbleStates.length > 0) {
                selectBubbleNew(0);
            }
            
            ui.showGeneralMessage(`自动检测到 ${response.bubble_coords.length} 个文本框。`, "success");
            console.log(`自动检测完成，检测到 ${response.bubble_coords.length} 个文本框`);
        } else {
            throw new Error(response.error || "检测失败，未返回坐标。");
        }
    } catch (error) {
        ui.hideLoading();
        ui.showGeneralMessage(`自动检测失败: ${error.message}`, "error");
        console.error("自动检测失败:", error);
    }
}

/**
 * 更新编辑模式进度条
 * @param {number} current - 当前进度
 * @param {number} total - 总数
 * @param {string} text - 显示文本
 */
function updateEditProgress(current, total, text = "检测中") {
    const container = $("#editProgressContainer");
    const progressText = $("#editProgressText");
    const progressCount = $("#editProgressCount");
    const progressFill = $("#editProgressFill");
    
    const percentage = total > 0 ? Math.floor((current / total) * 100) : 0;
    
    progressText.text(text);
    progressCount.text(`${current}/${total}`);
    progressFill.css('width', percentage + '%');
    
    if (current < total) {
        progressFill.addClass('animating');
        container.removeClass('completed');
    } else {
        progressFill.removeClass('animating');
        container.addClass('completed');
    }
}

/**
 * 显示编辑模式进度条
 * @param {string} text - 初始文本
 * @param {number} total - 总数
 */
function showEditProgress(text, total) {
    const container = $("#editProgressContainer");
    container.removeClass('completed').show();
    updateEditProgress(0, total, text);
}

/**
 * 隐藏编辑模式进度条
 * @param {number} delay - 延迟隐藏时间（毫秒）
 */
function hideEditProgress(delay = 1500) {
    setTimeout(() => {
        $("#editProgressContainer").fadeOut(300);
    }, delay);
}

/**
 * 批量检测所有图片的文本框
 * (从标注模式迁移)
 */
export async function detectAllImages() {
    if (state.images.length <= 1) {
        ui.showGeneralMessage("至少需要两张图片才能执行批量检测。", "warning");
        return;
    }

    // 确认对话框
    if (!confirm("此操作将对所有图片进行文本框检测，可能会覆盖已有的检测结果。确定继续吗？")) {
        return;
    }

    // 记录当前索引，以便处理完后恢复
    const originalIndex = state.currentImageIndex;
    const totalImages = state.images.length;
    
    // 显示编辑模式进度条
    showEditProgress("批量检测中", totalImages);
    
    // 同时更新通用进度条（用于非编辑模式场景）
    ui.updateProgressBar(0, `0/${totalImages}`);
    $("#translationProgressBar").show();
    
    let totalDetected = 0;
    
    // 处理每张图片
    for (let index = 0; index < totalImages; index++) {
        // 更新编辑模式进度条（显示"正在处理第X张"）
        updateEditProgress(index + 1, totalImages, "批量检测中");
        
        // 同时更新通用进度条
        const progress = Math.floor(((index + 1) / totalImages) * 100);
        ui.updateProgressBar(progress, `${index + 1}/${totalImages}`);
        
        // 获取当前处理的图片
        const image = state.images[index];
        if (!image || !image.originalDataURL) {
            continue; // 跳过无效图片
        }
        
        // 获取图片数据
        const imageData = image.originalDataURL.split(',')[1];
        
        try {
            // 调用API检测文本框
            const response = await api.detectBoxesApi(imageData);
            
            if (response.success && response.bubble_coords) {
                // 保存检测到的坐标和角度到图片对象
                image.bubbleCoords = response.bubble_coords;
                image.bubbleAngles = response.bubble_angles || [];
                
                // 【重要】保存自动检测的排版方向
                const autoDirections = response.auto_directions || [];
                
                // 初始化空的文本数组
                if (!image.bubbleTexts) image.bubbleTexts = [];
                if (!image.originalTexts) image.originalTexts = [];
                while (image.bubbleTexts.length < image.bubbleCoords.length) {
                    image.bubbleTexts.push('');
                }
                while (image.originalTexts.length < image.bubbleCoords.length) {
                    image.originalTexts.push('');
                }
                
                // 创建包含 autoTextDirection 的 bubbleStates
                const defaults = getDefaultBubbleSettings();
                image.bubbleStates = image.bubbleCoords.map((coords, i) => {
                    let autoDir;
                    if (autoDirections && i < autoDirections.length) {
                        autoDir = autoDirections[i] === 'v' ? 'vertical' : 'horizontal';
                    } else {
                        const [x1, y1, x2, y2] = coords;
                        autoDir = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal';
                    }
                    return bubbleStateModule.createBubbleState({
                        translatedText: image.bubbleTexts[i] || "",
                        coords: coords,
                        ...defaults,
                        autoTextDirection: autoDir,
                        rotationAngle: image.bubbleAngles[i] || 0
                    });
                });
                
                totalDetected += response.bubble_coords.length;
                
                // 如果是当前图片，同时更新显示
                if (index === state.currentImageIndex) {
                    initBubbleStates(autoDirections);
                    updateBubbleHighlightsNew();
                    updateBubbleNavigator();
                }
            }
        } catch (error) {
            console.error(`图片 ${index} 检测出错:`, error);
            // 继续处理下一张
        }
    }
    
    // 完成 - 更新编辑模式进度条
    updateEditProgress(totalImages, totalImages, "检测完成");
    hideEditProgress(2000);
    
    // 同时更新通用进度条
    ui.updateProgressBar(100, `${totalImages}/${totalImages}`);
    setTimeout(() => $("#translationProgressBar").hide(), 1000);
    
    // 返回到原始图片并刷新显示
    if (originalIndex !== state.currentImageIndex) {
        navigateToImage(originalIndex);
    }
    
    ui.showGeneralMessage(`批量检测完成！共处理 ${totalImages} 张图片，检测到 ${totalDetected} 个文本框。`, "success");
    ui.renderThumbnails(); // 更新缩略图显示
}

/**
 * 使用当前气泡框进行翻译
 * (从标注模式迁移)
 */
export async function translateWithCurrentBubbles() {
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.originalDataURL) {
        ui.showGeneralMessage("没有有效的图片用于翻译。", "warning");
        return;
    }
    
    if (!currentImage.bubbleCoords || currentImage.bubbleCoords.length === 0) {
        ui.showGeneralMessage("没有文本框可用于翻译。请先检测或添加文本框。", "warning");
        return;
    }
    
    // 显示加载提示
    ui.showLoading("正在使用当前文本框翻译...");
    
    // 准备参数
    const imageData = currentImage.originalDataURL.split(',')[1];
    
    // 组装翻译请求参数
    const requestData = {
        image: imageData,
        
        // 基础参数
        model_provider: $('#modelProvider').val(),
        model_name: $('#modelName').val(),
        api_key: $('#apiKey').val(),
        source_language: $('#sourceLanguage').val(),
        target_language: $('#targetLanguage').val(),
        
        // 提示词参数
        prompt_content: $('#promptContent').val(),
        textbox_prompt_content: $('#textboxPromptContent').val(),
        use_textbox_prompt: $('#enableTextboxPrompt').is(':checked'),
        
        // 字体参数
        fontSize: $('#autoFontSize').is(':checked') ? 'auto' : parseInt($('#fontSize').val()),
        autoFontSize: $('#autoFontSize').is(':checked'),
        fontFamily: $('#fontFamily').val(),
        textDirection: ($('#layoutDirection').val() === 'auto') ? 'vertical' : $('#layoutDirection').val(),
        
        // 颜色参数
        textColor: $('#textColor').val(),
        fillColor: $('#fillColor').val(),
        rotationAngle: 0,
        
        // 修复参数
        use_inpainting: $('#useInpainting').val() === 'inpainting',
        use_lama: $('#useInpainting').val() === 'lama_mpe' || $('#useInpainting').val() === 'litelama',
        lamaModel: ($('#useInpainting').val() === 'litelama') ? 'litelama' : 'lama_mpe',
        
        // OCR引擎参数
        ocr_engine: $('#ocrEngine').val(),
        
        // 百度OCR参数
        baidu_api_key: $('#baiduApiKey').val(),
        baidu_secret_key: $('#baiduSecretKey').val(),
        baidu_version: $('#baiduVersion').val() || 'standard',
        baidu_ocr_language: $('#baiduOcrLanguage').val() || 'auto_detect',
        
        // AI视觉OCR参数
        ai_vision_provider: $('#aiVisionProvider').val(),
        ai_vision_api_key: $('#aiVisionApiKey').val(),
        ai_vision_model_name: $('#aiVisionModelName').val(),
        ai_vision_ocr_prompt: $('#aiVisionOcrPrompt').val(),
        custom_ai_vision_base_url: state.customAiVisionBaseUrl,
        
        // 自定义OpenAI Base URL
        custom_base_url: state.customBaseUrl,
        
        // JSON格式参数
        use_json_format_translation: state.isTranslateJsonMode || false,
        use_json_format_ai_vision_ocr: state.isAiVisionOcrJsonMode || false,
        
        // 描边参数
        strokeEnabled: $('#strokeEnabled').is(':checked') || false,
        strokeColor: $('#strokeColor').val() || '#ffffff',
        strokeWidth: parseInt($('#strokeWidth').val() || 3),
        
        // 坐标和角度参数（角度优先从 bubbleStates 提取）
        bubble_coords: currentImage.bubbleCoords,
        bubble_angles: (currentImage.bubbleStates && currentImage.bubbleStates.length > 0) 
            ? currentImage.bubbleStates.map(s => s.rotationAngle || 0) 
            : (currentImage.bubbleAngles || null)
    };
    
    try {
        const response = await api.translateImageApi(requestData);
        ui.hideLoading();
        
        if (response.translated_image) {
            // 更新当前图片的翻译结果数据
            currentImage.translatedDataURL = `data:image/png;base64,${response.translated_image}`;
            currentImage.cleanImageData = response.clean_image || null;
            currentImage.bubbleCoords = response.bubble_coords || currentImage.bubbleCoords;
            currentImage.bubbleAngles = response.bubble_angles || currentImage.bubbleAngles || [];
            currentImage.originalTexts = response.original_texts || [];
            currentImage.bubbleTexts = response.bubble_texts || [];
            currentImage.textboxTexts = response.textbox_texts || [];
            
            // 使用新的统一 bubble_states 格式保存气泡状态
            const newBubbleStates = bubbleStateModule.createBubbleStatesFromResponse(response, {
                fontSize: requestData.fontSize,
                fontFamily: requestData.fontFamily,
                textDirection: requestData.textDirection,
                textColor: requestData.textColor,
                fillColor: requestData.fillColor,
                inpaintMethod: requestData.use_lama ? requestData.lamaModel : 'solid',
                strokeEnabled: requestData.strokeEnabled,
                strokeColor: requestData.strokeColor,
                strokeWidth: requestData.strokeWidth
            });
            
            // 保存统一的气泡状态
            currentImage.bubbleStates = newBubbleStates;
            state.setBubbleStates(newBubbleStates);
            
            // 更新图像显示
            ui.updateTranslatedImage(currentImage.translatedDataURL);
            updateTranslatedImageNew(currentImage.translatedDataURL);
            
            // 更新编辑模式显示
            updateBubbleHighlightsNew();
            updateBubbleNavigator();
            if (state.bubbleStates.length > 0) {
                selectBubbleNew(0);
            }
            
            // 保存模型历史
            api.saveModelInfoApi(requestData.model_provider, requestData.model_name)
                .catch(err => console.warn("保存模型历史失败:", err));
            
            ui.showGeneralMessage("翻译成功！", "success", false, 3000);
        } else {
            throw new Error(response.error || "翻译失败，未返回翻译结果。");
        }
    } catch (error) {
        ui.hideLoading();
        console.error("翻译失败:", error);
        ui.showGeneralMessage(`翻译失败: ${error.message || "未知错误"}`, "error");
    }
}
