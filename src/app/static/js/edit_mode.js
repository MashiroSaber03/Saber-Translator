import * as state from './state.js';
import * as ui from './ui.js';
import * as api from './api.js';
import * as constants from './constants.js'; // <--- 添加导入
<<<<<<< HEAD
import * as session from './session.js'; // <--- 添加导入，用于自动存档
import * as main from './main.js'; // 导入main模块以使用loadImage函数
=======
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
// import $ from 'jquery'; // 假设 jQuery 已全局加载

/**
 * 切换编辑模式
 */
export function toggleEditMode() {
    state.setEditModeActive(!state.editModeActive); // 更新状态

    if (state.editModeActive) {
        // --- 进入编辑模式 ---
        const currentImage = state.getCurrentImage();
        if (!currentImage || !currentImage.bubbleCoords || currentImage.bubbleCoords.length === 0) { // 检查是否有气泡
            ui.showGeneralMessage("当前图片没有可编辑的气泡", "warning");
            state.setEditModeActive(false); // 切换失败，恢复状态
            return;
        }

        ui.toggleEditModeUI(true); // 更新 UI
        initBubbleSettings(); // 初始化或加载气泡设置
        // 保存初始设置备份
        state.setInitialBubbleSettings(JSON.parse(JSON.stringify(state.bubbleSettings)));
        console.log("已保存初始气泡设置:", state.initialBubbleSettings);

        // 默认选择第一个气泡
        if (state.bubbleSettings.length > 0) {
            selectBubble(0);
        } else {
            // 没有气泡，清空编辑区 (理论上不会到这里，因为前面检查了)
            ui.updateBubbleEditArea(-1);
        }
        // 确保有干净背景
        ensureCleanBackground();

    } else {
        // --- 退出编辑模式 ---
<<<<<<< HEAD
        ui.toggleEditModeUI(false); // 更新 UI (这会清除所有高亮框)
        
        // 确保清除任何可能存在的"重新渲染中..."消息
        ui.clearGeneralMessageById("rendering_loading_message");
        
        // 保存最终的气泡设置到当前图片对象
        const currentImage = state.getCurrentImage();
        if (currentImage && state.bubbleSettings.length > 0) { // 仅在有设置时保存
            // 保存气泡设置到图像对象，确保完整复制所有属性，包括fillColor
=======
        ui.toggleEditModeUI(false); // 更新 UI
        // 保存最终的气泡设置到当前图片对象
        const currentImage = state.getCurrentImage();
        if (currentImage && state.bubbleSettings.length > 0) { // 仅在有设置时保存
            // 保存气泡设置到图像对象
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
            currentImage.bubbleSettings = JSON.parse(JSON.stringify(state.bubbleSettings));
            // 同时更新 bubbleTexts
            currentImage.bubbleTexts = state.bubbleSettings.map(s => s.text);
            
            // 同步更新当前图片的全局属性，确保一致性
            const firstBubbleSetting = state.bubbleSettings[0];
            currentImage.fontSize = firstBubbleSetting.fontSize;
            currentImage.autoFontSize = firstBubbleSetting.autoFontSize;
            currentImage.fontFamily = firstBubbleSetting.fontFamily;
            currentImage.layoutDirection = firstBubbleSetting.textDirection;
            
<<<<<<< HEAD
            // 重新渲染以确保更改立即可见，并等待渲染完成后再继续
            reRenderFullImage()
                .then(() => {
                    console.log("退出编辑模式，已保存气泡设置到图像对象，渲染完成");
                    state.setSelectedBubbleIndex(-1); // 重置选中索引
                    state.setInitialBubbleSettings([]); // 清空备份
                    session.triggerAutoSave(); // 触发自动保存
                })
                .catch(err => {
                    console.error("退出编辑模式时渲染失败:", err);
                    ui.showGeneralMessage("退出编辑模式时渲染失败，请重试", "error");
                    // 即使渲染失败，仍然重置状态
                    state.setSelectedBubbleIndex(-1);
                    state.setInitialBubbleSettings([]);
                });
                
            // 添加安全超时，确保即使渲染未完成，也能在10秒后自动清除消息
            setTimeout(() => {
                ui.clearGeneralMessageById("rendering_loading_message");
            }, 10000);
        } else {
            state.setSelectedBubbleIndex(-1); // 重置选中索引
            state.setInitialBubbleSettings([]); // 清空备份
        }
=======
            // 重新渲染以确保更改立即可见
            reRenderFullImage();
            
            console.log("退出编辑模式，已保存气泡设置到图像对象");
        }
        state.setSelectedBubbleIndex(-1); // 重置选中索引
        state.setInitialBubbleSettings([]); // 清空备份
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
    }
}

/**
 * 初始化或加载当前图片的气泡设置
 */
<<<<<<< HEAD
export function initBubbleSettings() {
=======
function initBubbleSettings() {
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.bubbleCoords) {
        console.error("无法初始化气泡设置：无效的图像或坐标");
        state.setBubbleSettings([]);
        return;
    }

<<<<<<< HEAD
    const imageGlobalFillColor = currentImage.fillColor || state.defaultFillColor; // 当前图片的全局填充色

    if (currentImage.bubbleSettings && currentImage.bubbleSettings.length === currentImage.bubbleCoords.length) {
        console.log("加载当前图像已保存的气泡设置 (含填充色检查)");
        const loadedSettings = JSON.parse(JSON.stringify(currentImage.bubbleSettings));
        // 确保每个加载的设置都有 fillColor
        loadedSettings.forEach(setting => {
            if (!setting.hasOwnProperty('fillColor') || setting.fillColor === null || setting.fillColor === undefined) {
                setting.fillColor = imageGlobalFillColor; // 如果缺失，用图片的全局填充色
            }
        });
        state.setBubbleSettings(loadedSettings);
    } else {
        console.log("创建新的默认气泡设置 (含填充色)");
        const newSettings = [];
=======
    // 检查当前图像是否已有保存的气泡设置
    if (currentImage.bubbleSettings && currentImage.bubbleSettings.length === currentImage.bubbleCoords.length) {
        console.log("加载当前图像已保存的气泡设置");
        
        // 更新：确保使用最新的全局设置更新现有的气泡设置
        // 从当前UI控件读取全局设置
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
        const globalFontSize = $('#autoFontSize').is(':checked') ? 'auto' : parseInt($('#fontSize').val());
        const globalAutoFontSize = $('#autoFontSize').is(':checked');
        const globalFontFamily = $('#fontFamily').val();
        const globalTextDirection = $('#layoutDirection').val();
<<<<<<< HEAD
        const globalTextColor = $('#textColor').val(); // 全局文本颜色
        // globalFillColor 已经通过 imageGlobalFillColor 获取
=======
        const globalTextColor = $('#textColor').val();
        
        // 创建设置的深拷贝并更新为当前全局设置
        const updatedSettings = currentImage.bubbleSettings.map(setting => ({
            ...setting,
            fontSize: globalFontSize,
            autoFontSize: globalAutoFontSize,
            fontFamily: globalFontFamily,
            textDirection: globalTextDirection,
            textColor: globalTextColor
        }));
        
        state.setBubbleSettings(updatedSettings);
    } else {
        // 创建新的默认设置
        console.log("创建新的默认气泡设置");
        const newSettings = [];
        // 从当前UI控件读取全局设置 (而不是从state读取默认值)
        const globalFontSize = $('#autoFontSize').is(':checked') ? 'auto' : parseInt($('#fontSize').val());
        const globalAutoFontSize = $('#autoFontSize').is(':checked');
        const globalFontFamily = $('#fontFamily').val();
        const globalTextDirection = $('#layoutDirection').val();
        const globalTextColor = $('#textColor').val();
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915

        if (!currentImage.bubbleTexts || currentImage.bubbleTexts.length !== currentImage.bubbleCoords.length) {
            currentImage.bubbleTexts = Array(currentImage.bubbleCoords.length).fill("");
        }

        for (let i = 0; i < currentImage.bubbleCoords.length; i++) {
            newSettings.push({
                text: currentImage.bubbleTexts[i] || "",
                fontSize: globalFontSize,
                autoFontSize: globalAutoFontSize,
                fontFamily: globalFontFamily,
                textDirection: globalTextDirection,
                position: { x: 0, y: 0 },
<<<<<<< HEAD
                textColor: globalTextColor, // 使用全局文本颜色
                rotationAngle: 0,
                fillColor: imageGlobalFillColor // <--- 新增：使用图片的全局填充色初始化
            });
        }
        state.setBubbleSettings(newSettings);
        currentImage.bubbleSettings = JSON.parse(JSON.stringify(newSettings));
    }
    ui.updateBubbleListUI();
=======
                textColor: globalTextColor,
                rotationAngle: 0
            });
        }
        state.setBubbleSettings(newSettings);
        // 同时保存到当前图像对象
        currentImage.bubbleSettings = JSON.parse(JSON.stringify(newSettings));
    }
    ui.updateBubbleListUI(); // 更新列表显示
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
}

/**
 * 选择一个气泡进行编辑
 * @param {number} index - 要选择的气泡索引
 */
export function selectBubble(index) {
    if (index < 0 || index >= state.bubbleSettings.length) {
        console.warn(`选择气泡失败：无效索引 ${index}`);
        return;
    }
    state.setSelectedBubbleIndex(index);
    ui.updateBubbleEditArea(index);
<<<<<<< HEAD
    ui.updateBubbleHighlight(index); // 这会更新所有气泡的高亮框并标记选中的
    
    // 滚动气泡列表到选中项，但不滚动页面
    const bubbleItem = $(`.bubble-item[data-index="${index}"]`);
    if (bubbleItem.length) {
        // 仅滚动气泡列表容器，不影响页面滚动
        const bubbleList = $('.bubble-list');
        bubbleList.animate({
            scrollTop: bubbleItem.position().top + bubbleList.scrollTop() - bubbleList.position().top
        }, 300);
=======
    ui.updateBubbleHighlight(index);
    // 滚动气泡列表到选中项
    const bubbleItem = $(`.bubble-item[data-index="${index}"]`);
    if (bubbleItem.length) {
        // 使用原生 scrollIntoView
        bubbleItem[0].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
    }
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
    if (state.selectedBubbleIndex < state.bubbleSettings.length - 1) {
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
    
    state.updateSingleBubbleSetting(index, { text: newText });
    ui.updateBubbleListUI(); // 更新列表显示
    renderBubblePreview(index); // 立即触发渲染预览
}

/**
 * 渲染单个气泡的预览（通过重新渲染整个图像实现）
 * @param {number} bubbleIndex - 要预览的气泡索引
 */
export function renderBubblePreview(bubbleIndex) {
    if (bubbleIndex < 0 || bubbleIndex >= state.bubbleSettings.length) return;
    console.log(`请求渲染气泡 ${bubbleIndex} 的预览`);
    
    // 检查当前在预览的气泡是否是从自动字号切换到手动字号
    const currentImage = state.getCurrentImage();
    const bubbleSetting = state.bubbleSettings[bubbleIndex];
    
    // 检查是否有之前的气泡设置在当前图像中
    if (currentImage && currentImage.bubbleSettings && currentImage.bubbleSettings[bubbleIndex]) {
        const prevSetting = currentImage.bubbleSettings[bubbleIndex];
        const isPrevAuto = prevSetting.autoFontSize;
        const isNowAuto = bubbleSetting.autoFontSize;
        
        // 如果是从自动切换到非自动
        if (isPrevAuto && !isNowAuto) {
            console.log(`气泡 ${bubbleIndex} 从自动字号切换到手动字号`);
<<<<<<< HEAD
            reRenderFullImage(true); // 参数表示是从自动切换到手动
=======
            reRenderFullImage(false, true); // 第二个参数表示是从自动切换到手动
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
            return;
        }
    }
    
    // 正常情况
    reRenderFullImage();
}

/**
 * 重新渲染整个图像
<<<<<<< HEAD
 * @param {boolean} [fromAutoToManual=false] - (保留) 是否是从自动字号切换到手动字号,用于后端特殊处理
 * @returns {Promise<void>} - 在渲染成功时 resolve，失败时 reject
 */
export function reRenderFullImage(fromAutoToManual = false) {
    return new Promise(async (resolve, reject) => { // 改为 async 以便使用 await
        const currentImage = state.getCurrentImage();
        if (!currentImage || (!currentImage.translatedDataURL && !currentImage.originalDataURL)) {
            console.error("无法重新渲染：缺少必要数据");
            ui.showGeneralMessage("无法重新渲染，缺少图像或气泡数据", "error");
            reject(new Error("无法重新渲染：缺少必要数据"));
            return;
        }
        if (!currentImage.bubbleCoords || currentImage.bubbleCoords.length === 0) {
            console.log("reRenderFullImage: 当前图片没有气泡坐标，跳过重新渲染。");
            resolve();
            return;
        }

        // 使用固定消息ID，确保相同操作只显示一条消息
        const loadingMessageId = "rendering_loading_message";
        ui.showGeneralMessage("重新渲染中...", "info", false, 0, loadingMessageId);

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

            // 2. 根据 state.bubbleSettings 中的独立 fillColor 填充气泡区域
            // 只有在编辑模式下，并且有有效的 bubbleSettings 时才应用独立填充色
            let appliedIndividualFills = false;
            
            // 检查是否使用了LAMA修复，如果是则不进行填充
            const usesLamaInpainting = currentImage.hasOwnProperty('_lama_inpainted') && currentImage._lama_inpainted === true;
            
            if (usesLamaInpainting && currentImage.cleanImageData) {
                console.log("reRenderFullImage: 检测到使用LAMA修复，跳过纯色填充，直接使用LAMA修复的背景。");
                // 直接使用LAMA修复的干净背景
                preFilledBackgroundBase64 = currentImage.cleanImageData;
            } else {
                // 普通填充逻辑
                if (state.editModeActive && state.bubbleSettings && state.bubbleSettings.length === currentImage.bubbleCoords.length) {
                    currentImage.bubbleCoords.forEach((coords, i) => {
                        const bubbleSetting = state.bubbleSettings[i];
                        if (bubbleSetting && bubbleSetting.fillColor) {
                            const [x1, y1, x2, y2] = coords;
                            ctx.fillStyle = bubbleSetting.fillColor;
                            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
                            appliedIndividualFills = true;
                        } else {
                            // 如果某个气泡没有独立填充色，则使用图片的全局填充色
                            const imageFillColor = currentImage.fillColor || state.defaultFillColor;
                            const [x1, y1, x2, y2] = coords;
                            ctx.fillStyle = imageFillColor;
                            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
                        }
                    });
                    if(appliedIndividualFills) {
                        console.log("reRenderFullImage: 已在前端应用独立的或全局的气泡填充色。");
                    }
                } else if (currentImage.bubbleSettings && Array.isArray(currentImage.bubbleSettings) && 
                    currentImage.bubbleSettings.length > 0) {
                    // 非编辑模式，但有保存的bubbleSettings，检查是否有独立fillColor
                    currentImage.bubbleCoords.forEach((coords, i) => {
                        const bubbleSetting = currentImage.bubbleSettings[i];
                        if (bubbleSetting && bubbleSetting.fillColor) {
                            const [x1, y1, x2, y2] = coords;
                            ctx.fillStyle = bubbleSetting.fillColor;
                            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
                            appliedIndividualFills = true;
                        } else {
                            // 如果某个气泡没有独立填充色，则使用图片的全局填充色
                            const imageFillColor = currentImage.fillColor || state.defaultFillColor;
                            const [x1, y1, x2, y2] = coords;
                            ctx.fillStyle = imageFillColor;
                            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
                        }
                    });
                    if(appliedIndividualFills) {
                        console.log("reRenderFullImage (非编辑模式): 应用了保存的独立气泡填充色。");
                    }
                } else {
                    // 没有可用的bubbleSettings，使用图片的全局填充色填充所有气泡
                    const imageFillColor = currentImage.fillColor || state.defaultFillColor;
                    currentImage.bubbleCoords.forEach(coords => {
                        const [x1, y1, x2, y2] = coords;
                        ctx.fillStyle = imageFillColor;
                        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
                    });
                    console.log("reRenderFullImage: 已在前端应用图片的全局填充色。");
                }
                
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

        let currentTexts = [];
        if (state.editModeActive && state.bubbleSettings && state.bubbleSettings.length > 0) {
            currentTexts = state.bubbleSettings.map(setting => setting.text || "");
        } else {
            currentTexts = currentImage.bubbleTexts || [];
        }

        if (!currentTexts || currentTexts.length !== currentImage.bubbleCoords.length) {
            console.error("重新渲染错误：文本与坐标不匹配！");
            ui.showGeneralMessage("内部错误：文本与坐标数据不匹配", "error");
            ui.clearGeneralMessageById(loadingMessageId);
            reject(new Error("文本与坐标数据不匹配"));
            return;
        }
        
        let allBubbleStyles = [];
        if (state.editModeActive && state.bubbleSettings && state.bubbleSettings.length === currentImage.bubbleCoords.length) {
            allBubbleStyles = state.bubbleSettings.map(setting => ({
                fontSize: setting.fontSize || state.defaultFontSize,
                autoFontSize: setting.autoFontSize || false,
                fontFamily: setting.fontFamily || state.defaultFontFamily,
                textDirection: setting.textDirection || state.defaultLayoutDirection,
                position: setting.position || { x: 0, y: 0 },
                textColor: setting.textColor || state.defaultTextColor,
                rotationAngle: setting.rotationAngle || 0
            }));
        } else if (currentImage.bubbleSettings && currentImage.bubbleSettings.length === currentImage.bubbleCoords.length) {
            allBubbleStyles = currentImage.bubbleCoords.map((_, i) => {
                const setting = currentImage.bubbleSettings[i] || {};
                return {
                    fontSize: setting.fontSize || state.defaultFontSize,
                    autoFontSize: setting.autoFontSize || false,
                    fontFamily: setting.fontFamily || state.defaultFontFamily,
                    textDirection: setting.textDirection || state.defaultLayoutDirection,
                    position: setting.position || { x: 0, y: 0 },
                    textColor: setting.textColor || state.defaultTextColor,
                    rotationAngle: setting.rotationAngle || 0
                };
            });
        } else {
            const globalFontSize = $('#autoFontSize').is(':checked') ? 'auto' : parseInt($('#fontSize').val());
            const globalAutoFontSize = $('#autoFontSize').is(':checked');
            const globalFontFamily = $('#fontFamily').val();
            const globalTextDirection = $('#layoutDirection').val();
            const globalTextColor = $('#textColor').val();
            const globalRotationAngle = parseFloat($('#rotationAngle').val() || '0');

            allBubbleStyles = currentImage.bubbleCoords.map(() => ({
                fontSize: globalFontSize,
                autoFontSize: globalAutoFontSize,
                fontFamily: globalFontFamily,
                textDirection: globalTextDirection,
                position: { x: 0, y: 0 },
                textColor: globalTextColor,
                rotationAngle: globalRotationAngle
            }));
        }


        const data = {
            clean_image: preFilledBackgroundBase64, // **发送预填充好的背景**
            bubble_texts: currentTexts,
            bubble_coords: currentImage.bubbleCoords,
            fontSize: $('#fontSize').val(), // 全局字号作为参考
            autoFontSize: $('#autoFontSize').is(':checked'),
            fontFamily: $('#fontFamily').val(), // 全局字体作为参考
            textDirection: $('#layoutDirection').val(), // 全局方向作为参考
            textColor: $('#textColor').val(), // 全局文本颜色作为参考
            rotationAngle: parseFloat($('#rotationAngle').val() || '0'), // 全局旋转作为参考
            all_bubble_styles: allBubbleStyles, // 每个气泡的文本样式
            use_individual_styles: true,

            // 因为背景已在前端处理，后端不需要修复
            use_inpainting: false, // 强制后端不修复
            use_lama: false,       // 强制后端不修复
            blend_edges: false,
            inpainting_strength: 0,
            fill_color: null,     // 后端不需要全局填充色了

            is_font_style_change: true,
            prev_auto_font_size: fromAutoToManual
        };

        api.reRenderImageApi(data)
            .then(response => {
                ui.clearGeneralMessageById(loadingMessageId);
                if (response.rendered_image) {
                    state.updateCurrentImageProperty('translatedDataURL', 'data:image/png;base64,' + response.rendered_image);
                    // **重要**：如果前端成功预填充了背景，那么这个预填充的背景应该成为新的 cleanImageData
                    // 这样，如果用户接下来修改其他文本样式（不改变填充色），可以基于这个最新的背景重绘
                    if (preFilledBackgroundBase64) {
                        state.updateCurrentImageProperty('cleanImageData', preFilledBackgroundBase64);
                        console.log("reRenderFullImage: 更新 cleanImageData 为前端预填充的背景。");
                    }
                    // 如果之前是 _tempCleanImageForFill，它已经被用掉了，不再需要。

                    state.updateCurrentImageProperty('bubbleTexts', currentTexts);
                    ui.updateTranslatedImage(state.getCurrentImage().translatedDataURL);
                    $('#translatedImageDisplay').one('load', () => {
                        ui.updateBubbleHighlight(state.selectedBubbleIndex);
                        resolve();
                    });
                } else {
                    throw new Error("渲染 API 未返回图像数据");
                }
            })
            .catch(error => {
                ui.clearGeneralMessageById(loadingMessageId);
                ui.showGeneralMessage(`重新渲染失败: ${error.message}`, "error");
                ui.updateBubbleHighlight(state.selectedBubbleIndex);
                reject(error);
            });
    });
=======
 * @param {boolean} [isGlobalChange=false] - 是否是全局设置变更触发的重渲染
 * @param {boolean} [fromAutoToManual=false] - 是否是从自动字号切换到手动字号
 */
export function reRenderFullImage(isGlobalChange = false, fromAutoToManual = false) {
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.bubbleCoords || !currentImage.translatedDataURL) {
        console.error("无法重新渲染：缺少必要数据");
        ui.showGeneralMessage("无法重新渲染，缺少图像或气泡数据", "error");
        return;
    }

    // 移除高亮
    ui.updateBubbleHighlight(-1);
    
    // 使用通用消息提示而不是全屏加载提示
    ui.showGeneralMessage("重新渲染中...", "info", false, 0);

    // 准备数据
    const imageToUse = currentImage.translatedDataURL.split(',')[1];
    // 优先使用持久化的干净图，其次是临时图
    const cleanImage = currentImage.cleanImageData || currentImage._tempCleanImage || null;

    // 在编辑模式下，优先使用bubbleSettings中的文本，否则使用图片的翻译文本
    let currentTexts = [];
    if (state.editModeActive && state.bubbleSettings && state.bubbleSettings.length > 0) {
        // 在编辑模式下，使用当前编辑的文本
        currentTexts = state.bubbleSettings.map(setting => setting.text || "");
    } else {
        // 非编辑模式下，使用已翻译的文本
        currentTexts = currentImage.bubbleTexts || [];
    }
    
    // 确保文本数组长度和气泡坐标匹配
    if (!currentTexts || currentTexts.length !== currentImage.bubbleCoords.length) {
        console.error("重新渲染错误：翻译文本数据缺失或与坐标数量不匹配！", {
            textsLength: currentTexts ? currentTexts.length : 'undefined',
            coordsLength: currentImage.bubbleCoords ? currentImage.bubbleCoords.length : 'undefined'
        });
        ui.showGeneralMessage("内部错误：文本与坐标数据不匹配，无法重新渲染", "error");
        ui.hideLoading(); // 隐藏加载状态
        return; // 阻止发送无效请求
    }

    const allBubbleStyles = state.bubbleSettings.map((setting, i) => {
        // 如果是全局变更，应用全局设置；否则使用气泡自己的设置
        const fontSize = isGlobalChange ? ($('#autoFontSize').is(':checked') ? 'auto' : parseInt($('#fontSize').val())) : setting.fontSize;
        const autoFontSize = isGlobalChange ? $('#autoFontSize').is(':checked') : setting.autoFontSize;
        const fontFamily = isGlobalChange ? $('#fontFamily').val() : setting.fontFamily;
        const textDirection = isGlobalChange ? $('#layoutDirection').val() : setting.textDirection;
        const textColor = isGlobalChange ? $('#textColor').val() : setting.textColor;
        const position = setting.position || { x: 0, y: 0 };
        const rotationAngle = setting.rotationAngle || 0;

        return {
            fontSize: fontSize,
            autoFontSize: autoFontSize,
            fontFamily: fontFamily, // 传递相对路径或名称
            textDirection: textDirection,
            position: position, // 后端需要 position_offset
            textColor: textColor,
            rotationAngle: rotationAngle
        };
    });

    // 准备 API 请求数据
    const data = {
        image: imageToUse,
        clean_image: cleanImage,
        bubble_texts: currentTexts,
        bubble_coords: currentImage.bubbleCoords,
        // 全局设置（兼容后端 API）
        fontSize: $('#fontSize').val(),
        autoFontSize: $('#autoFontSize').is(':checked'),
        fontFamily: $('#fontFamily').val(),
        textDirection: $('#layoutDirection').val(),
        // 增加全局文本颜色和旋转角度参数
        textColor: $('#textColor').val(),
        rotationAngle: parseFloat($('#rotationAngle').val() || '0'),
        // 传递所有气泡的详细样式
        all_bubble_styles: allBubbleStyles,
        // 强制使用单个气泡样式,而不是全局样式
        use_individual_styles: true,
        // 传递修复相关的全局设置
        use_inpainting: $('#useInpainting').val() === 'true',
        use_lama: $('#useInpainting').val() === 'lama',
        blend_edges: $('#blendEdges').prop('checked'),
        inpainting_strength: parseFloat($('#inpaintingStrength').val()),
        fill_color: $('#fillColor').val(),
        is_font_style_change: true,
        // 添加是否从自动字号切换到手动字号的标记
        prev_auto_font_size: fromAutoToManual
    };
    
    console.log('发送重新渲染请求，包含单独气泡样式数据:', allBubbleStyles.length, '个气泡', 
                fromAutoToManual ? '(从自动字号切换到手动字号)' : '');

    api.reRenderImageApi(data)
        .then(response => {
            // 移除重新渲染中的提示消息
            $(".message.info").fadeOut(300, function() { $(this).remove(); });
            
            if (response.rendered_image) {
                // 更新状态
                state.updateCurrentImageProperty('translatedDataURL', 'data:image/png;base64,' + response.rendered_image);
                // 更新 UI
                ui.updateTranslatedImage(state.getCurrentImage().translatedDataURL);
                // 保存更新后的设置到图像对象
                state.updateCurrentImageProperty('bubbleSettings', JSON.parse(JSON.stringify(state.bubbleSettings)));
                state.updateCurrentImageProperty('bubbleTexts', currentTexts); // 更新文本状态
                
                // 同时更新图片全局设置属性
                if (isGlobalChange || state.editModeActive) {
                    state.updateCurrentImageProperty('fontSize', $('#autoFontSize').is(':checked') ? 'auto' : parseInt($('#fontSize').val()));
                    state.updateCurrentImageProperty('autoFontSize', $('#autoFontSize').is(':checked'));
                    state.updateCurrentImageProperty('fontFamily', $('#fontFamily').val());
                    state.updateCurrentImageProperty('layoutDirection', $('#layoutDirection').val());
                }

                // 图片加载完成后重新应用高亮
                $('#translatedImageDisplay').one('load', () => {
                    ui.updateBubbleHighlight(state.selectedBubbleIndex);
                });
            } else {
                throw new Error("渲染 API 未返回图像数据");
            }
        })
        .catch(error => {
            // 移除重新渲染中的提示消息
            $(".message.info").fadeOut(300, function() { $(this).remove(); });
            
            ui.showGeneralMessage(`重新渲染失败: ${error.message}`, "error");
            ui.updateBubbleHighlight(state.selectedBubbleIndex); // 失败也要恢复高亮
        });
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
}


/**
 * 将当前选中气泡的样式应用到所有气泡
 */
export function applySettingsToAllBubbles() {
    const index = state.selectedBubbleIndex;
    if (index < 0) return;

    const currentSetting = state.bubbleSettings[index];
    const newSettings = state.bubbleSettings.map(setting => ({
        ...setting, // 保留 text 和 position
        fontSize: currentSetting.fontSize,
        autoFontSize: currentSetting.autoFontSize,
        fontFamily: currentSetting.fontFamily,
        textDirection: currentSetting.textDirection,
        textColor: currentSetting.textColor,
        rotationAngle: currentSetting.rotationAngle
    }));
    state.setBubbleSettings(newSettings); // 更新状态
    ui.updateBubbleListUI(); // 更新列表
    reRenderFullImage(); // 重新渲染整个图像
    ui.showGeneralMessage("样式已应用到所有气泡", "success", false, 2000);
}

/**
 * 重置当前选中气泡的设置为初始状态
 */
export function resetCurrentBubble() {
    const index = state.selectedBubbleIndex;
    if (index < 0 || !state.initialBubbleSettings || index >= state.initialBubbleSettings.length) {
        ui.showGeneralMessage("无法重置：未找到初始设置", "warning");
        return;
    }
    const initialSetting = state.initialBubbleSettings[index];
    // 深拷贝恢复状态
    state.updateSingleBubbleSetting(index, JSON.parse(JSON.stringify(initialSetting)));
    ui.updateBubbleEditArea(index); // 更新编辑区显示
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

    const currentSetting = state.bubbleSettings[index];
    const position = { ...(currentSetting.position || { x: 0, y: 0 }) }; // 创建副本或默认值
    const step = 2;
    const limit = 50; // 位置偏移限制

    switch (direction) {
        case 'moveUp':    position.y = Math.max(position.y - step, -limit); break;
        case 'moveDown':  position.y = Math.min(position.y + step, limit); break;
        case 'moveLeft':  position.x = Math.max(position.x - step, -limit); break;
        case 'moveRight': position.x = Math.min(position.x + step, limit); break;
    }

    state.updateSingleBubbleSetting(index, { position: position });
    ui.updateBubbleEditArea(index); // 更新数值显示
    triggerDelayedPreview(index); // 延迟渲染
}

/**
 * 重置当前选中气泡的位置
 */
export function resetPosition() {
    const index = state.selectedBubbleIndex;
    if (index < 0) return;
    state.updateSingleBubbleSetting(index, { position: { x: 0, y: 0 } });
    ui.updateBubbleEditArea(index);
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
                        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
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

// 用于位置和旋转调整的延迟渲染计时器
let previewTimer = null;
/**
 * 触发带延迟的预览渲染
 * @param {number} bubbleIndex - 气泡索引
 */
function triggerDelayedPreview(bubbleIndex) {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(() => {
        console.log(`准备渲染气泡 ${bubbleIndex} 的预览，当前设置:`, 
            JSON.stringify(state.bubbleSettings[bubbleIndex]));
        renderBubblePreview(bubbleIndex);
    }, 300); // 300ms 延迟
}

/**
<<<<<<< HEAD
 * 退出编辑模式
 */
export function exitEditMode() {
    if (state.editModeActive) {
        toggleEditMode(); // 这会处理所有清理工作，然后设置 editModeActive = false
=======
 * 退出编辑模式（如果当前处于编辑模式）
 */
export function exitEditMode() {
    if (state.editModeActive) {
        toggleEditMode();
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
    }
}
