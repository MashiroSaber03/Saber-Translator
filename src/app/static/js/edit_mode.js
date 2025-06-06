import * as state from './state.js';
import * as ui from './ui.js';
import * as api from './api.js';
import * as constants from './constants.js'; // <--- 添加导入
import * as session from './session.js'; // <--- 添加导入，用于自动存档
import * as main from './main.js'; // 导入main模块以使用loadImage函数
// import $ from 'jquery'; // 假设 jQuery 已全局加载

/**
 * 切换编辑模式
 */
export function toggleEditMode() {
    state.setEditModeActive(!state.editModeActive); // 更新状态

    if (state.editModeActive) {
        // --- 进入编辑模式 ---
        const currentImage = state.getCurrentImage();
        
        // 删除以下检查，允许没有气泡的图片也能进入编辑模式
        // if (!currentImage || !currentImage.bubbleCoords || currentImage.bubbleCoords.length === 0) { // 检查是否有气泡
        //     ui.showGeneralMessage("当前图片没有可编辑的气泡", "warning");
        //     state.setEditModeActive(false); // 切换失败，恢复状态
        //     return;
        // }

        ui.toggleEditModeUI(true); // 更新 UI
        initBubbleSettings(); // 初始化或加载气泡设置
        // 保存初始设置备份
        state.setInitialBubbleSettings(JSON.parse(JSON.stringify(state.bubbleSettings)));
        console.log("已保存初始气泡设置:", state.initialBubbleSettings);

        // 默认选择第一个气泡，如果有的话
        if (state.bubbleSettings && state.bubbleSettings.length > 0) {
            selectBubble(0);
        } else {
            // 没有气泡，清空编辑区
            ui.updateBubbleEditArea(-1);
        }
        // 确保有干净背景
        ensureCleanBackground();

    } else {
        // --- 退出编辑模式 ---
        ui.toggleEditModeUI(false); // 更新 UI (这会清除所有高亮框)
        
        // 确保清除任何可能存在的"重新渲染中..."消息
        ui.clearGeneralMessageById("rendering_loading_message");
        
        // 保存最终的气泡设置到当前图片对象
        const currentImage = state.getCurrentImage();
        if (currentImage && state.bubbleSettings.length > 0) { // 仅在有设置时保存
            // 保存气泡设置到图像对象，确保完整复制所有属性，包括fillColor
            currentImage.bubbleSettings = JSON.parse(JSON.stringify(state.bubbleSettings));
            // 同时更新 bubbleTexts
            currentImage.bubbleTexts = state.bubbleSettings.map(s => s.text);
            
            // 同步更新当前图片的全局属性，确保一致性
            const firstBubbleSetting = state.bubbleSettings[0];
            currentImage.fontSize = firstBubbleSetting.fontSize;
            currentImage.autoFontSize = firstBubbleSetting.autoFontSize;
            currentImage.fontFamily = firstBubbleSetting.fontFamily;
            currentImage.layoutDirection = firstBubbleSetting.textDirection;
            
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
    }
}

/**
 * 初始化或加载当前图片的气泡设置
 */
export function initBubbleSettings() {
    const currentImage = state.getCurrentImage();
    if (!currentImage) {
        console.error("无法初始化气泡设置：无效的图像");
        state.setBubbleSettings([]);
        return;
    }

    // 处理没有气泡的情况
    if (!currentImage.bubbleCoords || currentImage.bubbleCoords.length === 0) {
        console.log("当前图片没有气泡，初始化空的气泡设置");
        state.setBubbleSettings([]);
        return;
    }

    const imageGlobalFillColor = currentImage.fillColor || state.defaultFillColor; // 当前图片的全局填充色

    if (currentImage.bubbleSettings && currentImage.bubbleSettings.length === currentImage.bubbleCoords.length) {
        console.log("加载当前图像已保存的气泡设置 (含填充色检查)");
        const loadedSettings = JSON.parse(JSON.stringify(currentImage.bubbleSettings));
        // 确保每个加载的设置都有 fillColor
        loadedSettings.forEach((setting, index) => {
            if (!setting.hasOwnProperty('fillColor') || setting.fillColor === null || setting.fillColor === undefined) {
                setting.fillColor = imageGlobalFillColor; // 如果缺失，用图片的全局填充色
            }
            // 确保每个气泡都有正确的文本内容（从bubbleTexts数组获取）
            if ((!setting.text || setting.text === "") && currentImage.bubbleTexts && currentImage.bubbleTexts[index]) {
                setting.text = currentImage.bubbleTexts[index];
            }
        });
        state.setBubbleSettings(loadedSettings);
    } else {
        console.log("创建新的默认气泡设置 (含填充色)");
        const newSettings = [];
        const globalFontSize = $('#autoFontSize').is(':checked') ? 'auto' : parseInt($('#fontSize').val());
        const globalAutoFontSize = $('#autoFontSize').is(':checked');
        const globalFontFamily = $('#fontFamily').val();
        const globalTextDirection = $('#layoutDirection').val();
        const globalTextColor = $('#textColor').val(); // 全局文本颜色
        // globalFillColor 已经通过 imageGlobalFillColor 获取

        if (!currentImage.bubbleTexts || currentImage.bubbleTexts.length !== currentImage.bubbleCoords.length) {
            currentImage.bubbleTexts = Array(currentImage.bubbleCoords.length).fill("");
        } else if (currentImage.bubbleTexts.length < currentImage.bubbleCoords.length) {
            // 如果bubbleTexts长度小于bubbleCoords，则扩展数组
            while (currentImage.bubbleTexts.length < currentImage.bubbleCoords.length) {
                currentImage.bubbleTexts.push("");
            }
        }

        for (let i = 0; i < currentImage.bubbleCoords.length; i++) {
            newSettings.push({
                // 优先使用bubbleTexts中的文本内容
                text: currentImage.bubbleTexts[i] || "",
                fontSize: globalFontSize,
                autoFontSize: globalAutoFontSize,
                fontFamily: globalFontFamily,
                textDirection: globalTextDirection,
                position: { x: 0, y: 0 },
                textColor: globalTextColor, // 使用全局文本颜色
                rotationAngle: 0,
                fillColor: imageGlobalFillColor, // <--- 新增：使用图片的全局填充色初始化
                // === 新增：气泡描边设置 START ===
                enableStroke: state.enableTextStroke, // 初始使用全局描边设置
                strokeColor: state.textStrokeColor,   // 初始使用全局描边颜色
                strokeWidth: state.textStrokeWidth    // 初始使用全局描边宽度
                // === 新增：气泡描边设置 END ===
            });
        }
        state.setBubbleSettings(newSettings);
        currentImage.bubbleSettings = JSON.parse(JSON.stringify(newSettings));
    }
    ui.updateBubbleListUI();
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
    ui.updateBubbleHighlight(index); // 这会更新所有气泡的高亮框并标记选中的
    
    // 滚动气泡列表到选中项，但不滚动页面
    const bubbleItem = $(`.bubble-item[data-index="${index}"]`);
    if (bubbleItem.length) {
        // 仅滚动气泡列表容器，不影响页面滚动
        const bubbleList = $('.bubble-list');
        bubbleList.animate({
            scrollTop: bubbleItem.position().top + bubbleList.scrollTop() - bubbleList.position().top
        }, 300);
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
            reRenderFullImage(true); // 参数表示是从自动切换到手动
            return;
        }
    }
    
    // 正常情况
    reRenderFullImage();
}

/**
 * 重新渲染整个图像
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
        ui.showGeneralMessage("重新渲染中，请不要在重渲染时快速切换图片", "info", false, 0, loadingMessageId);

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
            const usesLamaInpainting = (
                (currentImage.hasOwnProperty('_lama_inpainted') && currentImage._lama_inpainted === true) || 
                (currentImage.originalUseLama === true)
            );
            
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
                            ctx.fillRect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
                            appliedIndividualFills = true;
                        } else {
                            // 如果某个气泡没有独立填充色，则使用图片的全局填充色
                            const imageFillColor = currentImage.fillColor || state.defaultFillColor;
                            const [x1, y1, x2, y2] = coords;
                            ctx.fillStyle = imageFillColor;
                            ctx.fillRect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
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
                            ctx.fillRect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
                            appliedIndividualFills = true;
                        } else {
                            // 如果某个气泡没有独立填充色，则使用图片的全局填充色
                            const imageFillColor = currentImage.fillColor || state.defaultFillColor;
                            const [x1, y1, x2, y2] = coords;
                            ctx.fillStyle = imageFillColor;
                            ctx.fillRect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
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
                        ctx.fillRect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
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
            console.error("重新渲染错误：文本与坐标不匹配！", 
                          `文本数量: ${currentTexts ? currentTexts.length : 0}, 坐标数量: ${currentImage.bubbleCoords ? currentImage.bubbleCoords.length : 0}`);
            
            // 修复：如果文本数量不匹配，则调整文本数组长度以匹配坐标数量
            if (currentImage.bubbleCoords && currentImage.bubbleCoords.length > 0) {
                if (!currentTexts) {
                    currentTexts = [];
                }
                
                // 扩展或截断文本数组以匹配坐标数组长度
                if (currentTexts.length < currentImage.bubbleCoords.length) {
                    // 如果文本数量不足，用空字符串填充
                    while (currentTexts.length < currentImage.bubbleCoords.length) {
                        currentTexts.push("");
                    }
                    console.log("已自动填充缺失的文本条目");
                } else if (currentTexts.length > currentImage.bubbleCoords.length) {
                    // 如果文本数量过多，截断多余部分
                    currentTexts = currentTexts.slice(0, currentImage.bubbleCoords.length);
                    console.log("已截断多余的文本条目");
                }
            } else {
                ui.clearGeneralMessageById(loadingMessageId);
                ui.showGeneralMessage("错误：没有有效的气泡坐标", "error");
                reject(new Error("没有有效的气泡坐标"));
                return;
            }
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
                rotationAngle: setting.rotationAngle || 0,
                // === 新增：添加描边参数 START ===
                enableStroke: setting.enableStroke !== undefined ? setting.enableStroke : state.enableTextStroke,
                strokeColor: setting.strokeColor || state.textStrokeColor,
                strokeWidth: setting.strokeWidth !== undefined ? setting.strokeWidth : state.textStrokeWidth
                // === 新增：添加描边参数 END ===
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
                    rotationAngle: setting.rotationAngle || 0,
                    // === 新增：添加描边参数 START ===
                    enableStroke: setting.enableStroke !== undefined ? setting.enableStroke : state.enableTextStroke,
                    strokeColor: setting.strokeColor || state.textStrokeColor,
                    strokeWidth: setting.strokeWidth !== undefined ? setting.strokeWidth : state.textStrokeWidth
                    // === 新增：添加描边参数 END ===
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
                rotationAngle: globalRotationAngle,
                // === 新增：添加描边参数 START ===
                enableStroke: state.enableTextStroke,
                strokeColor: state.textStrokeColor,
                strokeWidth: state.textStrokeWidth
                // === 新增：添加描边参数 END ===
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
            prev_auto_font_size: fromAutoToManual,

            // === 新增：传递描边参数 START ===
            enableTextStroke: state.enableTextStroke,
            textStrokeColor: state.textStrokeColor,
            textStrokeWidth: state.textStrokeWidth
            // === 新增：传递描边参数 END ===
        };

        // 添加日志显示描边参数
        console.log(`reRenderFullImage: 发送描边参数 - 启用=${data.enableTextStroke}, 颜色=${data.textStrokeColor}, 宽度=${data.textStrokeWidth}`);

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
        rotationAngle: currentSetting.rotationAngle,
        // === 新增：复制描边设置 START ===
        enableStroke: currentSetting.enableStroke,
        strokeColor: currentSetting.strokeColor,
        strokeWidth: currentSetting.strokeWidth
        // === 新增：复制描边设置 END ===
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
 * 退出编辑模式
 */
export function exitEditMode() {
    if (state.editModeActive) {
        toggleEditMode(); // 这会处理所有清理工作，然后设置 editModeActive = false
    }
}
