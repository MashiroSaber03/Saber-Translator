import * as state from './state.js';
import * as ui from './ui.js';
import * as api from './api.js';
import * as constants from './constants.js'; // <--- 添加导入
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
        ui.toggleEditModeUI(false); // 更新 UI
        // 保存最终的气泡设置到当前图片对象
        const currentImage = state.getCurrentImage();
        if (currentImage && state.bubbleSettings.length > 0) { // 仅在有设置时保存
            // 保存气泡设置到图像对象
            currentImage.bubbleSettings = JSON.parse(JSON.stringify(state.bubbleSettings));
            // 同时更新 bubbleTexts
            currentImage.bubbleTexts = state.bubbleSettings.map(s => s.text);
            
            // 同步更新当前图片的全局属性，确保一致性
            const firstBubbleSetting = state.bubbleSettings[0];
            currentImage.fontSize = firstBubbleSetting.fontSize;
            currentImage.autoFontSize = firstBubbleSetting.autoFontSize;
            currentImage.fontFamily = firstBubbleSetting.fontFamily;
            currentImage.layoutDirection = firstBubbleSetting.textDirection;
            
            // 重新渲染以确保更改立即可见
            reRenderFullImage();
            
            console.log("退出编辑模式，已保存气泡设置到图像对象");
        }
        state.setSelectedBubbleIndex(-1); // 重置选中索引
        state.setInitialBubbleSettings([]); // 清空备份
    }
}

/**
 * 初始化或加载当前图片的气泡设置
 */
function initBubbleSettings() {
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.bubbleCoords) {
        console.error("无法初始化气泡设置：无效的图像或坐标");
        state.setBubbleSettings([]);
        return;
    }

    // 检查当前图像是否已有保存的气泡设置
    if (currentImage.bubbleSettings && currentImage.bubbleSettings.length === currentImage.bubbleCoords.length) {
        console.log("加载当前图像已保存的气泡设置");
        
        // 更新：确保使用最新的全局设置更新现有的气泡设置
        // 从当前UI控件读取全局设置
        const globalFontSize = $('#autoFontSize').is(':checked') ? 'auto' : parseInt($('#fontSize').val());
        const globalAutoFontSize = $('#autoFontSize').is(':checked');
        const globalFontFamily = $('#fontFamily').val();
        const globalTextDirection = $('#layoutDirection').val();
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
                textColor: globalTextColor,
                rotationAngle: 0
            });
        }
        state.setBubbleSettings(newSettings);
        // 同时保存到当前图像对象
        currentImage.bubbleSettings = JSON.parse(JSON.stringify(newSettings));
    }
    ui.updateBubbleListUI(); // 更新列表显示
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
    ui.updateBubbleHighlight(index);
    // 滚动气泡列表到选中项
    const bubbleItem = $(`.bubble-item[data-index="${index}"]`);
    if (bubbleItem.length) {
        // 使用原生 scrollIntoView
        bubbleItem[0].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
            reRenderFullImage(false, true); // 第二个参数表示是从自动切换到手动
            return;
        }
    }
    
    // 正常情况
    reRenderFullImage();
}

/**
 * 重新渲染整个图像
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
 * 退出编辑模式（如果当前处于编辑模式）
 */
export function exitEditMode() {
    if (state.editModeActive) {
        toggleEditMode();
    }
}
