import * as state from './state.js';
import * as ui from './ui.js'; // 稍后会用到
import * as api from './api.js'; // <--- 新增导入
import * as main from './main.js'; // <--- 新增导入 main.js
import * as session from './session.js'; // <--- 新增导入，用于自动存档

// --- 新增：绘制状态变量 ---
let isDrawing = false;      // 是否正在绘制
let startX, startY;       // 绘制起始点 (相对于页面)
let currentRect = null;   // 当前正在绘制的矩形 DOM 元素
let drawingOffsetX, drawingOffsetY; // 图片容器相对于页面的偏移
// ------------------------

// --- 新增：拖动状态变量 ---
let isDragging = false;     // 是否正在拖动框
let dragTarget = null;    // 当前拖动的框 (jQuery 对象)
let dragIndex = -1;       // 当前拖动的框的索引
let dragStartX, dragStartY; // 拖动起始时鼠标相对于页面的位置
let boxInitialX, boxInitialY; // 拖动起始时框的左上角位置 (相对于容器)
// ------------------------

// --- 新增：调整大小状态 ---
let isResizing = false;      // 是否正在调整大小
let resizeHandleType = null; // 调整手柄类型 (e.g., 'top-left')
let resizeTargetIndex = -1; // 正在调整大小的框的索引
let resizeStartX, resizeStartY; // 调整大小起始鼠标位置 (相对于页面)
let initialBoxCoords = null;  // 调整大小起始时的框的原始坐标 [x1,y1,x2,y2]
let resizeImageInfo = null;   // 调整大小起始时的图片信息 (偏移、缩放比例、自然尺寸)
// ---------------------------

/**
 * 进入标注模式
 */
export function enterLabelingMode() {
    if (state.isLabelingModeActive) return;

    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.originalDataURL) {
        ui.showGeneralMessage("无法进入标注模式：没有有效图片。", "warning");
        return;
    }

    console.log("进入标注模式...");
    state.setLabelingModeActive(true); // 先设置模式状态

    // --- 加载坐标逻辑 ---
    let initialCoords = [];
    // 优先加载已保存的手动坐标
    if (currentImage.savedManualCoords && currentImage.savedManualCoords.length > 0) {
        initialCoords = JSON.parse(JSON.stringify(currentImage.savedManualCoords));
        console.log(`加载了图片 ${state.currentImageIndex} 已保存的 ${initialCoords.length} 个手动标注框。`);
        state.setManualCoords(initialCoords, true); // 标记为从加载，不触发 unsavedChanges
        state.setHasUnsavedChanges(false); // 重置当前模式的未保存状态
        currentImage.hasUnsavedChanges = false; // 重置图片对象的状态
    }
    // 如果没有保存的手动框，再尝试加载自动检测的框作为起点
    else if (currentImage.bubbleCoords && currentImage.bubbleCoords.length > 0) {
        initialCoords = JSON.parse(JSON.stringify(currentImage.bubbleCoords));
        console.log(`加载了图片 ${state.currentImageIndex} 自动检测的 ${initialCoords.length} 个框作为初始标注。`);
        state.setManualCoords(initialCoords, false); // *非*加载，标记为需要保存
        state.setHasUnsavedChanges(true); // 标记为需要保存
        currentImage.hasUnsavedChanges = true;
    } else {
        // 如果两者都没有，则为空
        state.setManualCoords([], true); // 清空，标记为非更改
        state.setHasUnsavedChanges(false);
        currentImage.hasUnsavedChanges = false;
    }
    // --------------------

    state.setSelectedManualBoxIndex(-1); // 清除选中
    ui.drawBoundingBoxes(state.manualBubbleCoords); // 绘制加载的或空的框
    ui.toggleLabelingModeUI(true); // 更新 UI

    console.log("进入标注模式完成。");
}

/**
 * 加载当前图片的气泡坐标用于标注
 * 在切换图片但保持标注模式时被调用
 */
export function loadBubbleCoordsForLabeling() {
    if (!state.isLabelingModeActive) return;
    
    const currentImage = state.getCurrentImage();
    if (!currentImage) return;

    console.log("切换图片：加载新图片的标注框...");
    
    // 清除选中状态
    state.setSelectedManualBoxIndex(-1);
    
    // --- 加载坐标逻辑（与enterLabelingMode中相同）---
    let initialCoords = [];
    // 优先加载已保存的手动坐标
    if (currentImage.savedManualCoords && currentImage.savedManualCoords.length > 0) {
        initialCoords = JSON.parse(JSON.stringify(currentImage.savedManualCoords));
        console.log(`切换图片：加载了图片 ${state.currentImageIndex} 已保存的 ${initialCoords.length} 个手动标注框。`);
        state.setManualCoords(initialCoords, true); // 标记为从加载，不触发 unsavedChanges
        state.setHasUnsavedChanges(false); // 重置当前模式的未保存状态
        currentImage.hasUnsavedChanges = false; // 重置图片对象的状态
    }
    // 如果没有保存的手动框，再尝试加载自动检测的框作为起点
    else if (currentImage.bubbleCoords && currentImage.bubbleCoords.length > 0) {
        initialCoords = JSON.parse(JSON.stringify(currentImage.bubbleCoords));
        console.log(`切换图片：加载了图片 ${state.currentImageIndex} 自动检测的 ${initialCoords.length} 个框作为初始标注。`);
        state.setManualCoords(initialCoords, false); // *非*加载，标记为需要保存
        state.setHasUnsavedChanges(true); // 标记为需要保存
        currentImage.hasUnsavedChanges = true;
    } else {
        // 如果两者都没有，则为空
        state.setManualCoords([], true); // 清空，标记为非更改
        state.setHasUnsavedChanges(false);
        currentImage.hasUnsavedChanges = false;
    }
    // --------------------
}

/**
 * 退出标注模式
 * @param {boolean} [promptSave=true] - 参数保留但不再使用，总是自动保存
 */
export function exitLabelingMode(promptSave = true) {
    if (!state.isLabelingModeActive) return;

    console.log("尝试退出标注模式...");

    // --- 自动保存未保存的更改 ---
    let shouldExit = true; // 默认允许退出
    if (state.hasUnsavedChanges) {
        // 自动保存，不再提示
        if (!state.saveManualCoordsToImage()) { // 调用保存函数
            // 如果保存失败（理论上不应发生，除非状态混乱）
            ui.showGeneralMessage("保存标注失败！", "error");
            // 即使保存失败也继续退出
        } else {
            ui.showGeneralMessage("标注已保存。", "success", false, 2000);
            ui.renderThumbnails(); // 保存成功后更新缩略图
        }
    }
    // ----------------------------

    if (shouldExit) {
        console.log("确认退出标注模式...");
        // 清理拖动/缩放/绘制状态
        if (isDrawing && currentRect) { currentRect.remove(); }
        isDrawing = false; currentRect = null;
        if (isDragging && dragTarget) { dragTarget.css('cursor', 'grab'); }
        isDragging = false; dragTarget = null; dragIndex = -1;
        if (isResizing) { $('body').css('cursor', 'default'); }
        isResizing = false; resizeHandleType = null; resizeTargetIndex = -1;
        initialBoxCoords = null; resizeImageInfo = null;

        // 重置状态和 UI
        state.setLabelingModeActive(false); // 这会隐式调用 setHasUnsavedChanges(false)
        state.setManualCoords([], true); // 清空当前工作区的坐标
        state.setSelectedManualBoxIndex(-1);
        ui.clearBoundingBoxes();
        ui.toggleLabelingModeUI(false);
        console.log("已退出标注模式。");
        
        session.triggerAutoSave(); // <--- 退出标注模式后触发自动存档
    } else {
         console.log("退出标注模式失败。");
    }
}

/**
 * 处理"自动检测文本框"按钮点击
 */
export function handleAutoDetectClick() {
    if (!state.isLabelingModeActive) return;

    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.originalDataURL) {
        ui.showGeneralMessage("没有有效的图片用于检测。", "warning");
        return;
    }

    ui.showLoading("正在自动检测文本框..."); // 使用通用消息

    const imageData = currentImage.originalDataURL.split(',')[1];
    // 可选：从 UI 获取置信度阈值，如果需要的话
    // const confThreshold = parseFloat($('#yoloConfidenceSlider').val() || 0.6);

    api.detectBoxesApi(imageData /*, confThreshold*/)
        .then(response => {
            ui.hideLoading(); // 隐藏加载消息
            if (response.success && response.bubble_coords) {
                state.setManualCoords(response.bubble_coords);
                state.setSelectedManualBoxIndex(-1); // 清除选中状态
                ui.drawBoundingBoxes(state.manualBubbleCoords); // 绘制检测到的框
                ui.showGeneralMessage(`自动检测到 ${response.bubble_coords.length} 个文本框。`, "success");
                // 更新按钮状态（例如启用"清除"和"使用"按钮）
                ui.toggleLabelingModeUI(true); // 重新调用以更新按钮禁用状态
            } else {
                throw new Error(response.error || "检测失败，未返回坐标。");
            }
        })
        .catch(error => {
            ui.hideLoading(); // 隐藏加载消息
            state.setManualCoords([]); // 清空坐标
            ui.clearBoundingBoxes(); // 清除可能存在的旧框
            ui.showGeneralMessage(`自动检测失败: ${error.message}`, "error");
            ui.toggleLabelingModeUI(true); // 更新按钮状态
        });
}

/**
 * 处理"检测所有图片"按钮点击
 * 会对所有上传的图片进行文本框检测，并保存检测结果
 */
export function handleDetectAllImagesClick() {
    if (!state.isLabelingModeActive) return;

    if (state.images.length <= 1) {
        ui.showGeneralMessage("至少需要两张图片才能执行批量检测。", "warning");
        return;
    }

    // 确认对话框，因为这会影响所有图片
    if (!confirm("此操作将对所有图片进行文本框检测，可能会覆盖已有的手动标注。确定继续吗？")) {
        return;
    }

    // 记录当前索引，以便处理完后恢复
    const originalIndex = state.currentImageIndex;
    
    // 显示处理开始消息
    ui.showLoading("正在对所有图片进行文本框检测...");
    ui.updateProgressBar(0, "0/" + state.images.length);
    $("#translationProgressBar").show();
    
    // 定义一个递归函数来处理每张图片
    const processNextImage = (index = 0, totalDetected = 0) => {
        if (index >= state.images.length) {
            // 所有图片处理完成
            ui.hideLoading();
            $("#translationProgressBar").hide();
            
            // 返回到原始图片并刷新显示
            if (originalIndex !== state.currentImageIndex) {
                state.setCurrentImageIndex(originalIndex);
                loadBubbleCoordsForLabeling(); // 重新加载当前图片的坐标
                ui.drawBoundingBoxes(state.manualBubbleCoords);
            }
            
            ui.showGeneralMessage(`批量检测完成！共处理 ${state.images.length} 张图片，检测到 ${totalDetected} 个文本框。`, "success");
            ui.renderThumbnails(); // 更新缩略图显示 (可能会添加标注图标)
            session.triggerAutoSave(); // 触发自动保存
            return;
        }
        
        // 更新进度条
        const progress = Math.floor((index / state.images.length) * 100);
        ui.updateProgressBar(progress, `${index}/${state.images.length}`);
        
        // 获取当前处理的图片
        const image = state.images[index];
        if (!image || !image.originalDataURL) {
            // 跳过无效图片
            processNextImage(index + 1, totalDetected);
            return;
        }
        
        // 获取图片数据
        const imageData = image.originalDataURL.split(',')[1];
        
        // 调用API检测文本框
        api.detectBoxesApi(imageData)
            .then(response => {
                if (response.success && response.bubble_coords) {
                    // 保存检测到的坐标到图片对象
                    image.savedManualCoords = response.bubble_coords;
                    image.hasUnsavedChanges = false; // 标记为已保存
                    
                    // 如果是当前图片，同时更新显示
                    if (index === state.currentImageIndex) {
                        state.setManualCoords(response.bubble_coords, true);
                        ui.drawBoundingBoxes(state.manualBubbleCoords);
                    }
                    
                    // 继续处理下一张图片
                    processNextImage(index + 1, totalDetected + response.bubble_coords.length);
                } else {
                    console.error(`图片 ${index} 检测失败:`, response.error || "未返回坐标");
                    // 跳过错误，继续处理下一张
                    processNextImage(index + 1, totalDetected);
                }
            })
            .catch(error => {
                console.error(`图片 ${index} 检测出错:`, error);
                // 跳过错误，继续处理下一张
                processNextImage(index + 1, totalDetected);
            });
    };
    
    // 开始处理第一张图片
    processNextImage(0);
}

/**
 * 处理"清除所有框"按钮点击
 */
export function handleClearManualBoxesClick() {
    if (!state.isLabelingModeActive) return;

    if (state.manualBubbleCoords.length > 0) {
        // 清除前检查是否有未保存更改
        const hadChangesBeforeClear = state.hasUnsavedChanges;
        if (confirm("确定要清除所有手动标注的框吗？")) {
            state.setManualCoords([]); // 更新状态，这会触发 hasUnsavedChanges=true (如果之前没有)
            state.setSelectedManualBoxIndex(-1);
            ui.clearBoundingBoxes();
            ui.toggleLabelingModeUI(true); // 更新按钮状态
        }
    } else {
        ui.showGeneralMessage("当前没有手动框可清除。", "info");
    }
}

/**
 * 处理在图片容器上按下鼠标按钮的事件 (开始绘制)
 * @param {jQuery.Event} e - 事件对象
 */
export function handleMouseDownOnImage(e) {
    // 仅在标注模式下，并且鼠标左键按下时生效
    if (!state.isLabelingModeActive || e.button !== 0) return;
    // 如果点击发生在已有的框上，则不开始绘制新框（这是选择逻辑）
    if ($(e.target).hasClass('manual-bounding-box')) return;

    isDrawing = true;
    const imageContainer = $(e.currentTarget); // 事件目标是 .image-container
    const offset = imageContainer.offset(); // 获取容器相对于文档的偏移
    drawingOffsetX = offset.left;
    drawingOffsetY = offset.top;

    startX = e.pageX - drawingOffsetX; // 相对于容器的起始 X
    startY = e.pageY - drawingOffsetY; // 相对于容器的起始 Y

    // 创建一个临时的绘制矩形
    currentRect = $('<div class="drawing-rect"></div>').css({
        position: 'absolute',
        left: `${startX}px`,
        top: `${startY}px`,
        width: '0px',
        height: '0px',
        border: '2px dashed #007bff', // 蓝色虚线框
        boxSizing: 'border-box',
        pointerEvents: 'none', // 不干扰其他鼠标事件
        zIndex: 11 // 比标注框高一点
    });
    imageContainer.append(currentRect);

    console.log(`开始绘制: startX=${startX.toFixed(1)}, startY=${startY.toFixed(1)}`);
    e.preventDefault(); // 阻止可能的文本选择等默认行为
}

/**
 * 处理鼠标移动事件 (更新绘制、拖动或调整大小)
 * @param {jQuery.Event} e - 事件对象
 */
export function handleMouseMove(e) {
    if (isDrawing && currentRect) {
        const currentX = e.pageX - drawingOffsetX;
        const currentY = e.pageY - drawingOffsetY;
        const width = Math.abs(currentX - startX);
        const height = Math.abs(currentY - startY);
        const left = Math.min(startX, currentX);
        const top = Math.min(startY, currentY);
        currentRect.css({ left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px` });

    } else if (isDragging && dragTarget) {
        const deltaX = e.pageX - dragStartX;
        const deltaY = e.pageY - dragStartY;
        const newLeft = boxInitialX + deltaX;
        const newTop = boxInitialY + deltaY;
        dragTarget.css({ left: `${newLeft}px`, top: `${newTop}px` });

    } else if (isResizing && resizeTargetIndex !== -1 && resizeImageInfo) { // 确保 resizeImageInfo 有效
        const deltaXScreen = e.pageX - resizeStartX;
        const deltaYScreen = e.pageY - resizeStartY;

        const [x1_orig_native, y1_orig_native, x2_orig_native, y2_orig_native] = initialBoxCoords; // 原始图片原生坐标
        const metrics = resizeImageInfo; // 使用存储的精确指标

        // 将屏幕上的鼠标位移转换为图像原生坐标系下的位移
        const deltaX_native = deltaXScreen / metrics.scaleX;
        const deltaY_native = deltaYScreen / metrics.scaleY;

        let newX1_native = x1_orig_native, newY1_native = y1_orig_native, newX2_native = x2_orig_native, newY2_native = y2_orig_native;

        if (resizeHandleType.includes('left'))   newX1_native = x1_orig_native + deltaX_native;
        if (resizeHandleType.includes('right'))  newX2_native = x2_orig_native + deltaX_native;
        if (resizeHandleType.includes('top'))    newY1_native = y1_orig_native + deltaY_native;
        if (resizeHandleType.includes('bottom')) newY2_native = y2_orig_native + deltaY_native;

        // 保证 x1 < x2, y1 < y2 (在原生坐标系下)
        let finalX1_native = Math.min(newX1_native, newX2_native);
        let finalY1_native = Math.min(newY1_native, newY2_native);
        let finalX2_native = Math.max(newX1_native, newX2_native);
        let finalY2_native = Math.max(newY1_native, newY2_native);

        // --- 更新视觉效果 (转换为屏幕坐标) ---
        const targetBox = $(`.manual-bounding-box[data-index="${resizeTargetIndex}"]`);
        if (targetBox.length) {
            // 从图像原生坐标计算屏幕坐标
            const screenLeft = metrics.visualContentOffsetX + finalX1_native * metrics.scaleX;
            const screenTop = metrics.visualContentOffsetY + finalY1_native * metrics.scaleY;
            const screenWidth = (finalX2_native - finalX1_native) * metrics.scaleX;
            const screenHeight = (finalY2_native - finalY1_native) * metrics.scaleY;

            const minScreenSize = 10; // 屏幕上的最小尺寸
            if (screenWidth >= minScreenSize && screenHeight >= minScreenSize) {
                targetBox.css({
                    left: `${screenLeft}px`,
                    top: `${screenTop}px`,
                    width: `${screenWidth}px`,
                    height: `${screenHeight}px`
                });
            }
        }
    }
}

/**
 * 处理松开鼠标按钮的事件 (结束绘制、拖动或调整大小)
 * @param {jQuery.Event} e - 事件对象
 */
export function handleMouseUp(e) {
    let coordsChanged = false;

    if (isDrawing && currentRect) { // 结束绘制
        const endXPage = e.pageX; // 鼠标相对于页面的结束X
        const endYPage = e.pageY; // 鼠标相对于页面的结束Y

        currentRect.remove();
        currentRect = null;
        isDrawing = false;

        // startX, startY 是在 handleMouseDownOnImage 中计算的，它们是相对于 .image-container 的绘制起始点
        // drawingOffsetX, drawingOffsetY 是 .image-container 相对于文档的偏移
        // 我们需要将绘制的矩形（其坐标是相对于 .image-container 的）转换回图像原生坐标

        const imageContainer = $('.image-container');
        // const containerOffset = imageContainer.offset(); // drawingOffsetX/Y 已经是这个了

        // 绘制的矩形在屏幕上的坐标 (相对于 .image-container)
        const rectScreenLeft = Math.min(startX, endXPage - drawingOffsetX);
        const rectScreenTop = Math.min(startY, endYPage - drawingOffsetY);
        const rectScreenWidth = Math.abs((endXPage - drawingOffsetX) - startX);
        const rectScreenHeight = Math.abs((endYPage - drawingOffsetY) - startY);

        if (rectScreenWidth < 10 || rectScreenHeight < 10) {
            console.log("绘制的框太小，已忽略。");
            return;
        }

        const imageElement = $('#translatedImageDisplay');
        const metrics = ui.calculateImageDisplayMetrics(imageElement); // 使用辅助函数
        if (!metrics) {
            ui.showGeneralMessage("错误：无法获取图片尺寸以转换坐标。", "error");
            return;
        }

        // 将绘制的矩形屏幕坐标（相对于.image-container）转换回图像原生坐标
        // 注意：rectScreenLeft/Top 是相对于 .image-container 的，而 metrics.visualContentOffsetX/Y 也是相对于 .image-container 的
        const x1_rel = (rectScreenLeft - metrics.visualContentOffsetX) / metrics.scaleX;
        const y1_rel = (rectScreenTop - metrics.visualContentOffsetY) / metrics.scaleY;
        const x2_rel = x1_rel + rectScreenWidth / metrics.scaleX;
        const y2_rel = y1_rel + rectScreenHeight / metrics.scaleY;

        const x1 = Math.max(0, Math.round(x1_rel));
        const y1 = Math.max(0, Math.round(y1_rel));
        const x2 = Math.min(metrics.naturalWidth, Math.round(x2_rel));
        const y2 = Math.min(metrics.naturalHeight, Math.round(y2_rel));

        if (x1 >= x2 || y1 >= y2) {
            console.warn("转换后的坐标无效，已忽略。", {x1, y1, x2, y2});
            return;
        }
        const newCoords = [x1, y1, x2, y2];
        const currentCoords = [...state.manualBubbleCoords];
        state.setManualCoords([...currentCoords, newCoords]);
        state.setSelectedManualBoxIndex(state.manualBubbleCoords.length - 1);
        ui.drawBoundingBoxes(state.manualBubbleCoords, state.selectedManualBoxIndex);
        coordsChanged = true;
        console.log("新标注框已添加 (图像原生坐标):", newCoords);

    } else if (isDragging && dragTarget) { // 结束拖动
        // ... (拖动逻辑，也需要更新) ...
        // finalLeft, finalTop 是拖动后框的屏幕坐标 (相对于.image-container)
        const finalLeft = parseFloat(dragTarget.css('left'));
        const finalTop = parseFloat(dragTarget.css('top'));
        const finalWidth = parseFloat(dragTarget.css('width')); // 屏幕宽度
        const finalHeight = parseFloat(dragTarget.css('height')); // 屏幕高度

        const imageElement = $('#translatedImageDisplay');
        const metrics = ui.calculateImageDisplayMetrics(imageElement);
        if (!metrics) {
            ui.showGeneralMessage("错误：无法获取图片尺寸以更新坐标。", "error");
            ui.drawBoundingBoxes(state.manualBubbleCoords, state.selectedManualBoxIndex); // 恢复
            isDragging = false; dragTarget = null; dragIndex = -1; $('.image-container').css('cursor', 'crosshair');
            return;
        }

        const x1_rel = (finalLeft - metrics.visualContentOffsetX) / metrics.scaleX;
        const y1_rel = (finalTop - metrics.visualContentOffsetY) / metrics.scaleY;
        const x2_rel = x1_rel + finalWidth / metrics.scaleX;
        const y2_rel = y1_rel + finalHeight / metrics.scaleY;

        const x1 = Math.max(0, Math.round(x1_rel));
        const y1 = Math.max(0, Math.round(y1_rel));
        const x2 = Math.min(metrics.naturalWidth, Math.round(x2_rel));
        const y2 = Math.min(metrics.naturalHeight, Math.round(y2_rel));

        // ... (后续状态更新逻辑不变，但现在基于更准确的原生坐标)
        const originalCoords = state.manualBubbleCoords[dragIndex];
            if (x1 < x2 && y1 < y2) {
                const newCoordArray = [x1, y1, x2, y2];
                if (JSON.stringify(originalCoords) !== JSON.stringify(newCoordArray)) {
                    const currentCoords = [...state.manualBubbleCoords];
                    currentCoords[dragIndex] = newCoordArray;
                    state.setManualCoords(currentCoords);
                    coordsChanged = true;
                    console.log(`框 ${dragIndex} 坐标已更新:`, newCoordArray);
                } else {
                     console.log(`框 ${dragIndex} 拖动后位置未改变。`);
                }
            } else {
                console.warn("拖动后坐标无效，未更新状态。", {x1, y1, x2, y2});
                ui.drawBoundingBoxes(state.manualBubbleCoords, state.selectedManualBoxIndex);
            }

        dragTarget.css('cursor', 'grab');
        $('.image-container').css('cursor', 'crosshair');
        isDragging = false;
        dragTarget = null;
        dragIndex = -1;


    } else if (isResizing && resizeTargetIndex !== -1) { // 结束调整大小
        // ... (调整大小逻辑，也需要更新) ...
        console.log(`结束调整大小: 框 ${resizeTargetIndex}, 句柄 ${resizeHandleType}`);
        const targetBox = $(`.manual-bounding-box[data-index="${resizeTargetIndex}"]`);

        // 在 handleResizeHandleMouseDown 中，resizeImageInfo 应该存储的是 metrics
        // const { offsetLeft, offsetTop, scaleX, scaleY, naturalWidth, naturalHeight } = resizeImageInfo;
        // 改为：
        const metrics = resizeImageInfo; // 假设 resizeImageInfo 现在存储的是 calculateImageDisplayMetrics 的结果

        if (targetBox.length && metrics) {
            const finalLeft = parseFloat(targetBox.css('left'));
            const finalTop = parseFloat(targetBox.css('top'));
            const finalWidth = parseFloat(targetBox.css('width'));
            const finalHeight = parseFloat(targetBox.css('height'));

            const x1_rel = (finalLeft - metrics.visualContentOffsetX) / metrics.scaleX;
            const y1_rel = (finalTop - metrics.visualContentOffsetY) / metrics.scaleY;
            const x2_rel = x1_rel + finalWidth / metrics.scaleX;
            const y2_rel = y1_rel + finalHeight / metrics.scaleY;

            const x1 = Math.max(0, Math.round(x1_rel));
            const y1 = Math.max(0, Math.round(y1_rel));
            const x2 = Math.min(metrics.naturalWidth, Math.round(x2_rel));
            const y2 = Math.min(metrics.naturalHeight, Math.round(y2_rel));

            // ... (后续状态更新逻辑不变，但现在基于更准确的原生坐标)
            const originalCoords = state.manualBubbleCoords[resizeTargetIndex];
            if (x1 < x2 && y1 < y2 && (x2 - x1) >= 5 && (y2 - y1) >= 5) {
                const newCoordArray = [x1, y1, x2, y2];
                 if (JSON.stringify(originalCoords) !== JSON.stringify(newCoordArray)) {
                    const currentCoords = [...state.manualBubbleCoords];
                    currentCoords[resizeTargetIndex] = newCoordArray;
                    state.setManualCoords(currentCoords);
                    coordsChanged = true;
                    console.log(`框 ${resizeTargetIndex} 调整大小后坐标已更新:`, newCoordArray);
                 } else {
                      console.log(`框 ${resizeTargetIndex} 调整大小后坐标未改变。`);
                 }
            } else {
                console.warn("调整大小后坐标或尺寸无效，撤销更改。", {x1, y1, x2, y2});
                ui.showGeneralMessage("调整后的框尺寸过小或无效，已撤销。", "warning");
                ui.drawBoundingBoxes(state.manualBubbleCoords, state.selectedManualBoxIndex);
            }
        } else {
            console.error("结束调整大小时找不到目标框或图片信息。");
            ui.drawBoundingBoxes(state.manualBubbleCoords, state.selectedManualBoxIndex);
        }

        isResizing = false;
        resizeHandleType = null;
        resizeTargetIndex = -1;
        initialBoxCoords = null;
        resizeImageInfo = null;
        $('.image-container').css('cursor', 'crosshair');
        $('body').css('cursor', 'default');
        ui.drawBoundingBoxes(state.manualBubbleCoords, state.selectedManualBoxIndex);
    }

    if(coordsChanged && state.isLabelingModeActive) {
        ui.toggleLabelingModeUI(true);
    }
}

/**
 * 处理点击标注框的事件 (选择框)
 * @param {jQuery.Event} e - 事件对象
 */
export function handleBoxClick(e) {
    if (!state.isLabelingModeActive) return;

    const clickedBox = $(e.currentTarget);
    const index = parseInt(clickedBox.data('index'));

    if (!isNaN(index)) {
        state.setSelectedManualBoxIndex(index);
        // 重新绘制所有框以更新高亮
        ui.drawBoundingBoxes(state.manualBubbleCoords, index);
        // 更新"删除选中框"按钮状态
        $('#deleteSelectedBoxButton').prop('disabled', false);
        console.log(`选中了标注框: ${index}`);
    }
    e.stopPropagation(); // 阻止事件冒泡到图片容器，防止误触发绘制
}

/**
 * 处理"删除选中框"按钮点击
 */
export function handleDeleteSelectedBoxClick() {
    // 1. --- 前置检查 ---
    if (!state.isLabelingModeActive || state.selectedManualBoxIndex === -1) {
        console.warn("删除操作被阻止：不在标注模式或没有选中框。");
        return;
    }

    const indexToDelete = state.selectedManualBoxIndex;
    const currentImage = state.getCurrentImage();

    if (!currentImage || !currentImage.originalDataURL) {
        ui.showGeneralMessage("无法删除：缺少原始图像数据。", "error");
        console.error("删除操作失败：缺少必要的图像数据。");
        return;
    }

    // 2. --- 获取关键数据 (在修改 state 之前!) ---
    const deletedCoords = [...state.manualBubbleCoords[indexToDelete]]; // 获取要删除框的原生坐标 (深拷贝)

    // 3. --- 用户确认 ---
    if (confirm(`确定要删除选中的标注框 #${indexToDelete + 1} 吗？对应区域将恢复为原图内容。`)) {

        // 4. --- UI 反馈：显示处理中 ---
        ui.showGeneralMessage("正在更新图像...", "info", false, 0); // 使用通用消息显示加载状态，不自动消失

        // 5. --- 准备 Canvas 操作 ---
        const originalImage = new Image();
        const currentDisplayImage = new Image();
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d', { willReadFrequently: true }); // 优化频繁读取

        // 6. --- 使用 Promise 加载图片 ---
        Promise.all([
            new Promise((resolve, reject) => {
                originalImage.onload = () => { console.log("原图加载完成"); resolve(); };
                originalImage.onerror = (err) => { console.error("原图加载失败", err); reject(new Error("无法加载原图")); };
                originalImage.src = currentImage.originalDataURL;
            }),
            new Promise((resolve, reject) => {
                currentDisplayImage.onload = () => { console.log("当前显示图加载完成"); resolve(); };
                currentDisplayImage.onerror = (err) => { console.error("当前显示图加载失败", err); reject(new Error("无法加载当前显示的图像")); };
                // 使用当前显示的图像，可能是原图或翻译图
                currentDisplayImage.src = $('#translatedImageDisplay').attr('src');
            })
        ]).then(() => {
            // 7. --- 图片加载成功后执行 Canvas 绘制 ---
            try {
                console.log("图片加载成功，开始 Canvas 操作...");
                // 7.1 获取精确的图像显示指标
                const imageElement = $('#translatedImageDisplay');
                const metrics = ui.calculateImageDisplayMetrics(imageElement);
                if (!metrics) {
                    throw new Error("无法计算图像显示指标，Canvas 操作中止。");
                }
                console.log("获取到的图像指标:", metrics);

                // 7.2 设置 Canvas 尺寸为图像原生尺寸
                canvas.width = metrics.naturalWidth;
                canvas.height = metrics.naturalHeight;
                console.log(`Canvas 尺寸设置为: ${canvas.width}x${canvas.height}`);

                // 7.3 将当前显示的图像绘制到 Canvas 上 (使用原生尺寸绘制)
                ctx.drawImage(currentDisplayImage, 0, 0, canvas.width, canvas.height);
                console.log("已将当前显示图绘制到 Canvas。");

                // 7.4 计算要从原图复制的区域 (原生坐标 sx, sy, sWidth, sHeight)
                const sx = deletedCoords[0];
                const sy = deletedCoords[1];
                const sWidth = deletedCoords[2] - sx;
                const sHeight = deletedCoords[3] - sy;

                // 7.5 计算要绘制到 Canvas 上的目标区域 (原生坐标 dx, dy, dWidth, dHeight)
                const dx = sx;
                const dy = sy;
                const dWidth = sWidth;
                const dHeight = sHeight;

                // 7.6 从原图复制指定区域到 Canvas，覆盖掉已删除气泡的内容
                if (sWidth > 0 && sHeight > 0 && dWidth > 0 && dHeight > 0) {
                    console.log(`准备绘制原图区域: sx=${sx}, sy=${sy}, sW=${sWidth}, sH=${sHeight} 到 Canvas 区域: dx=${dx}, dy=${dy}, dW=${dWidth}, dH=${dHeight}`);
                    ctx.drawImage(originalImage, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight);
                    console.log("已将原图区域绘制到 Canvas。");
                } else {
                    console.warn(`删除框 #${indexToDelete} 时跳过绘制，源或目标尺寸无效 (sW=${sWidth}, sH=${sHeight}, dW=${dWidth}, dH=${dHeight})`);
                }

                // 8. --- 获取新的图像数据并更新状态 ---
                const newDataURL = canvas.toDataURL('image/png');
                console.log("生成新的 DataURL 完成。");

                // 8.1 更新前端显示的图像
                ui.updateTranslatedImage(newDataURL);
                console.log("已更新前端显示的图像。");

                // 8.2 如果当前有翻译图，则更新state中的translatedDataURL
                if (currentImage.translatedDataURL) {
                    state.updateCurrentImageProperty('translatedDataURL', newDataURL);
                    console.log("已更新 state 中的 translatedDataURL。");
                } else {
                    // 如果是未翻译的图片，则保存修改后的原图到state
                    state.updateCurrentImageProperty('originalDataURL', newDataURL);
                    console.log("图片未翻译，已更新 state 中的 originalDataURL。");
                }

                // 8.5 --- 【新增 Canvas 操作：同步修改 cleanImageData】 *** ---
                if (currentImage.cleanImageData) {
                    console.log("开始同步修补 cleanImageData...");
                    const cleanImage = new Image(); // 用于加载干净背景
                    const cleanCanvas = document.createElement('canvas'); // 新的 Canvas
                    const cleanCtx = cleanCanvas.getContext('2d', { willReadFrequently: true });

                    // 创建 Promise 来加载干净背景图
                    const cleanLoadPromise = new Promise((resolve, reject) => {
                        cleanImage.onload = () => { console.log("干净背景图加载完成"); resolve(); };
                        cleanImage.onerror = (err) => { console.error("干净背景图加载失败", err); reject(new Error("无法加载干净背景图")); };
                        // 从 state 获取干净背景数据，并加上 Data URL 前缀
                        cleanImage.src = 'data:image/png;base64,' + state.getCurrentImage().cleanImageData;
                    });

                    // originalImage 已经加载好了，所以只需要等待 cleanImage 加载
                    cleanLoadPromise.then(() => {
                        try {
                            // 使用与之前相同的 metrics
                            cleanCanvas.width = metrics.naturalWidth;
                            cleanCanvas.height = metrics.naturalHeight;

                            // 将干净背景绘制到其 Canvas 上
                            cleanCtx.drawImage(cleanImage, 0, 0, cleanCanvas.width, cleanCanvas.height);
                            console.log("已将干净背景绘制到其 Canvas。");

                            // 将原图的相同区域绘制到干净背景的 Canvas 上
                            // 使用之前计算好的 sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight
                            if (sWidth > 0 && sHeight > 0 && dWidth > 0 && dHeight > 0) {
                                console.log(`准备将原图区域绘制到 cleanImageData Canvas: sx=${sx}, sy=${sy}, sW=${sWidth}, sH=${sHeight}, dx=${dx}, dy=${dy}, dW=${dWidth}, dH=${dHeight}`);
                                cleanCtx.drawImage(originalImage, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight);
                                console.log("已将原图区域绘制到 cleanImageData 的 Canvas 副本。");

                                // 获取修改后的干净背景 DataURL
                                const newCleanDataURL = cleanCanvas.toDataURL('image/png');
                                // 更新 state 中的 cleanImageData (存储时不带前缀)
                                state.updateCurrentImageProperty('cleanImageData', newCleanDataURL.split(',')[1]);
                                console.log("已更新 state 中的 cleanImageData。");
                            } else {
                                 console.warn(`修补 cleanImageData 时跳过绘制，源或目标尺寸无效`);
                            }
                        } catch (cleanPatchError) {
                            console.error("修补 cleanImageData 时发生 Canvas 错误:", cleanPatchError);
                            // 如果修补干净背景失败，仅记录错误，主流程继续
                        }
                    }).catch(cleanLoadError => {
                        console.error("加载 cleanImageData 进行修补时失败:", cleanLoadError);
                        // 加载失败，无法修补干净背景
                    });
                } else {
                    console.warn("删除框时未找到 cleanImageData，无法进行同步修补。");
                }
                // --- *** 结束新增 Canvas 操作 *** ---

                // 8.3 更新 state 中的坐标数组 (删除对应项)
                const currentCoords = [...state.manualBubbleCoords];
                currentCoords.splice(indexToDelete, 1);
                state.setManualCoords(currentCoords); // 这会设置 hasUnsavedChanges=true
                console.log(`已从 state 中删除坐标 #${indexToDelete}，剩余 ${currentCoords.length} 个。`);

                // 8.4 同步删除 state 中对应的文本和设置数据（仅当图片已翻译时）
                if (currentImage.translatedDataURL) {
                    const syncArrays = ['bubbleTexts', 'textboxTexts', 'originalTexts', 'bubbleSettings', 'bubbleCoords'];
                    syncArrays.forEach(key => {
                        if (currentImage[key] && Array.isArray(currentImage[key]) && currentImage[key].length > indexToDelete) {
                            const currentArray = [...currentImage[key]];
                            currentArray.splice(indexToDelete, 1);
                            state.updateCurrentImageProperty(key, currentArray);
                            console.log(`已同步删除 state 中 ${key} 的第 ${indexToDelete} 项。`);
                        }
                        // 特别处理内存中的 bubbleSettings (如果正在编辑模式)
                        if (key === 'bubbleSettings' && state.editModeActive && state.bubbleSettings && state.bubbleSettings.length > indexToDelete) {
                            const currentMemSettings = [...state.bubbleSettings];
                            currentMemSettings.splice(indexToDelete, 1);
                            state.setBubbleSettings(currentMemSettings); // 更新内存状态
                            console.log(`已同步删除内存中 bubbleSettings 的第 ${indexToDelete} 项。`);
                        }
                    });
                }

                // 9. --- UI 清理与更新 ---
                // 9.1 重置选中索引
                state.setSelectedManualBoxIndex(-1);

                // 9.2 重新绘制所有标注框
                ui.drawBoundingBoxes(state.manualBubbleCoords);
                console.log("已重新绘制更新后的标注框。");

                // 9.3 完成操作，隐藏提示并显示成功消息
                $(".message.info").fadeOut(300, function() { $(this).remove(); });
                ui.showGeneralMessage(`标注框 #${indexToDelete + 1} 已删除。`, "success", false, 3000);

                // 10. 触发自动存档
                session.triggerAutoSave();

            } catch (error) {
                // 捕获 Canvas 操作中可能出现的错误
                console.error("删除标注框时发生错误:", error);
                $(".message.info").fadeOut(300, function() { $(this).remove(); });
                ui.showGeneralMessage(`删除标注框时发生错误: ${error.message}`, "error");
            }
        }).catch(error => {
            // 捕获图片加载失败的错误
            console.error("加载图片失败:", error);
            $(".message.info").fadeOut(300, function() { $(this).remove(); });
            ui.showGeneralMessage(`删除标注框时加载图片失败: ${error.message}`, "error");
        });

    } else {
        console.log("用户取消了删除操作。");
    }
}

/**
 * 处理"使用手动框翻译"按钮点击
 */
export function handleUseManualBoxesClick() {
    if (!state.isLabelingModeActive) return;

    // --- 代码继续 ---
    
    // 提取当前图片和坐标
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.originalDataURL) {
        ui.showGeneralMessage("没有有效的图片用于翻译。", "warning");
        return;
    }
    
    if (state.manualBubbleCoords.length === 0) {
        ui.showGeneralMessage("没有标注框可用于翻译。", "warning");
        return;
    }
    
    // 检查是否有未保存的更改，如果有则先保存
    if (state.hasUnsavedChanges) {
        if (!confirm("您有未保存的标注更改，在翻译之前需要先保存。要继续吗？")) {
            return; // 用户取消
        }
        // 直接保存并继续，不退出标注模式
        if (!state.saveManualCoordsToImage()) { // 调用保存函数
            ui.showGeneralMessage("保存标注失败！无法继续翻译。", "error");
            return;
        }
        ui.showGeneralMessage("标注已保存，继续翻译...", "success", false, 1500);
        ui.renderThumbnails(); // 保存成功后更新缩略图
    }
    
    // 1. 显示加载提示
    ui.showLoading("正在使用手动标注框翻译...");
    
    // 2. 准备参数
    const imageData = currentImage.originalDataURL.split(',')[1];
    
    // 3. 组装要传递给翻译 API 的参数
    const requestData = {
        image: imageData,
        model_provider: $('#modelProvider').val(),
        model_name: $('#modelName').val(),
        api_key: $('#apiKey').val(),
        source_language: $('#sourceLanguage').val(),
        target_language: $('#targetLanguage').val(),
        prompt_content: $('#promptContent').val(),
        textbox_prompt_content: $('#textboxPromptContent').val(),
        use_textbox_prompt: $('#enableTextboxPrompt').is(':checked'),
        font_size: $('#autoFontSize').is(':checked') ? 'auto' : parseInt($('#fontSize').val()),
        auto_font_size: $('#autoFontSize').is(':checked'),
        font_family: $('#fontFamily').val(),
        layout_direction: $('#layoutDirection').val(),
        text_color: $('#textColor').val(),
        fill_color: $('#fillColor').val(),
        use_inpainting: $('#useInpainting').val(),
        ocr_engine: $('#ocrEngine').val(),
        provided_coords: state.manualBubbleCoords // 手动标注的坐标
    };
    
    // 添加百度OCR相关参数（如果选择了百度OCR）
    if ($('#ocrEngine').val() === 'baidu_standard' || $('#ocrEngine').val() === 'baidu_accurate') {
        requestData.baidu_api_key = $('#baiduApiKey').val();
        requestData.baidu_secret_key = $('#baiduSecretKey').val();
    }
    
    // 调用翻译 API
    api.translateImageApi(requestData) // 这是通用翻译接口
        .then(response => {
            ui.hideLoading();
            
            if (response.success) {
                // 更新当前图片的翻译结果数据
                currentImage.translatedDataURL = `data:image/png;base64,${response.translated_image}`;
                currentImage.cleanImageData = response.clean_background_image || null;
                currentImage.bubbleCoords = response.bubble_coords || state.manualBubbleCoords;
                currentImage.bubbleTexts = response.translated_texts || [];
                
                // 更新图像显示和翻译结果面板
                $('#translatedImage').attr('src', currentImage.translatedDataURL);
                ui.displayTranslationResults(response.original_texts, response.translated_texts);
                
                // 更新编辑按钮状态
                $('#editButton').prop('disabled', false);
                
                // 保存模型历史（排除敏感API）
                if (requestData.model_provider !== 'baidu_translate' && requestData.model_provider !== 'youdao_translate') {
                    api.saveModelInfoApi(requestData.model_provider, requestData.model_name);
                }
                
                // 成功翻译消息
                ui.showGeneralMessage("使用手动标注框翻译成功！", "success", false, 3000);
                
                // 退出标注模式（因为完成了目标）
                exitLabelingMode(false); // 传入 false 表示不提示保存（因为已经保存过了）
                
                // 触发自动存档
                session.triggerAutoSave();
            } else {
                throw new Error(response.error || "翻译失败，未返回结果。");
            }
        })
        .catch(error => {
            ui.hideLoading();
            ui.showGeneralMessage(`翻译失败: ${error.message}`, "error");
        });
}

/**
 * 处理在标注框上按下鼠标按钮的事件 (开始拖动或选择)
 * @param {jQuery.Event} e - 事件对象
 */
export function handleBoxMouseDown(e) {
    // --- 新增：如果点击的是手柄，则忽略此事件 ---
    if ($(e.target).hasClass('resize-handle')) {
        return;
    }
    // ----------------------------------------
    if (!state.isLabelingModeActive || e.button !== 0 || isResizing) return; // 调整大小时不进行拖动

    isDragging = true;
    dragTarget = $(e.currentTarget);
    dragIndex = parseInt(dragTarget.data('index'));

    // 如果点击的不是当前选中的框，先执行选择逻辑
    if (dragIndex !== state.selectedManualBoxIndex) {
        handleBoxClick(e); // 调用之前的选择逻辑
    }

    // 记录拖动起始信息
    dragStartX = e.pageX;
    dragStartY = e.pageY;
    boxInitialX = parseFloat(dragTarget.css('left'));
    boxInitialY = parseFloat(dragTarget.css('top'));

    dragTarget.css('cursor', 'grabbing'); // 设置拖动光标
    $('.image-container').css('cursor', 'grabbing'); // 容器也显示拖动光标

    console.log(`开始拖动框: ${dragIndex}`);
    e.stopPropagation(); // 阻止冒泡到容器的 mousedown (防止开始绘制新框)
    e.preventDefault();
}

/**
 * 新增：处理在调整大小手柄上按下鼠标的事件
 * @param {jQuery.Event} e - 事件对象
 */
export function handleResizeHandleMouseDown(e) {
    if (!state.isLabelingModeActive || e.button !== 0 || isDrawing || isDragging) return;

    isResizing = true;
    const handle = $(e.currentTarget);
    resizeHandleType = handle.data('handle');
    resizeTargetIndex = parseInt(handle.data('parent-index'));

    if (resizeTargetIndex !== state.selectedManualBoxIndex) {
        state.setSelectedManualBoxIndex(resizeTargetIndex);
        ui.drawBoundingBoxes(state.manualBubbleCoords, resizeTargetIndex);
    }

    resizeStartX = e.pageX;
    resizeStartY = e.pageY;
    initialBoxCoords = [...state.manualBubbleCoords[resizeTargetIndex]];

    // 修改点：存储精确的图像显示指标
    const imageElement = $('#translatedImageDisplay');
    resizeImageInfo = ui.calculateImageDisplayMetrics(imageElement);
    if (!resizeImageInfo) {
        console.error("handleResizeHandleMouseDown: 无法获取图像指标，取消调整大小。");
        isResizing = false;
        return;
    }
    // ---------------------------------

    $('body').css('cursor', handle.css('cursor'));

    console.log(`开始调整大小: 框 ${resizeTargetIndex}, 句柄 ${resizeHandleType}`);
    e.stopPropagation();
    e.preventDefault();
} 