// src/app/static/js/session.js

import * as state from './state.js'; // 引入状态模块以获取当前状态
import * as ui from './ui.js';     // 引入 UI 模块以显示消息和加载状态
import * as api from './api.js';     // 引入 API 模块以调用后端接口
import * as main from './main.js'; // 需要 main 来切换图片
import * as editMode from './edit_mode.js'; // 需要退出编辑模式
import * as constants from './constants.js';

/**
 * 收集当前应用的完整状态，用于保存。
 * @returns {object | null} 包含当前状态的对象，如果无法获取则返回 null。
 */
function collectCurrentSessionData() {
    console.log("开始收集当前会话状态 (含描边)...");

    // 检查是否有图片数据，如果没有则无法保存有意义的会话
    if (!state.images || state.images.length === 0) {
        ui.showGeneralMessage("没有图片数据，无法保存会话。", "warning");
        console.warn("收集状态失败：图片列表为空。");
        return null;
    }

    // 1. 收集 UI 设置（仅保存渲染相关设置，AI设置使用全局 user_settings.json）
    const uiSettings = {
        // 语言设置（与漫画相关）
        targetLanguage: $('#targetLanguage').val(),
        sourceLanguage: $('#sourceLanguage').val(),
        // 渲染设置
        fontSize: $('#fontSize').val(),
        autoFontSize: $('#autoFontSize').is(':checked'),
        fontFamily: $('#fontFamily').val(),
        layoutDirection: $('#layoutDirection').val(),
        textColor: $('#textColor').val(),
        // 注意: 全局 rotationAngle 已移除，现在使用每个气泡独立的旋转角度
        // 修复/填充设置
        useInpaintingMethod: $('#useInpainting').val(),
        fillColor: $('#fillColor').val(),
        // 描边设置
        strokeEnabled: state.strokeEnabled,
        strokeColor: state.strokeColor,
        strokeWidth: state.strokeWidth,
        // 注意：AI服务商、模型、提示词、RPM限制等设置不再保存到会话中
        // 这些设置统一使用 config/user_settings.json 中的全局配置
    };
    console.log("收集到的 UI 设置 (含描边):", uiSettings);

    // 2. 收集图片状态 (深拷贝以避免修改原始状态)
    // 注意：这里包含了 Base64 数据，会比较大
    const imagesData = JSON.parse(JSON.stringify(state.images));
    console.log(`收集到 ${imagesData.length} 张图片的状态数据。`);

    // 3. 收集当前图片索引
    const currentIndex = state.currentImageIndex;
    console.log(`当前图片索引: ${currentIndex}`);

    // 4. 组合所有数据
    const sessionData = {
        ui_settings: uiSettings,
        images: imagesData,
        currentImageIndex: currentIndex
    };

    console.log("会话状态收集完成 (含描边)。");
    return sessionData;
}

/**
 * 触发"保存"当前会话流程。
 * 仅在书籍/章节模式下可用，自动保存到章节路径。
 * 快速翻译模式下不支持保存功能。
 */
export function triggerSaveCurrentSession() {
    console.log("触发保存当前会话流程...");

    // 检查是否在书籍/章节模式
    const bookId = state.currentBookId;
    const chapterId = state.currentChapterId;
    
    if (bookId && chapterId) {
        // 书籍/章节模式：直接保存到章节路径，不弹出命名对话框
        console.log(`书籍/章节模式保存: book=${bookId}, chapter=${chapterId}`);
        saveChapterSession();
        return;
    }

    // 快速翻译模式：不支持保存
    console.log("快速翻译模式不支持保存功能");
    ui.showGeneralMessage("快速翻译模式不支持保存功能，请从书架进入章节进行翻译以使用保存功能", "warning");
}

// --- 书籍/章节会话支持 ---

/**
 * 将图片 URL 转换为 Base64 数据
 * @param {string} url - 图片 URL
 * @returns {Promise<string|null>} Base64 数据 URL，失败返回 null
 */
async function imageUrlToBase64(url) {
    if (!url || typeof url !== 'string') return null;
    // 如果已经是 Base64，直接返回
    if (url.startsWith('data:')) return url;
    // 如果不是 API URL，返回 null
    if (!url.startsWith('/api/')) return null;
    
    try {
        const response = await fetch(url);
        if (!response.ok) return null;
        
        const blob = await response.blob();
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.onerror = () => resolve(null);
            reader.readAsDataURL(blob);
        });
    } catch (error) {
        console.error(`转换图片 URL 失败: ${url}`, error);
        return null;
    }
}

/**
 * 将会话中的所有图片 URL 转换为 Base64
 * @param {Array} images - 图片数组
 * @param {Function} progressCallback - 进度回调 (current, total)
 */
async function convertImagesToBase64(images, progressCallback) {
    const total = images.length;
    
    for (let i = 0; i < total; i++) {
        const img = images[i];
        if (progressCallback) progressCallback(i + 1, total);
        
        // 转换原图
        if (img.originalDataURL && img.originalDataURL.startsWith('/api/')) {
            const base64 = await imageUrlToBase64(img.originalDataURL);
            if (base64) img.originalDataURL = base64;
        }
        
        // 转换翻译图
        if (img.translatedDataURL && img.translatedDataURL.startsWith('/api/')) {
            const base64 = await imageUrlToBase64(img.translatedDataURL);
            if (base64) img.translatedDataURL = base64;
        }
        
        // 转换干净背景（cleanImageData 存储的是纯 Base64，不带 data: 前缀）
        if (img.cleanImageData && img.cleanImageData.startsWith('/api/')) {
            const base64 = await imageUrlToBase64(img.cleanImageData);
            if (base64) {
                // 移除 data:image/png;base64, 前缀
                img.cleanImageData = base64.replace(/^data:image\/\w+;base64,/, '');
            }
        }
    }
}

/**
 * 按路径加载会话数据
 * @param {string} sessionPath - 会话路径（相对于 data/sessions 目录）
 * @returns {Promise<boolean>} 是否加载成功
 */
export async function loadSessionByPath(sessionPath) {
    console.log(`请求按路径加载会话: ${sessionPath}`);
    
    // 退出可能存在的编辑模式
    if (state.editModeActive) {
        editMode.exitEditMode();
    }
    
    try {
        // 调用 API 按路径加载会话
        const response = await api.loadSessionByPathApi(sessionPath);
        
        if (response.success && response.session_data) {
            console.log("成功按路径加载会话数据:", response.session_data);
            const loadedData = response.session_data;
            const uiSettings = loadedData.ui_settings || {};
            
            // 恢复状态
            state.clearImages();
            state.setImages(loadedData.images || []);
            
            // 将图片 URL 转换为 Base64（用于 Canvas 操作和翻译功能）
            if (state.images.length > 0) {
                // 显示进度条
                $("#translationProgressBar").show();
                ui.updateProgressBar(0, "正在加载图片...");
                
                await convertImagesToBase64(state.images, (current, total) => {
                    const progress = (current / total) * 100;
                    ui.updateProgressBar(progress, `加载图片 ${current}/${total}...`);
                });
                
                ui.updateProgressBar(100, "加载完成");
                console.log("图片加载完成，已转换为 Base64");
                
                // 延迟隐藏进度条
                setTimeout(() => {
                    $("#translationProgressBar").hide();
                }, 500);
            }
            
            let newIndex = loadedData.currentImageIndex !== undefined ? loadedData.currentImageIndex : -1;
            if (newIndex >= state.images.length || newIndex < 0) {
                newIndex = state.images.length > 0 ? 0 : -1;
            }
            state.setCurrentImageIndex(newIndex);
            state.setCurrentSessionName(sessionPath);
            
            // 刷新界面
            ui.renderThumbnails();
            if (state.currentImageIndex !== -1) {
                main.switchImage(state.currentImageIndex);
                ui.showResultSection(true);
            } else {
                ui.showResultSection(false);
            }
            
            // 恢复 UI 设置
            restoreUiSettings(uiSettings);
            ui.updateButtonStates();
            
            return true;
        } else {
            console.warn("按路径加载会话失败:", response.error);
            return false;
        }
    } catch (error) {
        // 加载失败可能是会话不存在（首次进入章节），静默处理
        console.log("按路径加载会话失败（可能是首次进入）:", error.message || error);
        return false;
    }
}

/**
 * 为书籍章节保存会话（使用分批保存，避免大数据量导致的字符串长度限制）
 * 自动使用章节的会话路径进行保存
 */
export async function saveChapterSession() {
    const bookId = state.currentBookId;
    const chapterId = state.currentChapterId;
    
    if (!bookId || !chapterId) {
        console.log("未在书籍/章节模式，不支持保存");
        ui.showGeneralMessage("快速翻译模式不支持保存功能", "warning");
        return false;
    }
    
    console.log(`保存章节会话（分批）: book=${bookId}, chapter=${chapterId}`);
    
    // 检查是否有图片数据
    if (!state.images || state.images.length === 0) {
        ui.showGeneralMessage("没有图片数据可保存", "warning");
        return false;
    }
    
    // 构建会话路径（格式必须与 bookshelf_manager.get_chapter_session_path 一致）
    const sessionPath = `bookshelf/${bookId}/${chapterId}`;
    
    try {
        // 使用分批保存
        const result = await saveSessionInBatches(sessionPath);
        
        if (result.success) {
            ui.showGeneralMessage("章节进度已保存", "success");
            return true;
        } else {
            ui.showGeneralMessage("保存失败: " + (result.error || "未知错误"), "error");
            return false;
        }
    } catch (error) {
        console.error("保存章节会话出错:", error);
        ui.showGeneralMessage("保存出错: " + error.message, "error");
        return false;
    }
}

/**
 * 分批保存会话（避免一次性传输大量 Base64 数据）
 * @param {string} sessionPath - 会话路径
 * @returns {Promise<{success: boolean, error?: string}>}
 */
async function saveSessionInBatches(sessionPath) {
    console.log(`开始分批保存会话: ${sessionPath}`);
    
    const allImages = state.images;
    const totalImages = allImages.length;
    
    // 显示进度
    $("#translationProgressBar").show();
    ui.updateProgressBar(0, "准备保存...");
    
    try {
        // 步骤1: 收集元数据（仅渲染相关设置，AI设置使用全局配置）
        ui.updateProgressBar(5, "收集元数据...");
        
        const uiSettings = {
            targetLanguage: $('#targetLanguage').val(),
            sourceLanguage: $('#sourceLanguage').val(),
            fontSize: $('#fontSize').val(),
            autoFontSize: $('#autoFontSize').is(':checked'),
            fontFamily: $('#fontFamily').val(),
            layoutDirection: $('#layoutDirection').val(),
            textColor: $('#textColor').val(),
            // 全局 rotationAngle 已移除
            useInpaintingMethod: $('#useInpainting').val(),
            fillColor: $('#fillColor').val(),
            strokeEnabled: state.strokeEnabled,
            strokeColor: state.strokeColor,
            strokeWidth: state.strokeWidth,
        };
        
        // 图片元数据（不含 Base64 数据）
        const imagesMeta = allImages.map((img, idx) => {
            const meta = {};
            for (const key in img) {
                if (key !== 'originalDataURL' && key !== 'translatedDataURL' && key !== 'cleanImageData') {
                    meta[key] = img[key];
                }
            }
            // 标记哪些图片数据存在
            meta.hasOriginalData = !!img.originalDataURL;
            meta.hasTranslatedData = !!img.translatedDataURL;
            meta.hasCleanData = !!img.cleanImageData;
            return meta;
        });
        
        const metadata = {
            ui_settings: uiSettings,
            images_meta: imagesMeta,
            currentImageIndex: state.currentImageIndex
        };
        
        // 步骤2: 调用开始保存 API
        ui.updateProgressBar(10, "初始化保存...");
        
        const startResponse = await api.batchSaveStartApi(sessionPath, metadata);
        if (!startResponse.success) {
            throw new Error(startResponse.error || '初始化保存失败');
        }
        
        const sessionFolder = startResponse.session_folder;
        console.log(`会话文件夹: ${sessionFolder}`);
        
        // 步骤3: 逐张保存图片
        let savedCount = 0;
        let failedCount = 0;
        
        // 辅助函数：判断是否是 Base64 数据（需要上传）还是 URL（已保存，跳过）
        const isBase64Data = (data) => data && typeof data === 'string' && data.startsWith('data:');
        
        for (let i = 0; i < totalImages; i++) {
            const img = allImages[i];
            const progress = 10 + (i / totalImages) * 80; // 10% - 90%
            ui.updateProgressBar(progress, `保存图片 ${i + 1}/${totalImages}...`);
            
            // 保存原图（仅当是 Base64 数据时才上传，URL 说明已保存过）
            if (isBase64Data(img.originalDataURL)) {
                try {
                    const resp = await api.batchSaveImageApi(sessionFolder, i, 'original', img.originalDataURL);
                    if (!resp.success) {
                        console.error(`保存图片 ${i} original 失败:`, resp.error);
                        failedCount++;
                    }
                } catch (e) {
                    console.error(`保存图片 ${i} original 出错:`, e);
                    failedCount++;
                }
            }
            
            // 保存翻译图
            if (isBase64Data(img.translatedDataURL)) {
                try {
                    const resp = await api.batchSaveImageApi(sessionFolder, i, 'translated', img.translatedDataURL);
                    if (!resp.success) {
                        console.error(`保存图片 ${i} translated 失败:`, resp.error);
                        failedCount++;
                    }
                } catch (e) {
                    console.error(`保存图片 ${i} translated 出错:`, e);
                    failedCount++;
                }
            }
            
            // 保存干净背景（cleanImageData 可能是纯 Base64 不带 data: 前缀）
            if (img.cleanImageData && typeof img.cleanImageData === 'string' && !img.cleanImageData.startsWith('/api/')) {
                try {
                    const resp = await api.batchSaveImageApi(sessionFolder, i, 'clean', img.cleanImageData);
                    if (!resp.success) {
                        console.error(`保存图片 ${i} clean 失败:`, resp.error);
                        failedCount++;
                    }
                } catch (e) {
                    console.error(`保存图片 ${i} clean 出错:`, e);
                    failedCount++;
                }
            }
            
            savedCount++;
        }
        
        // 步骤4: 完成保存
        ui.updateProgressBar(95, "完成保存...");
        
        const completeResponse = await api.batchSaveCompleteApi(sessionFolder, imagesMeta);
        if (!completeResponse.success) {
            throw new Error(completeResponse.error || '完成保存失败');
        }
        
        ui.updateProgressBar(100, "保存完成");
        
        console.log(`分批保存完成: ${savedCount} 张图片, ${failedCount} 个失败`);
        
        return { success: true };
        
    } catch (error) {
        console.error("分批保存失败:", error);
        return { success: false, error: error.message };
    } finally {
        setTimeout(() => {
            $("#translationProgressBar").hide();
        }, 1000);
    }
}

/**
 * 恢复 UI 设置的辅助函数
 * @param {object} uiSettings - UI 设置对象
 */
function restoreUiSettings(uiSettings) {
    try {
        // 语言设置
        $('#targetLanguage').val(uiSettings.targetLanguage || 'zh');
        $('#sourceLanguage').val(uiSettings.sourceLanguage || 'japan');
        // 渲染设置
        $('#fontSize').val(uiSettings.fontSize || state.defaultFontSize);
        $('#autoFontSize').prop('checked', uiSettings.autoFontSize || false);
        $('#fontFamily').val(uiSettings.fontFamily || state.defaultFontFamily);
        $('#layoutDirection').val(uiSettings.layoutDirection || state.defaultLayoutDirection);
        $('#textColor').val(uiSettings.textColor || state.defaultTextColor);
        // 全局 rotationAngle 已移除，现在使用每个气泡独立的旋转角度
        // 修复/填充设置
        $('#useInpainting').val(uiSettings.useInpaintingMethod || 'false');
        $('#fillColor').val(uiSettings.fillColor || state.defaultFillColor);
        
        // 注意：AI服务商、模型、提示词、RPM限制等设置不再从会话恢复
        // 这些设置保持使用 user_settings.json 中的全局配置
        
        // 描边设置
        const strokeEnabled = uiSettings.strokeEnabled === undefined ? state.defaultStrokeEnabled : uiSettings.strokeEnabled;
        const strokeColor = uiSettings.strokeColor || state.defaultStrokeColor;
        const strokeWidth = uiSettings.strokeWidth === undefined ? state.defaultStrokeWidth : parseInt(uiSettings.strokeWidth);
        
        state.setStrokeEnabled(strokeEnabled);
        state.setStrokeColor(strokeColor);
        state.setStrokeWidth(strokeWidth);
        
        $('#strokeEnabled').prop('checked', strokeEnabled);
        $('#strokeColor').val(strokeColor);
        $('#strokeWidth').val(strokeWidth);
        $("#strokeOptions").toggle(strokeEnabled);
        
        // 触发 change 事件
        $('#useInpainting').trigger('change');
        $('#autoFontSize').trigger('change');
        $('#strokeEnabled').trigger('change');
        
        console.log("UI 设置已恢复");
    } catch (error) {
        console.error("恢复 UI 设置时出错:", error);
    }
}