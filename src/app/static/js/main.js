// src/app/static/js/main.js

// 引入所有需要的模块
import * as state from './state.js';
import * as ui from './ui.js';
import * as api from './api.js';
import * as events from './events.js';
import * as editMode from './edit_mode.js';
import * as constants from './constants.js'; // 导入前端常量
import * as labelingMode from './labeling_mode.js';
import * as session from './session.js'; // 导入session模块，用于自动存档
import * as hqTranslation from './high_quality_translation.js'; // 导入高质量翻译模块
// import $ from 'jquery'; // 假设 jQuery 已全局加载

/**
 * 辅助函数：加载图片并返回 Image 对象
 * @param {string} src - 图片的 data URL
 * @returns {Promise<HTMLImageElement>}
 */
export function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = (err) => {
            console.error("图片加载失败:", src, err);
            reject(err);
        };
        img.src = src;
    });
}

/**
 * 使用新的全局填充颜色重新渲染当前图片。
 * @param {string} newFillColor - 新的十六进制填充颜色值 (e.g., '#RRGGBB')
 */
export async function reRenderWithNewFillColor(newFillColor) {
    const currentImage = state.getCurrentImage();
    if (!currentImage || (!currentImage.translatedDataURL && !currentImage.originalDataURL)) {
        ui.showGeneralMessage("没有可应用新填充色的图片。", "warning");
        return;
    }
    if (!currentImage.bubbleCoords || currentImage.bubbleCoords.length === 0) {
        ui.showGeneralMessage("当前图片没有气泡区域可填充。", "info");
        // 即使没有气泡，也应该更新图片记录的填充色，以便下次翻译时使用
        state.updateCurrentImageProperty('fillColor', newFillColor);
        console.log(`图片 ${state.currentImageIndex} 无气泡，仅更新记录的填充色为 ${newFillColor}`);
        return;
    }
    
    // 检查是否使用了LAMA修复
    const usesLamaInpainting = (
        (currentImage.hasOwnProperty('_lama_inpainted') && currentImage._lama_inpainted === true) ||
        (currentImage.originalUseLama === true)
    );
    if (usesLamaInpainting) {
        ui.showGeneralMessage("当前图片使用了LAMA智能修复，不能应用纯色填充。", "warning");
        console.log("图片使用LAMA修复，跳过填充色变更");
        // 仍然更新记录的fillColor，以便未来可能的非LAMA修复使用
        state.updateCurrentImageProperty('fillColor', newFillColor);
        return;
    }

    const loadingMessageId = "fill_color_loading_message";
    ui.showGeneralMessage("正在应用新的填充颜色...", "info", false, 0, loadingMessageId);

    try {
        // 1. 确定基础图像源
        let baseImageSrcToFill;
        // 优先使用已有的 cleanImageData，因为它代表了最干净的无文本背景
        if (currentImage.cleanImageData) {
            baseImageSrcToFill = 'data:image/png;base64,' + currentImage.cleanImageData;
            console.log("使用 cleanImageData 作为填充基础");
        }
        // 其次是 _tempCleanImage (可能是之前修复或填充的结果)
        else if (currentImage._tempCleanImage) {
            baseImageSrcToFill = 'data:image/png;base64,' + currentImage._tempCleanImage;
            console.log("使用 _tempCleanImage 作为填充基础");
        }
        // 再次是原始图像，因为翻译图可能已经有旧的填充色或文字
        else if (currentImage.originalDataURL) {
            baseImageSrcToFill = currentImage.originalDataURL;
            console.log("使用 originalDataURL 作为填充基础");
        }
        // 如果连原始图像都没有（理论上不应该），最后才考虑翻译图
        else if (currentImage.translatedDataURL) {
            baseImageSrcToFill = currentImage.translatedDataURL;
            console.warn("警告：使用 translatedDataURL 作为填充基础，效果可能不理想");
        } else {
            throw new Error("没有可用的基础图像进行填充。");
        }

        const baseImage = await loadImage(baseImageSrcToFill);

        // 2. 创建 Canvas 并绘制基础图像
        const canvas = document.createElement('canvas');
        canvas.width = baseImage.naturalWidth;
        canvas.height = baseImage.naturalHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(baseImage, 0, 0);

        // 3. 使用新的填充颜色填充气泡区域
        console.log(`使用新的填充颜色 "${newFillColor}" 填充 ${currentImage.bubbleCoords.length} 个气泡区域`);
        currentImage.bubbleCoords.forEach(coords => {
            const [x1, y1, x2, y2] = coords;
            ctx.fillStyle = newFillColor;
            // 修复：增加1像素填充范围，确保完全覆盖
            ctx.fillRect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
        });

        // 4. 获取填充后的图像数据 (这将作为新的"干净背景"传递给渲染API)
        const newFilledCleanBgBase64 = canvas.toDataURL('image/png').split(',')[1];

        // 5. 更新当前图片的 fillColor 状态，并临时设置 _tempCleanImageForFill
        state.updateCurrentImageProperty('fillColor', newFillColor);
        currentImage._tempCleanImageForFill = newFilledCleanBgBase64; // 供 reRenderFullImage 使用

        // 6. 更新所有气泡设置的填充颜色（不管是否在编辑模式）
        if (currentImage.bubbleSettings && Array.isArray(currentImage.bubbleSettings) && 
            currentImage.bubbleSettings.length > 0) {
            
            console.log("更新所有气泡的独立填充颜色为新的全局填充颜色");
            
            // 深拷贝当前设置
            const updatedBubbleSettings = JSON.parse(JSON.stringify(currentImage.bubbleSettings));
            
            // 将所有气泡的fillColor设置为新的全局填充颜色
            updatedBubbleSettings.forEach(setting => {
                setting.fillColor = newFillColor;
            });
            
            // 更新气泡设置
            state.updateCurrentImageProperty('bubbleSettings', updatedBubbleSettings);
            
            // 如果在编辑模式下，也更新state.bubbleSettings
            if (state.editModeActive && state.bubbleSettings) {
                state.bubbleSettings.forEach(setting => {
                    setting.fillColor = newFillColor;
                });
                // 不调用setBubbleSettings，因为那会覆盖currentImage.bubbleSettings
            }
        }

        console.log("新的填充背景已生成，准备调用 reRenderFullImage");

        // 7. 调用 reRenderFullImage 来在其上渲染文本
        await editMode.reRenderFullImage(); // 假设它返回 Promise

        // 清理临时属性
        delete currentImage._tempCleanImageForFill;

        // 8. 更新UI，自动存档
        ui.clearGeneralMessageById(loadingMessageId);
        ui.showGeneralMessage("填充颜色已更新！", "success");
        session.triggerAutoSave(); // 保存更改

    } catch (error) {
        console.error("应用新填充颜色失败:", error);
        ui.clearGeneralMessageById(loadingMessageId);
        ui.showGeneralMessage(`应用填充颜色失败: ${error.message || '未知错误'}`, "error");
    }
}

// --- 初始化函数 ---

/**
 * 初始化应用状态和 UI
 */
export function initializeApp() {
    console.log('初始化应用...');
    
    // 1. 从 DOM 读取初始/默认设置并更新状态
    state.setDefaultFontSize(parseInt($('#fontSize').val()) || 25);
    state.setDefaultFontFamily($('#fontFamily').val() || 'fonts/msyh.ttc'); // 设置默认字体为微软雅黑
    state.setDefaultLayoutDirection($('#layoutDirection').val() || 'vertical');
    state.setDefaultTextColor($('#textColor').val() || '#000000');
    state.setDefaultFillColor($('#fillColor').val() || constants.DEFAULT_FILL_COLOR);
    state.setUseTextboxPrompt($('#enableTextboxPrompt').is(':checked'));
    
    // 初始化翻译语言设置
    state.setSourceLanguage($('#sourceLanguage').val() || 'auto');
    state.setTargetLanguage($('#targetLanguage').val() || 'zh-CN');
    state.setOcrEngine($('#ocrEngine').val() || 'auto');

    // 1.1 加载动态字体列表
    ui.loadFontList(state.defaultFontFamily);
    
    // 2. 初始化提示词设置 (调用 API)
    initializePromptSettings();
    initializeTextboxPromptSettings();
    initializeAiVisionOcrPromptSettings();

    // --- 初始化 rpm 状态 (从 state.js 的默认值开始) ---
    // state.js 中 rpmLimitTranslation 和 rpmLimitAiVisionOcr 已经用 constants 初始化了
    // 如果之前实现了从 localStorage 加载，那会在这里执行
    // const savedrpmTranslation = localStorage.getItem('rpmLimitTranslation');
    // state.setrpmLimitTranslation(savedrpmTranslation !== null ? parseInt(savedrpmTranslation) : constants.DEFAULT_rpm_TRANSLATION);
    // const savedrpmAiVision = localStorage.getItem('rpmLimitAiVisionOcr');
    // state.setrpmLimitAiVisionOcr(savedrpmAiVision !== null ? parseInt(savedrpmAiVision) : constants.DEFAULT_rpm_AI_VISION_OCR);
    
    // --- 更新UI输入框以反映初始/加载的rpm状态 ---
    ui.updaterpmInputFields(); // <--- 新增调用
    // ---------------------------------------------

    // 3. 初始化可折叠面板
    initializeCollapsiblePanels();

    // 4. 初始化高质量翻译模块
    try {
        // 默认隐藏自定义Base URL输入框
        $('#hqCustomBaseUrlDiv').hide();
        
        // 初始化高质量翻译相关UI事件
        console.log("初始化高质量翻译模块...");
    } catch (error) {
        console.error("初始化高质量翻译模块失败:", error);
    }

    // 4.1 初始化AI校对模块
    try {
        console.log("初始化AI校对模块...");
        // 导入并初始化校对模块
        import('./ai_proofreading.js').then(proofreading => {
            proofreading.initProofreadingUI();
        }).catch(error => {
            console.error("加载AI校对模块失败:", error);
        });
    } catch (error) {
        console.error("初始化AI校对模块失败:", error);
    }

    // 5. 初始化亮暗模式
    initializeThemeMode();

    // 6. 检查初始模型提供商并更新 UI
    checkInitialModelProvider();
    
    // 7. 初始化OCR引擎设置
    initializeOcrEngineSettings();

    // --- 新增：设置 AI Vision OCR 默认提示词 ---
    // 直接使用常量设置 textarea 的值
    // 最好通过 ui.js 来操作 DOM
    ui.setAiVisionOcrPrompt(constants.DEFAULT_AI_VISION_OCR_PROMPT);
    // 确保 state.js 中的状态也与此默认值一致（已在 state.js 中完成）
    // ------------------------------------------

    // 8. 绑定所有事件监听器
    events.bindEventListeners();

    // 9. 更新初始按钮状态
    ui.updateButtonStates();

    // 10. 初始化修复选项的显示状态
    const initialRepairMethod = $('#useInpainting').val();
    ui.toggleInpaintingOptions(
        initialRepairMethod === 'true' || initialRepairMethod === 'lama',
        initialRepairMethod === 'false'
    );

    // 11. 初始化 UI 显示
    ui.updateTranslatePromptUI(); // 更新漫画翻译提示词UI
    ui.updateAiVisionOcrPromptUI(); // 更新AI视觉OCR提示词UI

    // 初始化选择器的默认值
    $('#translatePromptModeSelect').val(state.isTranslateJsonMode ? 'json' : 'normal');
    $('#aiVisionPromptModeSelect').val(state.isAiVisionOcrJsonMode ? 'json' : 'normal');

    // 加载自动存档设置
    try {
        const autoSaveEnabled = localStorage.getItem('autoSaveEnabled') === 'true';
        state.setAutoSaveEnabled(autoSaveEnabled);
        console.log(`从localStorage加载自动存档设置: ${autoSaveEnabled ? '启用' : '禁用'}`);
    } catch (e) {
        console.warn('无法从localStorage加载自动存档设置:', e);
    }

    // 新增：设置描边参数
    state.setDefaultTextStrokeSettings(
        $('#enableTextStroke').is(':checked'),
        $('#textStrokeColor').val(),
        parseInt($('#textStrokeWidth').val()) || 1
    );
    state.setEnableTextStroke(state.defaultEnableTextStroke);
    state.setTextStrokeColor(state.defaultTextStrokeColor);
    state.setTextStrokeWidth(state.defaultTextStrokeWidth);

    // 初始化UI显示 (确保描边参数选项根据初始状态正确显示/隐藏)
    $("#textStrokeOptions").toggle(state.enableTextStroke); // 新增

    console.log("应用程序初始化完成。");
}

// --- 辅助函数 (从原始 script.js 迁移) ---

/**
 * 初始化漫画翻译提示词设置
 */
export function initializePromptSettings() { // 导出以便外部调用（如果需要）
    api.getPromptsApi()
        .then(response => {
            state.setPromptState(
                state.isTranslateJsonMode ? state.defaultTranslateJsonPrompt : response.default_prompt_content,
                response.default_prompt_content, // 普通默认
                response.prompt_names || [],
                state.defaultTranslateJsonPrompt // JSON默认
            );
            ui.updateTranslatePromptUI(); // 根据当前模式更新文本框和按钮
            ui.populatePromptDropdown(state.savedPromptNames, $('#promptDropdown'), $('#promptDropdownButton'), loadPromptContent, deletePrompt);
        })
        .catch(error => {
            console.error("获取提示词信息失败:", error);
            const errorMsg = "获取默认提示词失败";
            state.setPromptState(errorMsg, errorMsg, []);
            ui.updateTranslatePromptUI();
            ui.populatePromptDropdown([], $('#promptDropdown'), $('#promptDropdownButton'), loadPromptContent, deletePrompt);
        });
}

/**
 * 初始化文本框提示词设置
 */
export function initializeTextboxPromptSettings() { // 导出
    api.getTextboxPromptsApi()
        .then(response => {
            state.setTextboxPromptState(response.default_prompt_content, response.default_prompt_content, response.prompt_names || []);
            $('#textboxPromptContent').val(state.currentTextboxPromptContent);
            ui.populatePromptDropdown(state.savedTextboxPromptNames, $('#textboxPromptDropdown'), $('#textboxPromptDropdownButton'), loadTextboxPromptContent, deleteTextboxPrompt);
        })
        .catch(error => {
            console.error("获取文本框提示词信息失败:", error);
            const errorMsg = "获取默认文本框提示词失败";
            state.setTextboxPromptState(errorMsg, errorMsg, []);
            $('#textboxPromptContent').val(errorMsg);
            ui.populatePromptDropdown([], $('#textboxPromptDropdown'), $('#textboxPromptDropdownButton'), loadTextboxPromptContent, deleteTextboxPrompt);
        });
}

/**
 * 初始化AI视觉OCR提示词
 */
export function initializeAiVisionOcrPromptSettings() {
    // AI视觉OCR提示词目前是前端常量定义的，不需要从后端加载
    // 只需要确保 state 中的值正确，并在UI上正确显示
    state.setAiVisionOcrPromptMode(state.isAiVisionOcrJsonMode); // 这会根据模式设置正确的当前提示词
    ui.updateAiVisionOcrPromptUI();
}

/**
 * 加载指定名称的漫画翻译提示词内容
 * @param {string} promptName - 提示词名称
 */
function loadPromptContent(promptName) { // 私有辅助函数
    if (promptName === constants.DEFAULT_PROMPT_NAME) {
        // 根据当前模式加载对应的默认提示词
        const contentToLoad = state.isTranslateJsonMode ? state.defaultTranslateJsonPrompt : state.defaultPromptContent;
        // 修复：不要直接赋值，应该使用setter方法
        state.setPromptState(contentToLoad, state.defaultPromptContent, state.savedPromptNames, state.defaultTranslateJsonPrompt);
        ui.updateTranslatePromptUI(); // 更新UI
    } else {
        api.getPromptContentApi(promptName)
            .then(response => {
                // 修复：不要直接赋值，应该使用setter方法
                state.setPromptState(response.prompt_content, state.defaultPromptContent, state.savedPromptNames, state.defaultTranslateJsonPrompt);
                ui.updateTranslatePromptUI();
                // 尝试智能判断并切换模式 (可选的高级功能)
                if (response.prompt_content.includes('"translated_text":')) {
                    if (!state.isTranslateJsonMode) {
                        state.setTranslatePromptMode(true, response.prompt_content); // 切换到JSON模式并设置内容
                        ui.updateTranslatePromptUI();
                        ui.showGeneralMessage("检测到JSON格式提示词，已自动切换到JSON模式。", "info", false, 3000);
                    }
                } else {
                    if (state.isTranslateJsonMode) {
                        state.setTranslatePromptMode(false, response.prompt_content); // 切换到普通模式并设置内容
                        ui.updateTranslatePromptUI();
                        ui.showGeneralMessage("检测到普通格式提示词，已自动切换到普通模式。", "info", false, 3000);
                    }
                }
            })
            .catch(error => {
                console.error("加载提示词内容失败:", error);
                ui.showGeneralMessage(`加载提示词 "${promptName}" 失败: ${error.message}`, "error");
            });
    }
}

/**
 * 删除指定名称的漫画翻译提示词
 * @param {string} promptName - 提示词名称
 */
function deletePrompt(promptName) { // 私有辅助函数
    ui.showLoading("删除提示词...");
    api.deletePromptApi(promptName)
        .then(response => {
            ui.hideLoading();
            ui.showGeneralMessage(`提示词 "${promptName}" 删除成功！`, "success");
            initializePromptSettings(); // 重新加载列表
        })
        .catch(error => {
            ui.hideLoading();
            ui.showGeneralMessage(`删除提示词失败: ${error.message}`, "error");
        });
}

/**
 * 加载指定名称的文本框提示词内容
 * @param {string} promptName - 提示词名称
 */
function loadTextboxPromptContent(promptName) { // 私有辅助函数
    if (promptName === constants.DEFAULT_PROMPT_NAME) {
        state.setTextboxPromptState(state.defaultTextboxPromptContent, state.defaultTextboxPromptContent, state.savedTextboxPromptNames);
        $('#textboxPromptContent').val(state.currentTextboxPromptContent);
    } else {
        api.getTextboxPromptContentApi(promptName)
            .then(response => {
                state.setTextboxPromptState(response.prompt_content, state.defaultTextboxPromptContent, state.savedTextboxPromptNames);
                $('#textboxPromptContent').val(state.currentTextboxPromptContent);
            })
            .catch(error => {
                console.error("加载文本框提示词内容失败:", error);
                ui.showGeneralMessage(`加载文本框提示词 "${promptName}" 失败: ${error.message}`, "error");
            });
    }
}

/**
 * 删除指定名称的文本框提示词
 * @param {string} promptName - 提示词名称
 */
function deleteTextboxPrompt(promptName) { // 私有辅助函数
    ui.showLoading("删除文本框提示词...");
    api.deleteTextboxPromptApi(promptName)
        .then(response => {
            ui.hideLoading();
            ui.showGeneralMessage(`文本框提示词 "${promptName}" 删除成功！`, "success");
            initializeTextboxPromptSettings(); // 重新加载列表
        })
        .catch(error => {
            ui.hideLoading();
            ui.showGeneralMessage(`删除文本框提示词失败: ${error.message}`, "error");
        });
}


/**
 * 初始化可折叠面板
 */
function initializeCollapsiblePanels() { // 私有辅助函数
    const collapsibleHeaders = $(".collapsible-header");
    collapsibleHeaders.on("click", function() {
        const header = $(this);
        const content = header.next(".collapsible-content");
        header.toggleClass("collapsed");
        content.toggleClass("collapsed");
        const icon = header.find(".toggle-icon");
        icon.text(header.hasClass("collapsed") ? "▶" : "▼");
    });
    collapsibleHeaders.each(function(index) {
        if (index > 0) {
            const header = $(this);
            header.addClass("collapsed");
            header.next(".collapsible-content").addClass("collapsed");
            header.find(".toggle-icon").text("▶");
        }
    });
}

/**
 * 初始化亮暗模式切换
 */
function initializeThemeMode() { // 私有辅助函数
    const savedTheme = localStorage.getItem('themeMode');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
        document.body.classList.remove('light-mode');
    } else {
        document.body.classList.add('light-mode');
        document.body.classList.remove('dark-mode');
    }
    // 事件绑定在 events.js 中处理
}

/**
 * 检查初始模型提供商并更新 UI
 */
function checkInitialModelProvider() { // 私有辅助函数
    const selectedProvider = $('#modelProvider').val();
    console.log("初始化模型提供商:", selectedProvider);  // 添加日志
    
    ui.updateApiKeyInputState(selectedProvider === 'ollama' || selectedProvider === 'sakura',
                              selectedProvider === 'ollama' || selectedProvider === 'sakura' ? '本地部署无需API Key' : '请输入API Key');
    ui.toggleOllamaUI(selectedProvider === 'ollama');
    ui.toggleSakuraUI(selectedProvider === 'sakura');
    ui.toggleCaiyunUI(selectedProvider === 'caiyun');
    ui.toggleBaiduTranslateUI(selectedProvider === 'baidu_translate');
    ui.toggleYoudaoTranslateUI(selectedProvider === 'youdao_translate');
    
    if (selectedProvider === 'ollama') {
        console.log("正在获取Ollama模型列表...");  // 添加日志
        fetchOllamaModels();
    } else if (selectedProvider === 'sakura') {
        console.log("正在获取Sakura模型列表...");  // 添加日志
        fetchSakuraModels();
    } else if (selectedProvider === 'volcano') {
        // 获取火山引擎历史模型建议
        api.getUsedModelsApi('volcano')
            .then(response => ui.updateModelSuggestions(response.models))
            .catch(error => console.error("获取火山引擎模型建议失败:", error));
    }
}

/**
 * 获取 Ollama 模型列表并更新 UI
 */
export function fetchOllamaModels() { // 导出
    ui.showLoading("正在获取本地 Ollama 模型...");
    api.testOllamaConnectionApi()
        .then(response => {
            ui.hideLoading();
            ui.updateOllamaModelList(response.models);
            if (response.models && response.models.length > 0 && !$('#modelName').val()) {
                $('#modelName').val(response.models[0]);
                $('#ollamaModelsList .model-button').first().addClass('selected');
            }
        })
        .catch(error => {
            ui.hideLoading();
            ui.updateOllamaModelList([]);
            console.error("获取 Ollama 模型列表失败:", error);
            ui.showGeneralMessage(`获取 Ollama 模型列表失败: ${error.message}`, "error");
        });
}

/**
 * 获取 Sakura 模型列表并更新 UI
 */
export function fetchSakuraModels() { // 导出
    ui.showLoading("正在获取本地 Sakura 模型...");
    api.testSakuraConnectionApi()
        .then(response => {
            ui.hideLoading();
            ui.updateSakuraModelList(response.models);
            if (response.models && response.models.length > 0 && !$('#modelName').val()) {
                $('#modelName').val(response.models[0]);
                $('#sakuraModelsList .model-button').first().addClass('selected');
            }
        })
        .catch(error => {
            ui.hideLoading();
            ui.updateSakuraModelList([]);
            console.error("获取 Sakura 模型列表失败:", error);
            ui.showGeneralMessage(`获取 Sakura 模型列表失败: ${error.message}`, "error");
        });
}


/**
 * 处理文件（图片或 PDF）
 * @param {FileList} files - 用户选择或拖放的文件列表
 */
export function handleFiles(files) { // 导出
    if (!files || files.length === 0) return;

    ui.showLoading("处理文件中...");
    ui.hideError();

    const imagePromises = [];
    const pdfFiles = [];

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (file.type.startsWith('image/')) {
            imagePromises.push(processImageFile(file));
        } else if (file.type === 'application/pdf') {
            pdfFiles.push(file);
        } else {
            console.warn(`不支持的文件类型: ${file.name} (${file.type})`);
        }
    }

    Promise.all(imagePromises)
        .then(() => {
            if (pdfFiles.length > 0) {
                return processPDFFiles(pdfFiles);
            }
        })
        .then(() => {
            ui.hideLoading();
            if (state.images.length > 0) {
                sortImagesByName();
                ui.renderThumbnails();
                switchImage(0); // 显示第一张
            } else {
                ui.showError("未能成功加载任何图片。");
            }
            ui.updateButtonStates();
            session.triggerAutoSave(); // <--- 添加文件成功后触发自动存档
        })
        .catch(error => {
            ui.hideLoading();
            console.error("处理文件失败:", error);
            ui.showError(`处理文件时出错: ${error.message || error}`);
            ui.updateButtonStates();
        });
}

/**
 * 处理单个图片文件
 * @param {File} file - 图片文件
 * @returns {Promise<void>}
 */
function processImageFile(file) { // 私有
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            const originalDataURL = e.target.result;
            state.addImage({
                originalDataURL: originalDataURL,
                translatedDataURL: null, cleanImageData: null,
                bubbleTexts: [], bubbleCoords: [], originalTexts: [], textboxTexts: [],
                bubbleSettings: null, fileName: file.name,
                fontSize: state.defaultFontSize, autoFontSize: $('#autoFontSize').is(':checked'),
                fontFamily: state.defaultFontFamily, layoutDirection: state.defaultLayoutDirection,
                showOriginal: false, translationFailed: false,
                originalUseInpainting: undefined, originalUseLama: undefined,
            });
            resolve();
        };
        reader.onerror = (error) => reject(error);
        reader.readAsDataURL(file);
    });
}

/**
 * 处理 PDF 文件列表
 * @param {Array<File>} pdfFiles - PDF 文件数组
 * @returns {Promise<void>}
 */
function processPDFFiles(pdfFiles) { // 私有
    return pdfFiles.reduce((promiseChain, file) => {
        return promiseChain.then(() => {
            ui.showLoading(`处理 PDF: ${file.name}...`);
            const formData = new FormData();
            formData.append('pdfFile', file);
            return api.uploadPdfApi(formData)
                .then(response => {
                    if (response.images && response.images.length > 0) {
                        response.images.forEach((imageData, idx) => {
                            const originalDataURL = "data:image/png;base64," + imageData;
                            const pdfFileName = `${file.name}_页面${idx+1}`;
                            state.addImage({
                                originalDataURL: originalDataURL,
                                translatedDataURL: null, cleanImageData: null,
                                bubbleTexts: [], bubbleCoords: [], originalTexts: [], textboxTexts: [],
                                bubbleSettings: null, fileName: pdfFileName,
                                fontSize: state.defaultFontSize, autoFontSize: $('#autoFontSize').is(':checked'),
                                fontFamily: state.defaultFontFamily, layoutDirection: state.defaultLayoutDirection,
                                showOriginal: false, translationFailed: false,
                                originalUseInpainting: undefined, originalUseLama: undefined,
                            });
                        });
                    } else {
                        ui.showGeneralMessage(`PDF文件 ${file.name} 中没有检测到图片`, "warning");
                    }
                })
                .catch(error => {
                    console.error(`处理PDF文件 ${file.name} 失败:`, error);
                    ui.showGeneralMessage(`处理PDF文件 ${file.name} 失败: ${error.message}`, "error");
                });
        });
    }, Promise.resolve());
}


/**
 * 按文件名对图片状态数组进行排序
 */
function sortImagesByName() { // 私有
    state.images.sort((a, b) => {
        return a.fileName.localeCompare(b.fileName, undefined, { numeric: true, sensitivity: 'base' });
    });
}

/**
 * 切换显示的图片
 * @param {number} index - 要显示的图片索引
 */
export function switchImage(index) {
    if (index < 0 || index >= state.images.length) return;

    const wasInLabelingMode = state.isLabelingModeActive; // 记录切换前的模式

    // --- 退出当前模式 (如果需要) ---
    // 如果在编辑模式，保存当前图片的 bubbleSettings
    if (state.editModeActive) {
        const prevImage = state.getCurrentImage();
        if (prevImage) {
            prevImage.bubbleSettings = JSON.parse(JSON.stringify(state.bubbleSettings));
            prevImage.bubbleTexts = state.bubbleSettings.map(s => s.text);
        }
        editMode.exitEditMode(); // 退出编辑模式
    }
    // 如果在标注模式，检查未保存更改并决定是否退出或保存
    if (wasInLabelingMode) {
        // 这里不再自动退出标注模式，而是保持模式，只切换图片内容和标注框
        // 退出逻辑现在只在点击"退出标注模式"按钮时触发
        // 自动保存标注更改，不再提示用户
        const prevImage = state.getCurrentImage();
        if (state.hasUnsavedChanges && prevImage) {
            // 自动保存标注，不再询问
            if (!state.saveManualCoordsToImage()) {
                ui.showGeneralMessage("保存标注失败！", "error");
                // 不阻止切换，但提示保存失败
            } else {
                ui.showGeneralMessage("标注已保存。", "success", false, 2000);
                ui.renderThumbnails(); // 更新缩略图标记
            }
        } else {
            // 没有未保存更改，直接重置全局标记
            state.setHasUnsavedChanges(false);
        }
    }
    // ------------------------------

    // 设置新的当前索引
    state.setCurrentImageIndex(index);
    const imageData = state.getCurrentImage(); // 获取新图片数据
    console.log("切换到图片:", index, imageData.fileName);

    // --- 更新基础 UI ---
    ui.hideError();
    ui.hideLoading();
    $('#translatingMessage').hide();

    ui.updateTranslatedImage(imageData.showOriginal ? imageData.originalDataURL : (imageData.translatedDataURL || imageData.originalDataURL));
    $('#toggleImageButton').text(imageData.showOriginal ? '显示翻译图' : '显示原图');
    ui.updateImageSizeDisplay($('#imageSize').val());
    ui.showResultSection(true);
    ui.updateDetectedTextDisplay();
    ui.updateRetranslateButton();
    // --------------------

    // --- 加载新图片的设置到 UI ---
    if (!imageData.translatedDataURL && !imageData.originalDataURL) {
        // 此条件分支代表一个未加载任何图片的空位置，可以重置到全局默认值
        $('#fontSize').val(state.defaultFontSize);
        $('#fontFamily').val(state.defaultFontFamily);
        $('#layoutDirection').val(state.defaultLayoutDirection);
        $('#textColor').val(state.defaultTextColor);
        $('#fillColor').val(state.defaultFillColor);
        $('#rotationAngle').val(0);
        // 新增：重置描边UI为全局默认
        $('#enableTextStroke').prop('checked', state.defaultEnableTextStroke);
        $('#textStrokeColor').val(state.defaultTextStrokeColor);
        $('#textStrokeWidth').val(state.defaultTextStrokeWidth);
        $("#textStrokeOptions").toggle(state.defaultEnableTextStroke);
        
        // 更新全局状态变量
        state.setEnableTextStroke(state.defaultEnableTextStroke);
        state.setTextStrokeColor(state.defaultTextStrokeColor);
        state.setTextStrokeWidth(state.defaultTextStrokeWidth);
    } else if (!imageData.translatedDataURL && imageData.originalDataURL) {
        // 通常我们希望保留用户在翻译前的全局设置，或者也可以重置
        // 这里我们选择保留当前UI控件的值（通常是上一个图片的或全局默认的）
        // 但要确保 fillColor 反映图片自身的记录，如果没有，则用全局的
        $('#fillColor').val(imageData.fillColor || state.defaultFillColor);
        
        // 新增：如果图片有独立描边设置则加载，否则使用全局
        const enableStroke = imageData.enableTextStroke === undefined ? state.enableTextStroke : imageData.enableTextStroke;
        const strokeColor = imageData.textStrokeColor || state.textStrokeColor;
        const strokeWidth = imageData.textStrokeWidth === undefined ? state.textStrokeWidth : imageData.textStrokeWidth;
        
        $('#enableTextStroke').prop('checked', enableStroke);
        $('#textStrokeColor').val(strokeColor);
        $('#textStrokeWidth').val(strokeWidth);
        $("#textStrokeOptions").toggle(enableStroke);
        
        // 更新全局状态变量
        state.setEnableTextStroke(enableStroke);
        state.setTextStrokeColor(strokeColor);
        state.setTextStrokeWidth(strokeWidth);
    } else { // 图片已翻译或处理过
        // 加载图片自身记录的设置或回退到全局默认设置
        $('#fontSize').val(imageData.fontSize || state.defaultFontSize);
        $('#fontFamily').val(imageData.fontFamily || state.defaultFontFamily);
        $('#layoutDirection').val(imageData.textDirection || state.defaultLayoutDirection);
        $('#textColor').val(imageData.textColor || state.defaultTextColor);
        // 加载图片自身记录的填充色，如果不存在，则使用全局默认填充色
        $('#fillColor').val(imageData.fillColor || state.defaultFillColor);
        $('#rotationAngle').val(imageData.rotationAngle || 0);
        
        // 新增：加载图片的描边设置或全局设置
        const enableStroke = imageData.enableTextStroke === undefined ? state.enableTextStroke : imageData.enableTextStroke;
        const strokeColor = imageData.textStrokeColor || state.textStrokeColor;
        const strokeWidth = imageData.textStrokeWidth === undefined ? state.textStrokeWidth : imageData.textStrokeWidth;
        
        $('#enableTextStroke').prop('checked', enableStroke);
        $('#textStrokeColor').val(strokeColor);
        $('#textStrokeWidth').val(strokeWidth);
        $("#textStrokeOptions").toggle(enableStroke);
        
        // 更新全局状态变量
        state.setEnableTextStroke(enableStroke);
        state.setTextStrokeColor(strokeColor);
        state.setTextStrokeWidth(strokeWidth);
    }

    // 触发 change 以更新依赖 UI (比如修复选项的显隐)
    $('#useInpainting').trigger('change'); // 这个会调用 toggleInpaintingOptions
    $('#autoFontSize').trigger('change');

    ui.updateButtonStates();
    $('.thumbnail-item').removeClass('active');
    $(`.thumbnail-item[data-index="${index}"]`).addClass('active');
    ui.scrollToActiveThumbnail();

    // --- 处理模式状态 ---
    if (wasInLabelingMode) {
        // 如果是从标注模式切换来的，加载新图片的标注框（如果有）
        // 显式清除旧的标注框，确保不会残留
        ui.clearBoundingBoxes();
        labelingMode.loadBubbleCoordsForLabeling();
        ui.drawBoundingBoxes(state.manualBubbleCoords);
    }
    // ---------------------

    // 触发自动存档
    session.triggerAutoSave();
}

/**
 * 翻译当前图片
 */
export function translateCurrentImage() {
    const currentImage = state.getCurrentImage();
    if (!currentImage) return Promise.reject("No current image"); // 返回一个被拒绝的Promise

    // 检查是否处于标注模式，如果是，则不允许普通翻译
    if (state.isLabelingModeActive) {
        ui.showGeneralMessage("请先退出标注模式，或使用 '使用手动框翻译' 按钮。", "warning");
        return Promise.reject("Translation not allowed in labeling mode"); // 返回一个被拒绝的Promise
    }

    ui.showGeneralMessage("翻译中...", "info", false, 0);
    ui.showTranslatingIndicator(state.currentImageIndex);

    const repairSettings = ui.getRepairSettings();
    const isAutoFontSize = $('#autoFontSize').is(':checked');
    const fontSize = isAutoFontSize ? 'auto' : $('#fontSize').val();
    const ocr_engine = $('#ocrEngine').val();
    const modelProvider = $('#modelProvider').val(); // 获取当前选中的服务商

    // --- 关键修改：检查并使用已保存的手动坐标 ---
    let coordsToUse = null; // 默认不传递，让后端自动检测
    let usedManualCoords = false; // 标记是否使用了手动坐标
    if (currentImage.savedManualCoords && currentImage.savedManualCoords.length > 0) {
        coordsToUse = currentImage.savedManualCoords;
        usedManualCoords = true;
        console.log(`翻译当前图片 ${state.currentImageIndex}: 将使用 ${coordsToUse.length} 个已保存的手动标注框。`);
        ui.showGeneralMessage("检测到手动标注框，将优先使用...", "info", false, 3000);
    } else {
        console.log(`翻译当前图片 ${state.currentImageIndex}: 未找到手动标注框，将进行自动检测。`);
    }
    // ------------------------------------------

    const params = {
        image: currentImage.originalDataURL.split(',')[1],
        target_language: $('#targetLanguage').val(),
        source_language: $('#sourceLanguage').val(),
        fontSize: fontSize, autoFontSize: isAutoFontSize,
        api_key: $('#apiKey').val(),
        model_name: $('#modelName').val(),
        model_provider: modelProvider, // 使用获取到的服务商
        fontFamily: $('#fontFamily').val(),
        textDirection: $('#layoutDirection').val(),
        prompt_content: $('#promptContent').val(),
        use_textbox_prompt: $('#enableTextboxPrompt').prop('checked'),
        textbox_prompt_content: $('#textboxPromptContent').val(),
        use_inpainting: repairSettings.useInpainting,
        use_lama: repairSettings.useLama,
        blend_edges: $('#blendEdges').prop('checked'),
        inpainting_strength: parseFloat($('#inpaintingStrength').val()),
        fill_color: $('#fillColor').val(),
        text_color: $('#textColor').val(),
        rotation_angle: parseFloat($('#rotationAngle').val() || '0'),
        skip_ocr: false,
        skip_translation: false,
        bubble_coords: coordsToUse,
        ocr_engine: ocr_engine,
        baidu_ocr_api_key: ocr_engine === 'baidu' ? $('#baiduOcrApiKey').val() : null,
        baidu_ocr_secret_key: ocr_engine === 'baidu' ? $('#baiduOcrSecretKey').val() : null,
        baidu_ocr_version: ocr_engine === 'baidu' ? $('#baiduOcrVersion').val() : null,
        ai_vision_provider: ocr_engine === 'ai_vision' ? $('#aiVisionProvider').val() : null,
        ai_vision_api_key: ocr_engine === 'ai_vision' ? $('#aiVisionApiKey').val() : null,
        ai_vision_model_name: ocr_engine === 'ai_vision' ? $('#aiVisionModelName').val() : null,
        ai_vision_ocr_prompt: ocr_engine === 'ai_vision' ? $('#aiVisionOcrPrompt').val() : null,

        // === 新增描边参数 START ===
        enableTextStroke: state.enableTextStroke,       // 从 state.js 获取
        textStrokeColor: state.textStrokeColor,         // 从 state.js 获取
        textStrokeWidth: state.textStrokeWidth,         // 从 state.js 获取
        // === 新增描边参数 END ===
        
        rpm_limit_translation: state.rpmLimitTranslation,
        rpm_limit_ai_vision_ocr: state.rpmLimitAiVisionOcr,
        use_json_format_translation: state.isTranslateJsonMode,
        use_json_format_ai_vision_ocr: state.isAiVisionOcrJsonMode
    };

    // --- 新增：如果选择自定义服务商，添加 custom_base_url ---
    if (modelProvider === 'custom_openai') { // 使用常量会更好
        const customBaseUrl = $('#customBaseUrl').val().trim();
        if (!customBaseUrl) {
            ui.showGeneralMessage("自定义 OpenAI 服务需要填写 Base URL！", "error");
            ui.hideTranslatingIndicator(state.currentImageIndex);
            // 确保 updateButtonStates 会被调用以恢复按钮状态
            ui.updateButtonStates();
            return Promise.reject("Custom Base URL is required."); // 返回被拒绝的Promise
        }
        params.custom_base_url = customBaseUrl;
    }
    // ----------------------------------------------------

    // 检查百度OCR配置
    if (ocr_engine === 'baidu_ocr' && (!params.baidu_ocr_api_key || !params.baidu_ocr_secret_key)) {
        ui.showGeneralMessage("使用百度OCR时必须提供API Key和Secret Key", "error");
        ui.hideTranslatingIndicator(state.currentImageIndex);
        return Promise.reject("Baidu OCR configuration error"); // 返回一个被拒绝的Promise
    }
    
    // 检查AI视觉OCR配置
    if (ocr_engine === 'ai_vision' && (!params.ai_vision_api_key || !params.ai_vision_model_name)) {
        ui.showGeneralMessage("使用AI视觉OCR时必须提供API Key和模型名称", "error");
        ui.hideTranslatingIndicator(state.currentImageIndex);
        return Promise.reject("AI Vision OCR configuration error"); // 返回一个被拒绝的Promise
    }

    return new Promise((resolve, reject) => {
        api.translateImageApi(params)
            .then(response => {
                $(".message.info").fadeOut(300, function() { $(this).remove(); }); // 移除加载消息
                ui.hideTranslatingIndicator(state.currentImageIndex);

                // 更新当前图片状态
                state.updateCurrentImageProperty('translatedDataURL', 'data:image/png;base64,' + response.translated_image);
                state.updateCurrentImageProperty('cleanImageData', response.clean_image);
                state.updateCurrentImageProperty('bubbleTexts', response.bubble_texts);
                // **重要**: 更新 bubbleCoords 为本次使用的坐标 (无论是手动还是自动检测返回的)
                state.updateCurrentImageProperty('bubbleCoords', response.bubble_coords);
                state.updateCurrentImageProperty('originalTexts', response.original_texts);
                state.updateCurrentImageProperty('textboxTexts', response.textbox_texts);
                state.updateCurrentImageProperty('fontSize', fontSize);
                state.updateCurrentImageProperty('autoFontSize', isAutoFontSize);
                state.updateCurrentImageProperty('fontFamily', params.fontFamily);
                state.updateCurrentImageProperty('layoutDirection', params.textDirection);
                state.updateCurrentImageProperty('translationFailed', false);
                state.updateCurrentImageProperty('showOriginal', false);
                state.updateCurrentImageProperty('originalUseInpainting', repairSettings.useInpainting);
                state.updateCurrentImageProperty('originalUseLama', repairSettings.useLama);
                state.updateCurrentImageProperty('inpaintingStrength', repairSettings.inpaintingStrength);
                state.updateCurrentImageProperty('blendEdges', repairSettings.blendEdges);
                state.updateCurrentImageProperty('fillColor', params.fill_color);
                state.updateCurrentImageProperty('textColor', params.text_color);
                state.updateCurrentImageProperty('bubbleSettings', null); // 清空编辑设置

                // === 新增：保存描边参数到图片属性 START ===
                state.updateCurrentImageProperty('enableTextStroke', params.enableTextStroke);
                state.updateCurrentImageProperty('textStrokeColor', params.textStrokeColor);
                state.updateCurrentImageProperty('textStrokeWidth', params.textStrokeWidth);
                // === 新增：保存描边参数到图片属性 END ===

                // 根据使用的修复方法设置标记
                if (repairSettings.useLama) {
                    // 如果使用LAMA修复，添加标记
                    state.updateCurrentImageProperty('_lama_inpainted', true);
                    console.log("设置LAMA修复标记：_lama_inpainted=true");
                } else {
                    // 如果没有使用LAMA修复，确保清除可能存在的标记
                    state.updateCurrentImageProperty('_lama_inpainted', false);
                }

                // 如果使用了手动坐标，保留标注状态
                if (usedManualCoords) {
                    // 不再清除标注状态，确保可以继续使用手动标注框
                    // state.clearSavedManualCoords(); 
                    ui.renderThumbnails(); // 更新缩略图，保留标注状态
                }
                // --------------------------------------------

                switchImage(state.currentImageIndex); // 重新加载以更新所有 UI
                ui.updateDetectedTextDisplay();
                ui.updateRetranslateButton();
                ui.updateButtonStates();

                // 仅在非敏感服务商时保存模型信息
                if (params.model_provider !== 'baidu_translate' && params.model_provider !== 'youdao_translate') {
                    api.saveModelInfoApi(params.model_provider, params.model_name);
                }
                ui.showGeneralMessage("翻译成功！", "success");
                resolve(); // 解决Promise
            })
            .catch(error => {
                $(".message.info").fadeOut(300, function() { $(this).remove(); });
                ui.hideTranslatingIndicator(state.currentImageIndex);
                state.updateCurrentImageProperty('translationFailed', true);
                ui.renderThumbnails(); // 更新缩略图状态
                ui.showError(`翻译失败: ${error.message}`);
                ui.updateButtonStates();
                ui.updateRetranslateButton();
                reject(error); // 拒绝Promise
            });
    });
}

/**
 * 翻译所有已加载的图片
 * 按照每张图片的当前状态（包括手动标注框）批量翻译
 */
export function translateAllImages() {
    // 检查是否处于标注模式
    if (state.isLabelingModeActive) {
        ui.showGeneralMessage("请先退出标注模式再执行批量翻译。", "warning");
        return;
    }

    if (state.images.length === 0) {
        ui.showGeneralMessage("请先添加图片", "warning");
        return;
    }
    
    // 立即显示进度条（移到前面来）
    $("#translationProgressBar").show();
    ui.updateProgressBar(0, `0/${state.images.length}`);
    ui.showGeneralMessage("批量翻译中...", "info", false, 0); // 显示模态提示，不自动消失

    // 设置批量翻译状态为进行中
    state.setBatchTranslationInProgress(true);
    ui.updateButtonStates(); // 禁用按钮

    // --- 获取全局设置 (保持不变) ---
    const targetLanguage = $('#targetLanguage').val();
    const sourceLanguage = $('#sourceLanguage').val();
    const isAutoFontSize = $('#autoFontSize').is(':checked');
    const fontSize = isAutoFontSize ? 'auto' : $('#fontSize').val();
    const apiKey = $('#apiKey').val();
    const modelName = $('#modelName').val();
    const modelProvider = $('#modelProvider').val();
    const fontFamily = $('#fontFamily').val();
    const textDirection = $('#layoutDirection').val();
    const useTextboxPrompt = $('#enableTextboxPrompt').prop('checked');
    const textboxPromptContent = $('#textboxPromptContent').val();
    const fillColor = $('#fillColor').val();
    const repairSettings = ui.getRepairSettings(); // ui.js 获取修复设置
    const useInpainting = repairSettings.useInpainting;
    const useLama = repairSettings.useLama;
    const inpaintingStrength = parseFloat($('#inpaintingStrength').val());
    const blendEdges = $('#blendEdges').prop('checked');
    const promptContent = $('#promptContent').val();
    const textColor = $('#textColor').val();
    const rotationAngle = parseFloat($('#rotationAngle').val() || '0');
    const ocr_engine = $('#ocrEngine').val();
    
    // === 新增：获取全局描边参数 START ===
    const enableTextStroke = state.enableTextStroke;
    const textStrokeColor = state.textStrokeColor;
    const textStrokeWidth = state.textStrokeWidth;
    // === 新增：获取全局描边参数 END ===

    // 百度OCR相关参数
    const baiduApiKey = ocr_engine === 'baidu_ocr' ? $('#baiduApiKey').val() : null;
    const baiduSecretKey = ocr_engine === 'baidu_ocr' ? $('#baiduSecretKey').val() : null;
    const baiduVersion = ocr_engine === 'baidu_ocr' ? $('#baiduVersion').val() : 'standard';
    
    // AI视觉OCR相关参数
    const aiVisionProvider = ocr_engine === 'ai_vision' ? $('#aiVisionProvider').val() : null;
    const aiVisionApiKey = ocr_engine === 'ai_vision' ? $('#aiVisionApiKey').val() : null;
    const aiVisionModelName = ocr_engine === 'ai_vision' ? $('#aiVisionModelName').val() : null;
    const aiVisionOcrPrompt = ocr_engine === 'ai_vision' ? $('#aiVisionOcrPrompt').val() : null;

    // 检查百度OCR配置
    if (ocr_engine === 'baidu_ocr' && (!baiduApiKey || !baiduSecretKey)) {
        ui.showGeneralMessage("使用百度OCR时必须提供API Key和Secret Key", "error");
        state.setBatchTranslationInProgress(false); // 确保错误时重置状态
        $("#translationProgressBar").hide(); // 隐藏进度条
        ui.updateButtonStates(); // 恢复按钮状态
        return;
    }
    
    // 检查AI视觉OCR配置
    if (ocr_engine === 'ai_vision' && (!aiVisionApiKey || !aiVisionModelName)) {
        ui.showGeneralMessage("使用AI视觉OCR时必须提供API Key和模型名称", "error");
        state.setBatchTranslationInProgress(false); // 确保错误时重置状态
        $("#translationProgressBar").hide(); // 隐藏进度条
        ui.updateButtonStates(); // 恢复按钮状态
        return;
    }

    // 在循环外获取一次JSON模式状态
    const aktuellenTranslateJsonMode = state.isTranslateJsonMode;
    const aktuellenAiVisionOcrJsonMode = state.isAiVisionOcrJsonMode;

    let currentIndex = 0;
    const totalImages = state.images.length;
    let failCount = 0; // 记录失败数量

    let customBaseUrlForAll = null;
    if (modelProvider === 'custom_openai') {
        customBaseUrlForAll = $('#customBaseUrl').val().trim();
        if (!customBaseUrlForAll) {
            ui.showGeneralMessage("批量翻译自定义 OpenAI 服务需要填写 Base URL！", "error");
            state.setBatchTranslationInProgress(false); // 确保错误时重置状态
            $("#translationProgressBar").hide(); // 隐藏进度条
            ui.updateButtonStates(); // 更新按钮状态
            return; // 或返回 Promise.reject
        }
    }

    function processNextImage() {
        if (currentIndex >= totalImages) {
            ui.updateProgressBar(100, `${totalImages - failCount}/${totalImages}`);
            $(".message.info").fadeOut(300, function() { $(this).remove(); }); // 移除加载消息
            ui.updateButtonStates(); // 恢复按钮状态
            if (failCount > 0) {
                ui.showGeneralMessage(`批量翻译完成，成功 ${totalImages - failCount} 张，失败 ${failCount} 张。`, "warning");
            } else {
                ui.showGeneralMessage('所有图片翻译完成', "success");
            }
            // 批量完成后保存一次模型历史
            if(modelName && modelProvider) { // 确保有模型信息才保存
                 // 仅在非敏感服务商时保存模型信息
                 if (modelProvider !== 'baidu_translate' && modelProvider !== 'youdao_translate') {
                     api.saveModelInfoApi(modelProvider, modelName);
                 }
            }
            
            // 设置批量翻译状态为已完成
            state.setBatchTranslationInProgress(false);
            
            // 批量翻译完成后执行一次自动存档
            session.triggerAutoSave();
            return;
        }

        ui.updateProgressBar((currentIndex / totalImages) * 100, `${currentIndex}/${totalImages}`);
        ui.showTranslatingIndicator(currentIndex);
        const imageData = state.images[currentIndex]; // 获取当前循环索引对应的图片数据

        // --- 关键修改：检查并使用当前图片的已保存手动坐标 (逻辑保持不变) ---
        let coordsToUse = null;
        let usedManualCoordsThisImage = false;
        if (imageData.savedManualCoords && imageData.savedManualCoords.length > 0) {
            coordsToUse = imageData.savedManualCoords;
            usedManualCoordsThisImage = true;
            console.log(`批量翻译图片 ${currentIndex}: 将使用 ${coordsToUse.length} 个已保存的手动标注框。`);
        } else {
            console.log(`批量翻译图片 ${currentIndex}: 未找到手动标注框，将进行自动检测。`);
        }
        // ----------------------------------------------

        const data = { // 准备 API 请求数据
            image: imageData.originalDataURL.split(',')[1],
            target_language: targetLanguage, source_language: sourceLanguage,
            fontSize: fontSize, autoFontSize: isAutoFontSize,
            api_key: apiKey, model_name: modelName, model_provider: modelProvider,
            fontFamily: fontFamily, textDirection: textDirection,
            prompt_content: promptContent, use_textbox_prompt: useTextboxPrompt,
            textbox_prompt_content: textboxPromptContent, use_inpainting: useInpainting,
            use_lama: useLama, blend_edges: blendEdges, inpainting_strength: inpaintingStrength,
            fill_color: fillColor,
            text_color: textColor,
            rotation_angle: rotationAngle,
            skip_translation: false, skip_ocr: false, remove_only: false,
            bubble_coords: coordsToUse, // 传递坐标
            ocr_engine: ocr_engine,
            baidu_api_key: baiduApiKey,
            baidu_secret_key: baiduSecretKey,
            baidu_version: baiduVersion,
            ai_vision_provider: aiVisionProvider,
            ai_vision_api_key: aiVisionApiKey,
            ai_vision_model_name: aiVisionModelName,
            ai_vision_ocr_prompt: aiVisionOcrPrompt,
            custom_base_url: customBaseUrlForAll, // --- 传递 custom_base_url ---
            
            // === 新增：传递描边参数 START ===
            enableTextStroke: enableTextStroke,
            textStrokeColor: textStrokeColor,
            textStrokeWidth: textStrokeWidth,
            // === 新增：传递描边参数 END ===
            
            use_json_format_translation: aktuellenTranslateJsonMode,
            use_json_format_ai_vision_ocr: aktuellenAiVisionOcrJsonMode
        };

        // --- 核心修改：直接调用 API，而不是 translateCurrentImage ---
        api.translateImageApi(data)
            .then(response => {
                ui.hideTranslatingIndicator(currentIndex);

                // --- 更新特定索引的图片状态 ---
                // 使用 state.js 中的辅助函数或直接修改 state.images[currentIndex]
                state.updateImagePropertyByIndex(currentIndex, 'translatedDataURL', 'data:image/png;base64,' + response.translated_image);
                state.updateImagePropertyByIndex(currentIndex, 'cleanImageData', response.clean_image);
                state.updateImagePropertyByIndex(currentIndex, 'bubbleTexts', response.bubble_texts);
                state.updateImagePropertyByIndex(currentIndex, 'bubbleCoords', response.bubble_coords);
                state.updateImagePropertyByIndex(currentIndex, 'originalTexts', response.original_texts);
                state.updateImagePropertyByIndex(currentIndex, 'textboxTexts', response.textbox_texts);
                state.updateImagePropertyByIndex(currentIndex, 'translationFailed', false);
                state.updateImagePropertyByIndex(currentIndex, 'showOriginal', false);
                // 保存本次翻译使用的设置 (全局设置)
                state.updateImagePropertyByIndex(currentIndex, 'fontSize', fontSize);
                state.updateImagePropertyByIndex(currentIndex, 'autoFontSize', isAutoFontSize);
                state.updateImagePropertyByIndex(currentIndex, 'fontFamily', fontFamily);
                state.updateImagePropertyByIndex(currentIndex, 'layoutDirection', textDirection);
                state.updateImagePropertyByIndex(currentIndex, 'originalUseInpainting', useInpainting);
                state.updateImagePropertyByIndex(currentIndex, 'originalUseLama', useLama);
                state.updateImagePropertyByIndex(currentIndex, 'inpaintingStrength', inpaintingStrength);
                state.updateImagePropertyByIndex(currentIndex, 'blendEdges', blendEdges);
                state.updateImagePropertyByIndex(currentIndex, 'fillColor', fillColor);
                state.updateImagePropertyByIndex(currentIndex, 'textColor', textColor);
                state.updateImagePropertyByIndex(currentIndex, 'bubbleSettings', null); // 清空编辑设置
                
                // === 新增：保存描边参数到图片属性 START ===
                state.updateImagePropertyByIndex(currentIndex, 'enableTextStroke', enableTextStroke);
                state.updateImagePropertyByIndex(currentIndex, 'textStrokeColor', textStrokeColor);
                state.updateImagePropertyByIndex(currentIndex, 'textStrokeWidth', textStrokeWidth);
                // === 新增：保存描边参数到图片属性 END ===

                // 根据使用的修复方法设置标记
                if (useLama) {
                    // 如果使用LAMA修复，添加标记
                    state.updateImagePropertyByIndex(currentIndex, '_lama_inpainted', true);
                    console.log(`批量翻译图片 ${currentIndex}: 设置LAMA修复标记`);
                } else {
                    // 如果没有使用LAMA修复，确保清除可能存在的标记
                    state.updateImagePropertyByIndex(currentIndex, '_lama_inpainted', false);
                }

                // 如果使用了手动坐标，保留标注状态
                if (usedManualCoordsThisImage) {
                    // 不再清除手动标注坐标，以保持标注状态
                    // state.updateImagePropertyByIndex(currentIndex, 'savedManualCoords', null);
                    state.updateImagePropertyByIndex(currentIndex, 'hasUnsavedChanges', false); // 标记已处理
                    console.log(`批量翻译图片 ${currentIndex}: 保留手动坐标以便后续使用。`);
                }
                // -----------------------------------

                ui.renderThumbnails(); // 更新缩略图列表（会显示新翻译的图和标记）

                // *** 不再调用 switchImage ***

                // 成功完成一张
                ui.updateProgressBar(((currentIndex + 1) / totalImages) * 100, `${currentIndex + 1}/${totalImages}`);
            })
            .catch(error => {
                ui.hideTranslatingIndicator(currentIndex);
                console.error(`图片 ${currentIndex} (${imageData.fileName}) 翻译失败:`, error);
                failCount++;

                // --- 更新特定索引的图片状态为失败 ---
                state.updateImagePropertyByIndex(currentIndex, 'translationFailed', true);
                // -----------------------------------

                ui.renderThumbnails(); // 更新缩略图显示失败标记
            })
            .finally(() => {
                currentIndex++;
                processNextImage(); // 处理下一张图片
            });
        // --- 结束核心修改 ---
    }
    processNextImage(); // 开始处理第一张图片
}

// --- 其他需要导出的函数 ---
// (downloadCurrentImage, downloadAllImages, applySettingsToAll, removeBubbleTextOnly 等)
// 需要将它们的实现从 script.js 移到这里，并添加 export

/**
 * 下载当前图片（翻译后或原始图片）
 */
export function downloadCurrentImage() {
    const currentImage = state.getCurrentImage();
    if (!currentImage) {
        ui.showGeneralMessage("没有可下载的图片", "warning");
        return;
    }
    
    // 优先使用翻译后图片，如无则使用原始图片
    const imageDataURL = currentImage.translatedDataURL || currentImage.originalDataURL;
    
    if (imageDataURL) {
        ui.showDownloadingMessage(true);
        try {
            const base64Data = imageDataURL.split(',')[1];
            const byteCharacters = atob(base64Data);
            const byteArrays = [];
            for (let offset = 0; offset < byteCharacters.length; offset += 512) {
                const slice = byteCharacters.slice(offset, offset + 512);
                const byteNumbers = new Array(slice.length);
                for (let i = 0; i < slice.length; i++) byteNumbers[i] = slice.charCodeAt(i);
                byteArrays.push(new Uint8Array(byteNumbers));
            }
            const blob = new Blob(byteArrays, {type: 'image/png'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            let fileName = currentImage.fileName || `image_${state.currentImageIndex}.png`;
            // 为已翻译和未翻译的图片使用不同前缀
            const prefix = currentImage.translatedDataURL ? 'translated' : 'original';
            fileName = `${prefix}_${fileName.replace(/\.[^/.]+$/, "")}.png`;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            ui.showGeneralMessage(`下载成功: ${fileName}`, "success");
        } catch (e) {
            console.error('下载图片时出错:', e);
            ui.showGeneralMessage("下载失败", "error");
        }
        ui.showDownloadingMessage(false);
    } else {
        ui.showGeneralMessage("没有可下载的图片", "warning");
    }
}

/**
 * 下载所有翻译后的图片
 */
export function downloadAllImages() {
    const selectedFormat = $('#downloadFormat').val();

    // 立即显示进度条
    $("#translationProgressBar").show();
    ui.updateProgressBar(0, "准备下载...");

    // --- 显示提示信息 ---
    ui.showGeneralMessage("下载中...处理可能需要一定时间，请耐心等待...", "info", false, 0);
    ui.showDownloadingMessage(true); // 显示下载中并禁用按钮

    // 延迟执行下载，给 UI 更新时间
    setTimeout(() => {
        try {
            // 收集所有图片数据
            const allImages = state.images;
            if (allImages.length === 0) {
                ui.showGeneralMessage("没有可下载的图片", "warning");
                $(".message.info").fadeOut(300, function() { $(this).remove(); });
                $("#translationProgressBar").hide();
                ui.showDownloadingMessage(false);
                return;
            }
            
            // 收集需要发送的图像数据
            const imageDataList = [];
            let translatedCount = 0;
            let originalCount = 0;
            
            ui.updateProgressBar(10, "收集图片数据...");
            
            for (let i = 0; i < allImages.length; i++) {
                const imgData = allImages[i];
                // 优先使用翻译后的图片，如果没有则使用原始图片
                if (imgData.translatedDataURL) {
                    imageDataList.push(imgData.translatedDataURL);
                    translatedCount++;
                } else if (imgData.originalDataURL) {
                    imageDataList.push(imgData.originalDataURL);
                    originalCount++;
                }
                // 更新进度条
                ui.updateProgressBar(10 + (i + 1) / allImages.length * 20, `收集图片 ${i + 1}/${allImages.length}`);
            }
            
            if (imageDataList.length === 0) {
                ui.showGeneralMessage("没有可下载的图片", "warning");
                $("#translationProgressBar").hide();
                $(".message.info").fadeOut(300, function() { $(this).remove(); });
                ui.showDownloadingMessage(false);
                return;
            }
            
            // 更新进度条状态
            ui.updateProgressBar(30, "保存图片到临时文件夹...");
            
            // 调用后端API
            $.ajax({
                url: '/api/download_all_images',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    images: imageDataList,
                    format: selectedFormat
                }),
                success: function(response) {
                    ui.updateProgressBar(80, "处理完成，准备下载...");
                    
                    if (response.success && response.file_id) {
                        // 创建下载链接
                        const downloadUrl = `/api/download_file/${response.file_id}?format=${response.format}`;
                        
                        // 通过创建临时链接触发下载
                        const link = document.createElement('a');
                        link.href = downloadUrl;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                        
                        ui.updateProgressBar(100, "下载已开始");
                        
                        // 更新下载成功信息，包括翻译和原始图片数量
                        let successMessage = `已成功处理 ${imageDataList.length} 张图片`;
                        if (translatedCount > 0 && originalCount > 0) {
                            successMessage += `（${translatedCount} 张翻译图片和 ${originalCount} 张原始图片）`;
                        } else if (translatedCount > 0) {
                            successMessage += `（全部为翻译后图片）`;
                        } else if (originalCount > 0) {
                            successMessage += `（全部为原始图片）`;
                        }
                        successMessage += "，下载即将开始";
                        
                        ui.showGeneralMessage(successMessage, "success");
                        
                        // 启动后台清理过期文件的请求
                        setTimeout(() => {
                            $.ajax({
                                url: '/api/clean_temp_files',
                                type: 'POST',
                                contentType: 'application/json',
                                data: JSON.stringify({}),
                                success: function(cleanResponse) {
                                    console.log("临时文件清理结果:", cleanResponse);
                                },
                                error: function(xhr, status, error) {
                                    console.error("清理临时文件失败:", error);
                                }
                            });
                        }, 60000); // 1分钟后清理
                    } else {
                        ui.showGeneralMessage(`处理失败: ${response.error || '未知错误'}`, "error");
                        $("#translationProgressBar").hide();
                    }
                },
                error: function(xhr, status, error) {
                    ui.showGeneralMessage(`下载请求失败: ${xhr.responseJSON?.error || error}`, "error");
                    $("#translationProgressBar").hide();
                },
                complete: function() {
                    setTimeout(() => {
                    $("#translationProgressBar").hide();
                    $(".message.info").fadeOut(300, function() { $(this).remove(); });
                    }, 2000); // 延迟2秒再隐藏，让用户能看到完成进度
                    ui.showDownloadingMessage(false);
                }
            });
        } catch (e) {
            console.error("下载所有图片时出错:", e);
            ui.showGeneralMessage("下载失败", "error");
            $("#translationProgressBar").hide();
            $(".message.info").fadeOut(300, function() { $(this).remove(); });
            ui.showDownloadingMessage(false);
        }
    }, 100); // 短暂延迟
}

/**
 * 将当前字体设置应用到所有图片
 */
export async function applySettingsToAll() { // 导出
    const currentImage = state.getCurrentImage();
    if (!currentImage) {
        ui.showGeneralMessage("请先选择一张图片以应用其设置", "warning");
        return;
    }
    if (state.images.length <= 1) {
        ui.showGeneralMessage("只有一张图片，无需应用到全部", "info");
        return;
    }

    const settingsToApply = {
        fontSize: currentImage.fontSize,
        autoFontSize: currentImage.autoFontSize,
        fontFamily: currentImage.fontFamily,
        textDirection: currentImage.layoutDirection,
        textColor: $('#textColor').val(), // 使用当前的全局颜色
        rotationAngle: 0, // 全局应用时不应用旋转
        // === 新增：描边参数 START ===
        enableStroke: state.enableTextStroke,
        strokeColor: state.textStrokeColor,
        strokeWidth: state.textStrokeWidth
        // === 新增：描边参数 END ===
    };

    ui.showLoading("应用设置到所有图片...");
    
    // 保存当前图片索引，以便处理完后恢复
    const originalImageIndex = state.currentImageIndex;
    
    try {
        // 遍历所有图片
        for (let imageIndex = 0; imageIndex < state.images.length; imageIndex++) {
            const img = state.images[imageIndex];
            if (!img.translatedDataURL) continue; // 跳过未翻译的图片
            
            // 更新进度显示
            ui.updateLoadingMessage(`应用设置到图片 ${imageIndex + 1}/${state.images.length}...`);
            
            // 切换到当前图片
            await new Promise(resolve => {
                switchImage(imageIndex);
                setTimeout(resolve, 100); // 短暂延迟确保切换完成
            });
            
            // 应用设置
            img.fontSize = settingsToApply.fontSize;
            img.autoFontSize = settingsToApply.autoFontSize;
            img.fontFamily = settingsToApply.fontFamily;
            img.layoutDirection = settingsToApply.textDirection;
            
            // 更新 bubbleSettings (如果存在)
            if (img.bubbleSettings) {
                img.bubbleSettings = img.bubbleSettings.map(setting => ({
                    ...setting,
                    fontSize: settingsToApply.fontSize,
                    autoFontSize: settingsToApply.autoFontSize,
                    fontFamily: settingsToApply.fontFamily,
                    textDirection: settingsToApply.textDirection,
                    textColor: settingsToApply.textColor,
                    rotationAngle: settingsToApply.rotationAngle,
                    // === 新增：描边参数 START ===
                    enableStroke: settingsToApply.enableStroke,
                    strokeColor: settingsToApply.strokeColor, 
                    strokeWidth: settingsToApply.strokeWidth
                    // === 新增：描边参数 END ===
                }));
            }
            
            // 对当前图片进行重渲染
            await new Promise(resolve => {
                import('./edit_mode.js').then(editMode => {
                    editMode.reRenderFullImage().then(resolve);
                });
            });
            
            // 短暂停顿，以便用户可以看到进度
            await new Promise(resolve => setTimeout(resolve, 200));
        }
        
        // 全部处理完成后，回到最初的图片
        await new Promise(resolve => {
            switchImage(originalImageIndex);
            setTimeout(resolve, 100);
        });
        
        ui.hideLoading();
        ui.showGeneralMessage("设置已应用并重新渲染所有已翻译图片", "success");
    } catch (error) {
        console.error("应用设置到所有图片时出错:", error);
        ui.hideLoading();
        ui.showGeneralMessage("应用设置时发生错误: " + error.message, "error");
        
        // 出错时也回到最初的图片
        switchImage(originalImageIndex);
    }
}

/**
 * 仅消除当前图片的气泡文字
 */
export function removeBubbleTextOnly() {
    const currentImage = state.getCurrentImage();
    if (!currentImage) return Promise.reject("No current image"); // 返回Promise

    ui.showTranslatingIndicator(state.currentImageIndex);

    const repairSettings = ui.getRepairSettings();
    const isAutoFontSize = $('#autoFontSize').is(':checked');
    const fontSize = isAutoFontSize ? 'auto' : $('#fontSize').val();
    const ocr_engine = $('#ocrEngine').val();

    let coordsToUse = null;
    let usedManualCoords = false; // 添加变量定义
    // --- 关键修改：检查并使用已保存的手动坐标 ---
    if (currentImage.savedManualCoords && currentImage.savedManualCoords.length > 0) {
        coordsToUse = currentImage.savedManualCoords;
        usedManualCoords = true; // 设置为true，表示使用了手动坐标
        console.log(`消除文字 ${state.currentImageIndex}: 将使用 ${coordsToUse.length} 个已保存的手动标注框。`);
        ui.showGeneralMessage("检测到手动标注框，将优先使用...", "info", false, 3000);
    } else {
        console.log(`消除文字 ${state.currentImageIndex}: 未找到手动标注框，将进行自动检测。`);
    }
    // ------------------------------------------

    const params = {
        image: currentImage.originalDataURL.split(',')[1],
        target_language: $('#targetLanguage').val(),
        source_language: $('#sourceLanguage').val(),
        fontSize: fontSize, 
        autoFontSize: isAutoFontSize,
        api_key: $('#apiKey').val(), 
        model_name: $('#modelName').val(),
        model_provider: $('#modelProvider').val(),
        fontFamily: $('#fontFamily').val(),
        textDirection: $('#layoutDirection').val(),
        prompt_content: $('#promptContent').val(),
        use_textbox_prompt: $('#enableTextboxPrompt').prop('checked'),
        textbox_prompt_content: $('#textboxPromptContent').val(),
        use_inpainting: repairSettings.useInpainting,
        use_lama: repairSettings.useLama,
        blend_edges: $('#blendEdges').prop('checked'),
        inpainting_strength: parseFloat($('#inpaintingStrength').val()),
        fill_color: $('#fillColor').val(),
        text_color: $('#textColor').val(),
        rotation_angle: parseFloat($('#rotationAngle').val() || '0'),
        skip_ocr: false, 
        skip_translation: true, // 关键：跳过翻译步骤
        remove_only: true, // 添加此参数，表示仅消除文字
        bubble_coords: coordsToUse,
        ocr_engine: ocr_engine,
        baidu_ocr_api_key: ocr_engine === 'baidu' ? $('#baiduOcrApiKey').val() : null,
        baidu_ocr_secret_key: ocr_engine === 'baidu' ? $('#baiduOcrSecretKey').val() : null,
        baidu_ocr_version: ocr_engine === 'baidu' ? $('#baiduOcrVersion').val() : null,
        ai_vision_provider: ocr_engine === 'ai_vision' ? $('#aiVisionProvider').val() : null,
        ai_vision_api_key: ocr_engine === 'ai_vision' ? $('#aiVisionApiKey').val() : null,
        ai_vision_model_name: ocr_engine === 'ai_vision' ? $('#aiVisionModelName').val() : null,
        ai_vision_ocr_prompt: ocr_engine === 'ai_vision' ? $('#aiVisionOcrPrompt').val() : null,

        // === 新增描边参数 START ===
        enableTextStroke: state.enableTextStroke,
        textStrokeColor: state.textStrokeColor,
        textStrokeWidth: state.textStrokeWidth,
        // === 新增描边参数 END ===

        rpm_limit_translation: state.rpmLimitTranslation,
        rpm_limit_ai_vision_ocr: state.rpmLimitAiVisionOcr,
        use_json_format_translation: state.isTranslateJsonMode,
        use_json_format_ai_vision_ocr: state.isAiVisionOcrJsonMode
    };

    // 检查百度OCR配置
    if (ocr_engine === 'baidu_ocr' && (!params.baidu_ocr_api_key || !params.baidu_ocr_secret_key)) {
        ui.showGeneralMessage("使用百度OCR时必须提供API Key和Secret Key", "error");
        ui.hideTranslatingIndicator(state.currentImageIndex);
        return Promise.reject("Baidu OCR configuration error"); // 返回一个被拒绝的Promise
    }
    
    // 检查AI视觉OCR配置
    if (ocr_engine === 'ai_vision' && (!params.ai_vision_api_key || !params.ai_vision_model_name)) {
        ui.showGeneralMessage("使用AI视觉OCR时必须提供API Key和模型名称", "error");
        ui.hideTranslatingIndicator(state.currentImageIndex);
        return Promise.reject("AI Vision OCR configuration error"); // 返回一个被拒绝的Promise
    }

    return new Promise((resolve, reject) => {
        api.translateImageApi(params)
            .then(response => {
                ui.hideTranslatingIndicator(state.currentImageIndex);

                // 更新当前图片对象
                currentImage.translatedDataURL = 'data:image/png;base64,' + response.translated_image;
                currentImage.cleanImageData = response.clean_image;
                
                // 确保bubbleTexts和bubbleCoords长度匹配
                let bubbleTexts = response.bubble_texts || [];
                const bubbleCoords = response.bubble_coords || [];
                
                // 修复可能的文本与坐标长度不匹配问题
                if (bubbleTexts.length !== bubbleCoords.length) {
                    console.warn(`消除文字: 文本数量(${bubbleTexts.length})与坐标数量(${bubbleCoords.length})不匹配，正在修复...`);
                    
                    if (bubbleCoords.length > 0) {
                        // 调整文本数组以匹配坐标数组长度
                        if (bubbleTexts.length < bubbleCoords.length) {
                            // 文本不足，用空字符串填充
                            while (bubbleTexts.length < bubbleCoords.length) {
                                bubbleTexts.push("");
                            }
                            console.log("已填充缺失的文本条目");
                        } else {
                            // 文本过多，截断
                            bubbleTexts = bubbleTexts.slice(0, bubbleCoords.length);
                            console.log("已截断多余的文本条目");
                        }
                    }
                }
                
                currentImage.bubbleTexts = bubbleTexts;
                currentImage.bubbleCoords = bubbleCoords;
                currentImage.originalTexts = response.original_texts || [];
                currentImage.textboxTexts = response.textbox_texts || [];
                currentImage.fontSize = fontSize;
                currentImage.autoFontSize = isAutoFontSize;
                currentImage.fontFamily = $('#fontFamily').val();
                currentImage.layoutDirection = $('#layoutDirection').val();
                currentImage.showOriginal = false;
                // 移除之前的翻译失败标记（如果有）
                currentImage.translationFailed = false;

                currentImage.originalUseInpainting = params.use_inpainting;
                currentImage.originalUseLama = params.use_lama;
                currentImage.inpaintingStrength = params.inpainting_strength;
                currentImage.blendEdges = params.blend_edges;
                currentImage.fillColor = params.fill_color;
                currentImage.textColor = params.text_color;
                
                // 添加LAMA修复标记，当use_lama为true时
                if (params.use_lama) {
                    currentImage._lama_inpainted = true;
                    console.log("仅消除文字模式：标记图片为LAMA修复");
                }
                
                // --- 如果使用了手动坐标，保留标注状态 ---
                if (usedManualCoords) {
                    // 不再清除标注状态，确保可以继续使用手动标注框
                    // state.clearSavedManualCoords(); 
                    ui.renderThumbnails(); // 更新缩略图，保留标注状态
                }
                // --------------------------------------------
                
                // --- 关键修改：初始化气泡设置 ---
                // 使用已经声明的bubbleCoords和bubbleTexts变量，不再重新声明
                if (bubbleCoords.length > 0) {
                    const newBubbleSettings = [];
                    for (let i = 0; i < bubbleCoords.length; i++) {
                        newBubbleSettings.push({
                            text: bubbleTexts[i] || "",
                            fontSize: fontSize,
                            autoFontSize: isAutoFontSize,
                            fontFamily: $('#fontFamily').val(),
                            textDirection: $('#layoutDirection').val(),
                            position: { x: 0, y: 0 },
                            textColor: params.text_color,
                            rotationAngle: 0
                        });
                    }
                    currentImage.bubbleSettings = newBubbleSettings;
                    state.setBubbleSettings(newBubbleSettings);
                } else {
                    currentImage.bubbleSettings = null;
                    state.setBubbleSettings([]);
                }

                switchImage(state.currentImageIndex); // 重新加载以更新所有 UI
                ui.showGeneralMessage("文字已消除", "success");
                ui.updateButtonStates();
                ui.updateRetranslateButton();
                ui.updateDetectedTextDisplay(); // 显示检测到的文本
                resolve(); // 解决Promise
            })
            .catch(error => {
                ui.hideTranslatingIndicator(state.currentImageIndex);
                ui.showGeneralMessage(`操作失败: ${error.message}`, "error");
                ui.updateButtonStates();
                reject(error); // 拒绝Promise
            });
    });
}

/**
 * 检查当前图片是否有翻译失败的句子
 * @returns {boolean}
 */
export function checkForFailedTranslations() { // 导出
    const currentImage = state.getCurrentImage();
    if (!currentImage) return false;
    const bubbleTexts = currentImage.bubbleTexts || [];
    const textboxTexts = currentImage.textboxTexts || [];
    const checkList = state.useTextboxPrompt ? textboxTexts : bubbleTexts;
    return checkList.some(text => text && text.includes("翻译失败"));
}

/**
 * 初始化OCR引擎设置
 */
function initializeOcrEngineSettings() {
    // 触发OCR引擎变更事件以设置初始UI状态
    const selectedEngine = $('#ocrEngine').val();
    
    // 根据选择显示/隐藏OCR设置区域
    $('#baiduOcrOptions').hide();
    $('#aiVisionOcrOptions').hide();
    
    if (selectedEngine === 'baidu_ocr') {
        $('#baiduOcrOptions').show();
    } else if (selectedEngine === 'ai_vision') {
        $('#aiVisionOcrOptions').show();
    }
    
    // 设置初始状态的样式
    $("#baiduOcrOptions, #aiVisionOcrOptions").css({
        'margin-bottom': '15px',
        'padding': '10px',
        'border-radius': '8px',
        'background-color': 'rgba(0,0,0,0.02)'
    });
    
    console.log(`初始化OCR引擎设置: 当前选择 ${selectedEngine}`);
}

/**
 * 消除所有图片中的文字
 * 按照每张图片的当前状态（包括手动标注框）批量消除文字
 */
export function removeAllBubblesText() {
    // 检查是否处于标注模式
    if (state.isLabelingModeActive) {
        ui.showGeneralMessage("请先退出标注模式再执行批量消除文字。", "warning");
        return;
    }

    if (state.images.length === 0) {
        ui.showGeneralMessage("请先添加图片", "warning");
        return;
    }
    
    // 立即显示进度条和全局提示
    $("#translationProgressBar").show();
    ui.updateProgressBar(0, '0/' + state.images.length);
    ui.showGeneralMessage("正在批量消除文字...", "info", false);

    // 设置批量翻译状态为进行中
    state.setBatchTranslationInProgress(true);
    ui.updateButtonStates(); // 禁用按钮

    // --- 获取全局设置 (保持不变) ---
    const targetLanguage = $('#targetLanguage').val();
    const sourceLanguage = $('#sourceLanguage').val();
    const isAutoFontSize = $('#autoFontSize').is(':checked');
    const fontSize = isAutoFontSize ? 'auto' : $('#fontSize').val();
    const fontFamily = $('#fontFamily').val();
    const textDirection = $('#layoutDirection').val();
    const fillColor = $('#fillColor').val();
    const repairSettings = ui.getRepairSettings(); // ui.js 获取修复设置
    const useInpainting = repairSettings.useInpainting;
    const useLama = repairSettings.useLama;
    const inpaintingStrength = parseFloat($('#inpaintingStrength').val());
    const blendEdges = $('#blendEdges').prop('checked');
    const textColor = $('#textColor').val();
    const rotationAngle = parseFloat($('#rotationAngle').val() || '0');
    const ocr_engine = $('#ocrEngine').val();

    // 百度OCR相关参数
    const baiduApiKey = ocr_engine === 'baidu_ocr' ? $('#baiduApiKey').val() : null;
    const baiduSecretKey = ocr_engine === 'baidu_ocr' ? $('#baiduSecretKey').val() : null;
    const baiduVersion = ocr_engine === 'baidu_ocr' ? $('#baiduVersion').val() : 'standard';
    
    // AI视觉OCR相关参数
    const aiVisionProvider = ocr_engine === 'ai_vision' ? $('#aiVisionProvider').val() : null;
    const aiVisionApiKey = ocr_engine === 'ai_vision' ? $('#aiVisionApiKey').val() : null;
    const aiVisionModelName = ocr_engine === 'ai_vision' ? $('#aiVisionModelName').val() : null;
    const aiVisionOcrPrompt = ocr_engine === 'ai_vision' ? $('#aiVisionOcrPrompt').val() : null;
    const aktuellenAiVisionOcrJsonMode = state.isAiVisionOcrJsonMode; // 当前设置的JSON模式
    
    // 检查必要的OCR参数
    if (ocr_engine === 'baidu_ocr' && (!baiduApiKey || !baiduSecretKey)) {
        ui.showGeneralMessage("使用百度OCR时必须提供API Key和Secret Key", "error");
        state.setBatchTranslationInProgress(false);
        $("#translationProgressBar").hide(); // 隐藏进度条
        ui.updateButtonStates();
        return;
    }
    
    if (ocr_engine === 'ai_vision' && (!aiVisionApiKey || !aiVisionModelName)) {
        ui.showGeneralMessage("使用AI视觉OCR时必须提供API Key和模型名称", "error");
        state.setBatchTranslationInProgress(false);
        $("#translationProgressBar").hide(); // 隐藏进度条
        ui.updateButtonStates();
        return;
    }

    let currentIndex = 0;
    const totalImages = state.images.length;
    let failCount = 0;
    let customBaseUrlForAll = null;

    function processNextImage() {
        if (currentIndex >= totalImages) {
            ui.updateProgressBar(100, `${totalImages - failCount}/${totalImages}`);
            $(".message.info").fadeOut(300, function() { $(this).remove(); }); // 移除加载消息
            ui.updateButtonStates(); // 恢复按钮状态
            if (failCount > 0) {
                ui.showGeneralMessage(`批量消除文字完成，成功 ${totalImages - failCount} 张，失败 ${failCount} 张。`, "warning");
            } else {
                ui.showGeneralMessage('所有图片文字消除完成', "success");
            }
            
            // 设置批量翻译状态为已完成
            state.setBatchTranslationInProgress(false);
            
            // 批量消除完成后执行一次自动存档
            session.triggerAutoSave();
            return;
        }

        ui.updateProgressBar((currentIndex / totalImages) * 100, `${currentIndex}/${totalImages}`);
        ui.showTranslatingIndicator(currentIndex);
        const imageData = state.images[currentIndex];

        // --- 检查并使用当前图片的已保存手动坐标 ---
        let coordsToUse = null;
        let usedManualCoordsThisImage = false;
        if (imageData.savedManualCoords && imageData.savedManualCoords.length > 0) {
            coordsToUse = imageData.savedManualCoords;
            usedManualCoordsThisImage = true;
            console.log(`批量消除文字 ${currentIndex}: 将使用 ${coordsToUse.length} 个已保存的手动标注框。`);
        } else {
            console.log(`批量消除文字 ${currentIndex}: 未找到手动标注框，将进行自动检测。`);
        }
        // ----------------------------------------------

        const data = { // 准备 API 请求数据
            image: imageData.originalDataURL.split(',')[1],
            target_language: targetLanguage, 
            source_language: sourceLanguage,
            fontSize: fontSize, 
            autoFontSize: isAutoFontSize,
            api_key: '', // 消除文字不需要API Key
            model_name: '',
            model_provider: '',
            fontFamily: fontFamily, 
            textDirection: textDirection,
            prompt_content: '',
            use_textbox_prompt: false,
            textbox_prompt_content: '',
            use_inpainting: useInpainting,
            use_lama: useLama, 
            blend_edges: blendEdges, 
            inpainting_strength: inpaintingStrength,
            fill_color: fillColor,
            text_color: textColor,
            rotation_angle: rotationAngle,
            skip_translation: true, // 关键设置：跳过翻译
            skip_ocr: false,
            remove_only: true, // 关键设置：仅消除文字
            bubble_coords: coordsToUse, // 传递坐标
            ocr_engine: ocr_engine,
            baidu_api_key: baiduApiKey,
            baidu_secret_key: baiduSecretKey,
            baidu_version: baiduVersion,
            ai_vision_provider: aiVisionProvider,
            ai_vision_api_key: aiVisionApiKey,
            ai_vision_model_name: aiVisionModelName,
            ai_vision_ocr_prompt: aiVisionOcrPrompt,
            use_json_format_translation: false,
            use_json_format_ai_vision_ocr: aktuellenAiVisionOcrJsonMode
        };

        // --- 调用API执行文字消除 ---
        api.translateImageApi(data)
            .then(response => {
                ui.hideTranslatingIndicator(currentIndex);

                // --- 更新特定索引的图片状态 ---
                state.updateImagePropertyByIndex(currentIndex, 'translatedDataURL', 'data:image/png;base64,' + response.translated_image);
                state.updateImagePropertyByIndex(currentIndex, 'cleanImageData', response.clean_image);
                
                // 确保bubbleTexts和bubbleCoords长度匹配
                let bubbleTexts = response.bubble_texts || [];
                const bubbleCoords = response.bubble_coords || [];
                
                // 修复可能的文本与坐标长度不匹配问题
                if (bubbleTexts.length !== bubbleCoords.length) {
                    console.warn(`批量消除文字 ${currentIndex}: 文本数量(${bubbleTexts.length})与坐标数量(${bubbleCoords.length})不匹配，正在修复...`);
                    
                    if (bubbleCoords.length > 0) {
                        // 调整文本数组以匹配坐标数组长度
                        if (bubbleTexts.length < bubbleCoords.length) {
                            // 文本不足，用空字符串填充
                            while (bubbleTexts.length < bubbleCoords.length) {
                                bubbleTexts.push("");
                            }
                            console.log(`批量消除文字 ${currentIndex}: 已填充缺失的文本条目`);
                        } else {
                            // 文本过多，截断
                            bubbleTexts = bubbleTexts.slice(0, bubbleCoords.length);
                            console.log(`批量消除文字 ${currentIndex}: 已截断多余的文本条目`);
                        }
                    }
                }
                
                state.updateImagePropertyByIndex(currentIndex, 'bubbleTexts', bubbleTexts);
                state.updateImagePropertyByIndex(currentIndex, 'bubbleCoords', bubbleCoords);
                state.updateImagePropertyByIndex(currentIndex, 'originalTexts', response.original_texts || []);
                state.updateImagePropertyByIndex(currentIndex, 'textboxTexts', response.textbox_texts || []);
                state.updateImagePropertyByIndex(currentIndex, 'translationFailed', false);
                state.updateImagePropertyByIndex(currentIndex, 'showOriginal', false);
                // 保存本次处理使用的设置
                state.updateImagePropertyByIndex(currentIndex, 'fontSize', fontSize);
                state.updateImagePropertyByIndex(currentIndex, 'autoFontSize', isAutoFontSize);
                state.updateImagePropertyByIndex(currentIndex, 'fontFamily', fontFamily);
                state.updateImagePropertyByIndex(currentIndex, 'layoutDirection', textDirection);
                state.updateImagePropertyByIndex(currentIndex, 'originalUseInpainting', useInpainting);
                state.updateImagePropertyByIndex(currentIndex, 'originalUseLama', useLama);
                state.updateImagePropertyByIndex(currentIndex, 'inpaintingStrength', inpaintingStrength);
                state.updateImagePropertyByIndex(currentIndex, 'blendEdges', blendEdges);
                state.updateImagePropertyByIndex(currentIndex, 'fillColor', fillColor);
                state.updateImagePropertyByIndex(currentIndex, 'textColor', textColor);
                state.updateImagePropertyByIndex(currentIndex, 'bubbleSettings', null); // 清空编辑设置

                // 根据使用的修复方法设置标记
                if (useLama) {
                    state.updateImagePropertyByIndex(currentIndex, '_lama_inpainted', true);
                    console.log(`批量消除文字 ${currentIndex}: 设置LAMA修复标记`);
                } else {
                    state.updateImagePropertyByIndex(currentIndex, '_lama_inpainted', false);
                }

                // 如果使用了手动坐标，保留标注状态
                if (usedManualCoordsThisImage) {
                    state.updateImagePropertyByIndex(currentIndex, 'hasUnsavedChanges', false); // 标记已处理
                    console.log(`批量消除文字 ${currentIndex}: 保留手动坐标以便后续使用。`);
                }

                ui.renderThumbnails(); // 更新缩略图列表

                // 成功完成一张
                ui.updateProgressBar(((currentIndex + 1) / totalImages) * 100, `${currentIndex + 1}/${totalImages}`);
            })
            .catch(error => {
                ui.hideTranslatingIndicator(currentIndex);
                console.error(`图片 ${currentIndex} (${imageData.fileName}) 消除文字失败:`, error);
                failCount++;

                // --- 更新特定索引的图片状态为失败 ---
                state.updateImagePropertyByIndex(currentIndex, 'translationFailed', true);
                // -----------------------------------

                ui.renderThumbnails(); // 更新缩略图显示失败标记
            })
            .finally(() => {
                currentIndex++;
                processNextImage(); // 处理下一张图片
            });
    }
    
    processNextImage(); // 开始处理第一张图片
}

/**
 * 导出所有图片的文本（原文和译文）为JSON文件
 */
export function exportText() {
    const allImages = state.images;
    if (allImages.length === 0) {
        ui.showGeneralMessage("没有可导出的图片文本", "warning");
        return;
    }

    // 准备导出数据
    const exportData = [];
    
    // 遍历所有图片
    for (let imageIndex = 0; imageIndex < allImages.length; imageIndex++) {
        const image = allImages[imageIndex];
        const bubbleTexts = image.bubbleTexts || [];
        const originalTexts = image.originalTexts || [];
        
        // 构建该图片的文本数据
        const imageTextData = {
            imageIndex: imageIndex,
            bubbles: []
        };
        
        // 构建每个气泡的文本数据
        for (let bubbleIndex = 0; bubbleIndex < originalTexts.length; bubbleIndex++) {
            const original = originalTexts[bubbleIndex] || '';
            const translated = bubbleTexts[bubbleIndex] || '';
            
            // 获取气泡的排版方向
            let textDirection = 'vertical'; // 默认为竖排
            
            // 如果图片有特定的气泡设置，从中获取排版方向
            if (image.bubbleSettings && Array.isArray(image.bubbleSettings) && 
                bubbleIndex < image.bubbleSettings.length && 
                image.bubbleSettings[bubbleIndex]) {
                textDirection = image.bubbleSettings[bubbleIndex].textDirection || textDirection;
            } else if (image.layoutDirection) {
                // 如果没有特定气泡设置，使用图片整体布局方向
                textDirection = image.layoutDirection;
            }
            
            imageTextData.bubbles.push({
                bubbleIndex: bubbleIndex,
                original: original,
                translated: translated,
                textDirection: textDirection  // 添加排版方向信息
            });
        }
        
        exportData.push(imageTextData);
    }
    
    // 将数据转换为JSON字符串
    const jsonData = JSON.stringify(exportData, null, 2);
    
    // 创建Blob并触发下载
    const blob = new Blob([jsonData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'comic_text_export.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    ui.showGeneralMessage("文本导出成功！", "success");
}

/**
 * 导入JSON文本文件并应用到当前图片集
 * @param {File} jsonFile - 用户选择的JSON文件
 */
export function importText(jsonFile) {
    if (!jsonFile) {
        ui.showGeneralMessage("未选择文件", "warning");
        return;
    }
    
    // 立即显示进度条
    $("#translationProgressBar").show();
    ui.updateProgressBar(0, "准备导入文本...");
    ui.showGeneralMessage("正在导入文本...", "info", false);
    
    const reader = new FileReader();
    
    reader.onload = async function(e) {
        try {
            // 解析JSON数据
            ui.updateProgressBar(10, "解析JSON数据...");
            const importedData = JSON.parse(e.target.result);
            
            // 验证数据格式
            if (!Array.isArray(importedData)) {
                ui.updateProgressBar(100, "导入失败");
                $("#translationProgressBar").hide();
                throw new Error("导入的JSON格式不正确，应为数组");
            }
            
            // 统计信息
            let updatedImages = 0;
            let updatedBubbles = 0;
            
            // 保存当前图片索引，以便导入完成后返回
            const originalImageIndex = state.currentImageIndex;
            
            // 获取当前左侧边栏的设置值
            const currentFontSize = $('#autoFontSize').is(':checked') ? 'auto' : parseInt($('#fontSize').val());
            const currentAutoFontSize = $('#autoFontSize').is(':checked');
            const currentFontFamily = $('#fontFamily').val();
            const currentTextColor = $('#textColor').val();
            const currentFillColor = $('#fillColor').val();
            const currentRotationAngle = parseFloat($('#rotationAngle').val() || '0');
            
            ui.updateProgressBar(20, "开始更新图片...");
            
            // 遍历导入的数据
            const totalImages = importedData.length;
            let processedImages = 0;
            
            for (const imageData of importedData) {
                processedImages++;
                const progress = 20 + (processedImages / totalImages * 70); // 从20%到90%
                ui.updateProgressBar(progress, `处理图片 ${processedImages}/${totalImages}`);
                
                const imageIndex = imageData.imageIndex;
                
                // 检查图片索引是否有效
                if (imageIndex < 0 || imageIndex >= state.images.length) {
                    console.warn(`跳过无效的图片索引: ${imageIndex}`);
                    continue;
                }
                
                const image = state.images[imageIndex];
                let imageUpdated = false;
                
                // 确保必要的数组存在
                if (!image.bubbleTexts) image.bubbleTexts = [];
                if (!image.originalTexts) image.originalTexts = [];
                
                // 遍历气泡数据
                for (const bubbleData of imageData.bubbles) {
                    const bubbleIndex = bubbleData.bubbleIndex;
                    
                    // 检查气泡索引是否有效
                    if (bubbleIndex < 0 || bubbleIndex >= image.originalTexts.length) {
                        console.warn(`跳过无效的气泡索引: 图片${imageIndex}，气泡${bubbleIndex}`);
                        continue;
                    }
                    
                    // 确保气泡数组长度足够
                    while (image.bubbleTexts.length <= bubbleIndex) {
                        image.bubbleTexts.push('');
                    }
                    
                    // 更新气泡译文
                    image.bubbleTexts[bubbleIndex] = bubbleData.translated;
                    
                    // 从JSON中获取排版方向信息，其余参数使用左侧边栏的当前设置
                    const textDirection = bubbleData.textDirection || $('#layoutDirection').val();
                    
                    // 如果图片没有bubbleSettings或长度不匹配，则初始化它
                    if (!image.bubbleSettings || 
                        !Array.isArray(image.bubbleSettings) || 
                        image.bubbleSettings.length !== image.bubbleCoords.length) {
                        // 创建新的气泡设置，使用当前左侧边栏的设置
                        const newSettings = [];
                        for (let i = 0; i < image.bubbleCoords.length; i++) {
                            const bubbleTextDirection = (i === bubbleIndex && textDirection) ? textDirection : $('#layoutDirection').val();
                            newSettings.push({
                                // 使用更新后的bubbleTexts数组的值，确保每个气泡都有正确的文本
                                text: image.bubbleTexts[i] || "",
                                fontSize: currentFontSize,
                                autoFontSize: currentAutoFontSize,
                                fontFamily: currentFontFamily,
                                textDirection: bubbleTextDirection,
                                position: { x: 0, y: 0 },
                                textColor: currentTextColor,
                                rotationAngle: currentRotationAngle,
                                fillColor: currentFillColor
                            });
                        }
                        image.bubbleSettings = newSettings;
                    } else {
                        // 更新对应气泡的所有设置，保留从JSON中导入的排版方向，其余使用左侧边栏的当前设置
                        if (image.bubbleSettings[bubbleIndex]) {
                            image.bubbleSettings[bubbleIndex] = {
                                ...image.bubbleSettings[bubbleIndex],
                                text: bubbleData.translated,
                                fontSize: currentFontSize,
                                autoFontSize: currentAutoFontSize,
                                fontFamily: currentFontFamily,
                                textDirection: textDirection, // 使用从JSON导入的排版方向
                                textColor: currentTextColor,
                                rotationAngle: currentRotationAngle,
                                fillColor: currentFillColor
                            };
                        }
                    }
                    
                    updatedBubbles++;
                    imageUpdated = true;
                }
                
                // 确保图片的bubbleSettings中的text属性与bubbleTexts一致
                if (imageUpdated && image.bubbleSettings && image.bubbleTexts) {
                    for (let i = 0; i < image.bubbleSettings.length; i++) {
                        if (image.bubbleSettings[i] && i < image.bubbleTexts.length) {
                            image.bubbleSettings[i].text = image.bubbleTexts[i] || "";
                        }
                    }
                }
                
                if (imageUpdated) {
                    updatedImages++;
                    
                    // 切换到当前更新的图片并重新渲染它
                    await new Promise(resolve => {
                        // 切换到当前处理的图片
                        switchImage(imageIndex);
                        
                        // 如果图片已经有翻译数据，重新渲染它
                        if (image.translatedDataURL) {
                            // 重新渲染当前图片
                            if (state.editModeActive) {
                                // 如果在编辑模式，使用编辑模式的方式重新渲染
                                import('./edit_mode.js').then(editMode => {
                                    editMode.reRenderFullImage().then(resolve);
                                });
                            } else {
                                // 否则也需要重新渲染来显示新文本
                                import('./edit_mode.js').then(editMode => {
                                    editMode.reRenderFullImage().then(resolve);
                                });
                            }
                        } else {
                            // 如果图片没有被翻译过，只切换到该图片但无需渲染
                            resolve();
                        }
                    });
                    
                    // 在导入并渲染每张图片后短暂停顿，以便用户可以看到进度
                    await new Promise(resolve => setTimeout(resolve, 300));
                }
            }
            
            ui.updateProgressBar(90, "完成图片更新，恢复原始视图...");
            
            // 全部导入完成后，回到最初的图片
            switchImage(originalImageIndex);
            
            // 更新UI
            ui.renderThumbnails();
            
            // 显示导入结果
            ui.updateProgressBar(100, "导入完成");
            ui.showGeneralMessage(`导入成功！更新了${updatedImages}张图片中的${updatedBubbles}个气泡文本`, "success");
            
            // 触发自动保存
            session.triggerAutoSave();
            
            // 延时隐藏进度条
            setTimeout(() => {
                $("#translationProgressBar").hide();
            }, 2000);
            
        } catch (error) {
            console.error("导入文本出错:", error);
            ui.showGeneralMessage(`导入失败: ${error.message}`, "error");
            $("#translationProgressBar").hide();
        }
    };
    
    reader.onerror = function() {
        ui.showGeneralMessage("读取文件时出错", "error");
        $("#translationProgressBar").hide();
    };
    
    reader.readAsText(jsonFile);
}

// --- 应用启动 ---
$(document).ready(initializeApp);
