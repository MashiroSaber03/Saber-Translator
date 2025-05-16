// src/app/static/js/main.js

// 引入所有需要的模块
import * as state from './state.js';
import * as ui from './ui.js';
import * as api from './api.js';
import * as events from './events.js';
import * as editMode from './edit_mode.js';
import * as constants from './constants.js'; // 导入前端常量
<<<<<<< HEAD
import * as labelingMode from './labeling_mode.js';
import * as session from './session.js'; // 导入session模块，用于自动存档
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
    const usesLamaInpainting = currentImage.hasOwnProperty('_lama_inpainted') && currentImage._lama_inpainted === true;
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
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
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

=======
// import $ from 'jquery'; // 假设 jQuery 已全局加载

>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
// --- 初始化函数 ---

/**
 * 初始化应用状态和 UI
 */
function initializeApp() {
    console.log("初始化应用程序...");

    // 1. 从 DOM 读取初始/默认设置并更新状态
    state.setDefaultFontSize(parseInt($('#fontSize').val()) || 25);
    state.setDefaultFontFamily($('#fontFamily').val() || 'fonts/STSONG.TTF'); // 确保与 HTML 默认值一致
    state.setDefaultLayoutDirection($('#layoutDirection').val() || 'vertical');
    state.setDefaultTextColor($('#textColor').val() || '#000000');
    state.setDefaultFillColor($('#fillColor').val() || constants.DEFAULT_FILL_COLOR);
    state.setUseTextboxPrompt($('#enableTextboxPrompt').is(':checked'));

    // 2. 初始化提示词设置 (调用 API)
    initializePromptSettings();
    initializeTextboxPromptSettings();
<<<<<<< HEAD
    initializeAiVisionOcrPromptSettings();

    // --- 初始化 RPD 状态 (从 state.js 的默认值开始) ---
    // state.js 中 rpdLimitTranslation 和 rpdLimitAiVisionOcr 已经用 constants 初始化了
    // 如果之前实现了从 localStorage 加载，那会在这里执行
    // const savedRpdTranslation = localStorage.getItem('rpdLimitTranslation');
    // state.setRpdLimitTranslation(savedRpdTranslation !== null ? parseInt(savedRpdTranslation) : constants.DEFAULT_RPD_TRANSLATION);
    // const savedRpdAiVision = localStorage.getItem('rpdLimitAiVisionOcr');
    // state.setRpdLimitAiVisionOcr(savedRpdAiVision !== null ? parseInt(savedRpdAiVision) : constants.DEFAULT_RPD_AI_VISION_OCR);
    
    // --- 更新UI输入框以反映初始/加载的RPD状态 ---
    ui.updateRpdInputFields(); // <--- 新增调用
    // ---------------------------------------------
=======
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915

    // 3. 初始化可折叠面板
    initializeCollapsiblePanels();

    // 4. 初始化亮暗模式
    initializeThemeMode();

    // 5. 检查初始模型提供商并更新 UI
    checkInitialModelProvider();
<<<<<<< HEAD
    
    // 6. 初始化OCR引擎设置
    initializeOcrEngineSettings();

    // --- 新增：设置 AI Vision OCR 默认提示词 ---
    // 直接使用常量设置 textarea 的值
    // 最好通过 ui.js 来操作 DOM
    ui.setAiVisionOcrPrompt(constants.DEFAULT_AI_VISION_OCR_PROMPT);
    // 确保 state.js 中的状态也与此默认值一致（已在 state.js 中完成）
    // ------------------------------------------

    // 7. 绑定所有事件监听器
    events.bindEventListeners();

    // 8. 更新初始按钮状态
    ui.updateButtonStates();

    // 9. 初始化修复选项的显示状态
=======

    // 6. 绑定所有事件监听器
    events.bindEventListeners();

    // 7. 更新初始按钮状态
    ui.updateButtonStates();

    // 8. 初始化修复选项的显示状态
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
    const initialRepairMethod = $('#useInpainting').val();
    ui.toggleInpaintingOptions(
        initialRepairMethod === 'true' || initialRepairMethod === 'lama',
        initialRepairMethod === 'false'
    );

<<<<<<< HEAD
    // 10. 初始化 UI 显示
    ui.updateTranslatePromptUI(); // 更新漫画翻译提示词UI
    ui.updateAiVisionOcrPromptUI(); // 更新AI视觉OCR提示词UI

=======
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
    console.log("应用程序初始化完成。");
}

// --- 辅助函数 (从原始 script.js 迁移) ---

/**
 * 初始化漫画翻译提示词设置
 */
export function initializePromptSettings() { // 导出以便外部调用（如果需要）
    api.getPromptsApi()
        .then(response => {
<<<<<<< HEAD
            state.setPromptState(
                state.isTranslateJsonMode ? state.defaultTranslateJsonPrompt : response.default_prompt_content,
                response.default_prompt_content, // 普通默认
                response.prompt_names || [],
                state.defaultTranslateJsonPrompt // JSON默认
            );
            ui.updateTranslatePromptUI(); // 根据当前模式更新文本框和按钮
=======
            state.setPromptState(response.default_prompt_content, response.default_prompt_content, response.prompt_names || []);
            $('#promptContent').val(state.currentPromptContent);
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
            ui.populatePromptDropdown(state.savedPromptNames, $('#promptDropdown'), $('#promptDropdownButton'), loadPromptContent, deletePrompt);
        })
        .catch(error => {
            console.error("获取提示词信息失败:", error);
            const errorMsg = "获取默认提示词失败";
            state.setPromptState(errorMsg, errorMsg, []);
<<<<<<< HEAD
            ui.updateTranslatePromptUI();
=======
            $('#promptContent').val(errorMsg);
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
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
<<<<<<< HEAD
 * 初始化AI视觉OCR提示词
 */
export function initializeAiVisionOcrPromptSettings() {
    // AI视觉OCR提示词目前是前端常量定义的，不需要从后端加载
    // 只需要确保 state 中的值正确，并在UI上正确显示
    state.setAiVisionOcrPromptMode(state.isAiVisionOcrJsonMode); // 这会根据模式设置正确的当前提示词
    ui.updateAiVisionOcrPromptUI();
}

/**
=======
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
 * 加载指定名称的漫画翻译提示词内容
 * @param {string} promptName - 提示词名称
 */
function loadPromptContent(promptName) { // 私有辅助函数
    if (promptName === constants.DEFAULT_PROMPT_NAME) {
<<<<<<< HEAD
        // 根据当前模式加载对应的默认提示词
        const contentToLoad = state.isTranslateJsonMode ? state.defaultTranslateJsonPrompt : state.defaultPromptContent;
        state.currentPromptContent = contentToLoad; // 直接更新当前内容
        ui.updateTranslatePromptUI(); // 更新UI
    } else {
        api.getPromptContentApi(promptName)
            .then(response => {
                state.currentPromptContent = response.prompt_content;
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
=======
        state.setPromptState(state.defaultPromptContent, state.defaultPromptContent, state.savedPromptNames);
        $('#promptContent').val(state.currentPromptContent);
    } else {
        api.getPromptContentApi(promptName)
            .then(response => {
                state.setPromptState(response.prompt_content, state.defaultPromptContent, state.savedPromptNames);
                $('#promptContent').val(state.currentPromptContent);
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
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
<<<<<<< HEAD
    console.log("初始化模型提供商:", selectedProvider);  // 添加日志
    
=======
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
    ui.updateApiKeyInputState(selectedProvider === 'ollama' || selectedProvider === 'sakura',
                              selectedProvider === 'ollama' || selectedProvider === 'sakura' ? '本地部署无需API Key' : '请输入API Key');
    ui.toggleOllamaUI(selectedProvider === 'ollama');
    ui.toggleSakuraUI(selectedProvider === 'sakura');
<<<<<<< HEAD
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
=======
    if (selectedProvider === 'ollama') {
        fetchOllamaModels();
    } else if (selectedProvider === 'sakura') {
        fetchSakuraModels();
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
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
<<<<<<< HEAD
            session.triggerAutoSave(); // <--- 添加文件成功后触发自动存档
=======
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
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
<<<<<<< HEAD
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
        // 但我们需要处理未保存的更改
        const prevImage = state.getCurrentImage();
        if (state.hasUnsavedChanges && prevImage) {
            if (confirm(`切换图片前，是否保存对 '${prevImage.fileName}' 的标注更改？\n(选择"确定"保存，选择"取消"放弃更改)`)) {
                 if (!state.saveManualCoordsToImage()) {
                     ui.showGeneralMessage("保存标注失败！", "error");
                     // 不阻止切换，但提示保存失败
                 } else {
                     ui.showGeneralMessage("标注已保存。", "success", false, 2000);
                     ui.renderThumbnails(); // 更新缩略图标记
                 }
            } else {
                 console.log("用户选择放弃标注更改。");
                 // 重置该图片未保存状态
                 prevImage.hasUnsavedChanges = false;
                 // 清除全局未保存状态，因为要加载新图片了
                 state.setHasUnsavedChanges(false);
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
=======
export function switchImage(index) { // 导出
    if (index < 0 || index >= state.images.length) return;

    if (state.editModeActive) {
        const currentImage = state.getCurrentImage();
        if (currentImage) {
            currentImage.bubbleSettings = JSON.parse(JSON.stringify(state.bubbleSettings));
            currentImage.bubbleTexts = state.bubbleSettings.map(s => s.text);
        }
    }

    state.setCurrentImageIndex(index);
    const imageData = state.getCurrentImage();
    console.log("切换到图片:", index, imageData.fileName);

>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
    ui.hideError();
    ui.hideLoading();
    $('#translatingMessage').hide();

    ui.updateTranslatedImage(imageData.showOriginal ? imageData.originalDataURL : (imageData.translatedDataURL || imageData.originalDataURL));
    $('#toggleImageButton').text(imageData.showOriginal ? '显示翻译图' : '显示原图');
    ui.updateImageSizeDisplay($('#imageSize').val());
    ui.showResultSection(true);
    ui.updateDetectedTextDisplay();
    ui.updateRetranslateButton();
<<<<<<< HEAD
    // --------------------

    // --- 加载新图片的设置到 UI ---
    if (!imageData.translatedDataURL && !imageData.originalDataURL) { // 如果是完全空的图片对象
        // 重置所有UI为全局默认值
        $('#autoFontSize').prop('checked', false); // 假设默认不自动
        $('#fontSize').prop('disabled', false).val(state.defaultFontSize);
        $('#fontFamily').val(state.defaultFontFamily);
        $('#layoutDirection').val(state.defaultLayoutDirection);
        $('#textColor').val(state.defaultTextColor);
        $('#rotationAngle').val(0);
        $('#rotationAngleValue').text('0°');
        $('#useInpainting').val('false'); // 默认纯色填充
        $('#fillColor').val(state.defaultFillColor); // <--- 使用全局默认填充色
        // ... 其他修复参数也重置 ...
    } else if (!imageData.translatedDataURL && imageData.originalDataURL) { // 有原图但未翻译
        // 通常我们希望保留用户在翻译前的全局设置，或者也可以重置
        // 这里我们选择保留当前UI控件的值（通常是上一个图片的或全局默认的）
        // 但要确保 fillColor 反映图片自身的记录，如果没有，则用全局的
        $('#fillColor').val(imageData.fillColor || state.defaultFillColor);
    }
    else { // 图片已翻译或处理过
        // 图片已翻译，加载其保存的设置到 UI
        $('#autoFontSize').prop('checked', imageData.autoFontSize || false);
        $('#fontSize').prop('disabled', imageData.autoFontSize || false).val(imageData.autoFontSize ? '-' : (imageData.fontSize || state.defaultFontSize));
        $('#fontFamily').val(imageData.fontFamily || state.defaultFontFamily);
        $('#layoutDirection').val(imageData.layoutDirection || state.defaultLayoutDirection); 
        $('#textColor').val(imageData.textColor || state.defaultTextColor);
        $('#rotationAngle').val(imageData.rotationAngle || 0);
        $('#rotationAngleValue').text((imageData.rotationAngle || 0) + '°');

        // 加载修复设置
        const useInpainting = imageData.originalUseInpainting;
        const useLama = imageData.originalUseLama;
        let repairMethod = 'false';
        if (useLama) repairMethod = 'lama';
        else if (useInpainting) repairMethod = 'true';
        $('#useInpainting').val(repairMethod);
        // ui.toggleInpaintingOptions(useInpainting || useLama, !useInpainting && !useLama); // 旧的，下面会触发change

        if(useInpainting || useLama) {
            $('#inpaintingStrength').val(imageData.inpaintingStrength === undefined ? constants.DEFAULT_INPAINTING_STRENGTH : imageData.inpaintingStrength);
            $('#inpaintingStrengthValue').text($('#inpaintingStrength').val());
            $('#blendEdges').prop('checked', imageData.blendEdges === undefined ? true : imageData.blendEdges);
        }
        // 加载图片自身记录的填充色，如果不存在，则使用全局默认填充色
        $('#fillColor').val(imageData.fillColor || state.defaultFillColor); // <--- 修改：加载图片自身的填充色
    }

    // 触发 change 以更新依赖 UI (比如修复选项的显隐)
    $('#useInpainting').trigger('change'); // 这个会调用 toggleInpaintingOptions
    $('#autoFontSize').trigger('change');
=======

    // 更新全局设置控件
    $('#autoFontSize').prop('checked', imageData.autoFontSize);
    $('#fontSize').prop('disabled', imageData.autoFontSize).val(imageData.autoFontSize ? '-' : imageData.fontSize);
    $('#fontFamily').val(imageData.fontFamily);
    $('#layoutDirection').val(imageData.layoutDirection);
    const useInpainting = imageData.originalUseInpainting;
    const useLama = imageData.originalUseLama;
    let repairMethod = 'false';
    if (useLama) repairMethod = 'lama';
    else if (useInpainting) repairMethod = 'true';
    $('#useInpainting').val(repairMethod);
    ui.toggleInpaintingOptions(useInpainting || useLama, !useInpainting && !useLama);
    if(useInpainting || useLama) {
        $('#inpaintingStrength').val(imageData.inpaintingStrength || constants.DEFAULT_INPAINTING_STRENGTH);
        $('#inpaintingStrengthValue').text($('#inpaintingStrength').val());
        $('#blendEdges').prop('checked', imageData.blendEdges !== undefined ? imageData.blendEdges : true);
    }
     $('#fillColor').val(imageData.fillColor || constants.DEFAULT_FILL_COLOR);

>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915

    ui.updateButtonStates();
    $('.thumbnail-item').removeClass('active');
    $(`.thumbnail-item[data-index="${index}"]`).addClass('active');
    ui.scrollToActiveThumbnail();

<<<<<<< HEAD
    // --- 处理模式状态 ---
    if (wasInLabelingMode) {
        // 如果是从标注模式切换来的，加载新图片的标注框（如果有）
        labelingMode.loadBubbleCoordsForLabeling();
        labelingMode.drawBoundingBoxes();
    }
    // ---------------------

    // 触发自动存档
    session.triggerAutoSave();
=======
    if (state.editModeActive) {
        editMode.initBubbleSettings();
        if (state.bubbleSettings.length > 0) {
            editMode.selectBubble(0);
        } else {
            ui.updateBubbleEditArea(-1);
            ui.updateBubbleHighlight(-1);
        }
    } else {
        editMode.exitEditMode();
    }
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
}

/**
 * 翻译当前图片
 */
<<<<<<< HEAD
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
=======
export function translateCurrentImage() { // 导出
    const currentImage = state.getCurrentImage();
    if (!currentImage) return;

    // 移除重复的showLoading调用，因为events.js已经显示了消息
    ui.showTranslatingIndicator(state.currentImageIndex);

    const repairSettings = ui.getRepairSettings(); // 需要在 ui.js 中导出
    const isAutoFontSize = $('#autoFontSize').is(':checked');
    const fontSize = isAutoFontSize ? 'auto' : $('#fontSize').val();
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915

    const params = {
        image: currentImage.originalDataURL.split(',')[1],
        target_language: $('#targetLanguage').val(),
        source_language: $('#sourceLanguage').val(),
        fontSize: fontSize,
        autoFontSize: isAutoFontSize,
        api_key: $('#apiKey').val(),
        model_name: $('#modelName').val(),
<<<<<<< HEAD
        model_provider: modelProvider, // 使用获取到的服务商
=======
        model_provider: $('#modelProvider').val(),
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
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
<<<<<<< HEAD
        skip_translation: false,
        skip_ocr: false,
        remove_only: false,
        bubble_coords: coordsToUse,
        ocr_engine: ocr_engine,
        baidu_api_key: ocr_engine === 'baidu_ocr' ? $('#baiduApiKey').val() : null,
        baidu_secret_key: ocr_engine === 'baidu_ocr' ? $('#baiduSecretKey').val() : null,
        baidu_version: ocr_engine === 'baidu_ocr' ? $('#baiduVersion').val() : 'standard',
        ai_vision_provider: ocr_engine === 'ai_vision' ? $('#aiVisionProvider').val() : null,
        ai_vision_api_key: ocr_engine === 'ai_vision' ? $('#aiVisionApiKey').val() : null,
        ai_vision_model_name: ocr_engine === 'ai_vision' ? $('#aiVisionModelName').val() : null,
        ai_vision_ocr_prompt: ocr_engine === 'ai_vision' ? $('#aiVisionOcrPrompt').val() : null,
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
    if (ocr_engine === 'baidu_ocr' && (!params.baidu_api_key || !params.baidu_secret_key)) {
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

    // 设置批量翻译状态为进行中
    state.setBatchTranslationInProgress(true);

    // --- 获取全局设置 (保持不变) ---
=======
        skip_translation: false, skip_ocr: false, remove_only: false,
    };

    api.translateImageApi(params)
        .then(response => {
            ui.hideTranslatingIndicator(state.currentImageIndex);

            state.updateCurrentImageProperty('translatedDataURL', 'data:image/png;base64,' + response.translated_image);
            state.updateCurrentImageProperty('cleanImageData', response.clean_image);
            state.updateCurrentImageProperty('bubbleTexts', response.bubble_texts);
            state.updateCurrentImageProperty('bubbleCoords', response.bubble_coords);
            state.updateCurrentImageProperty('originalTexts', response.original_texts);
            state.updateCurrentImageProperty('textboxTexts', response.textbox_texts);
            state.updateCurrentImageProperty('fontSize', fontSize);
            state.updateCurrentImageProperty('autoFontSize', isAutoFontSize);
            state.updateCurrentImageProperty('fontFamily', params.fontFamily);
            state.updateCurrentImageProperty('layoutDirection', params.textDirection);
            state.updateCurrentImageProperty('showOriginal', false);
            state.updateCurrentImageProperty('translationFailed', false);
            state.updateCurrentImageProperty('originalUseInpainting', params.use_inpainting);
            state.updateCurrentImageProperty('originalUseLama', params.use_lama);
            state.updateCurrentImageProperty('inpaintingStrength', params.inpainting_strength);
            state.updateCurrentImageProperty('blendEdges', params.blend_edges);
            state.updateCurrentImageProperty('fillColor', params.fill_color);
            state.updateCurrentImageProperty('textColor', params.text_color);
            state.updateCurrentImageProperty('bubbleSettings', null);
            state.setBubbleSettings([]);

            switchImage(state.currentImageIndex); // 重新加载以更新所有 UI
            ui.updateDetectedTextDisplay();
            ui.updateRetranslateButton();
            ui.updateButtonStates();

            api.saveModelInfoApi(params.model_provider, params.model_name);
        })
        .catch(error => {
            ui.hideTranslatingIndicator(state.currentImageIndex);
            state.updateCurrentImageProperty('translationFailed', true);
            ui.renderThumbnails();
            ui.showError(`翻译失败: ${error.message}`);
            ui.updateButtonStates();
            ui.updateRetranslateButton();
        });
}

/**
 * 翻译所有图片
 */
export function translateAllImages() { // 导出
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
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
<<<<<<< HEAD
    const repairSettings = ui.getRepairSettings(); // ui.js 获取修复设置
=======
    const repairSettings = ui.getRepairSettings();
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
    const useInpainting = repairSettings.useInpainting;
    const useLama = repairSettings.useLama;
    const inpaintingStrength = parseFloat($('#inpaintingStrength').val());
    const blendEdges = $('#blendEdges').prop('checked');
    const promptContent = $('#promptContent').val();
<<<<<<< HEAD
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

    // 检查百度OCR配置
    if (ocr_engine === 'baidu_ocr' && (!baiduApiKey || !baiduSecretKey)) {
        ui.showGeneralMessage("使用百度OCR时必须提供API Key和Secret Key", "error");
        state.setBatchTranslationInProgress(false); // 确保错误时重置状态
        return;
    }
    
    // 检查AI视觉OCR配置
    if (ocr_engine === 'ai_vision' && (!aiVisionApiKey || !aiVisionModelName)) {
        ui.showGeneralMessage("使用AI视觉OCR时必须提供API Key和模型名称", "error");
        state.setBatchTranslationInProgress(false); // 确保错误时重置状态
        return;
    }

    // 在循环外获取一次JSON模式状态
    const aktuellenTranslateJsonMode = state.isTranslateJsonMode;
    const aktuellenAiVisionOcrJsonMode = state.isAiVisionOcrJsonMode;

    let currentIndex = 0;
    const totalImages = state.images.length;
    let failCount = 0; // 记录失败数量
    ui.updateProgressBar(0, `0/${totalImages}`);
    ui.showGeneralMessage("批量翻译中...", "info", false, 0); // 显示模态提示，不自动消失
    ui.updateButtonStates(); // 禁用按钮

    let customBaseUrlForAll = null;
    if (modelProvider === 'custom_openai') {
        customBaseUrlForAll = $('#customBaseUrl').val().trim();
        if (!customBaseUrlForAll) {
            ui.showGeneralMessage("批量翻译自定义 OpenAI 服务需要填写 Base URL！", "error");
            state.setBatchTranslationInProgress(false); // 确保错误时重置状态
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
=======

    // 参数检查... (省略，假设与 translateCurrentImage 类似)

    let currentIndex = 0;
    const totalImages = state.images.length;
    ui.updateProgressBar(0, `0/${totalImages}`);
    ui.showGeneralMessage("批量翻译中...", "info", false, 0);

    function processNextImage() {
        if (currentIndex >= totalImages) {
            ui.updateProgressBar(100, `${totalImages}/${totalImages}`);
            $(".message.info").fadeOut(300, function() { $(this).remove(); });
            ui.updateButtonStates();
            ui.showGeneralMessage('所有图片翻译完成', "success");
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
            return;
        }

        ui.updateProgressBar((currentIndex / totalImages) * 100, `${currentIndex}/${totalImages}`);
        ui.showTranslatingIndicator(currentIndex);
<<<<<<< HEAD
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
=======
        const imageData = state.images[currentIndex];

        const data = {
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
            image: imageData.originalDataURL.split(',')[1],
            target_language: targetLanguage, source_language: sourceLanguage,
            fontSize: fontSize, autoFontSize: isAutoFontSize,
            api_key: apiKey, model_name: modelName, model_provider: modelProvider,
            fontFamily: fontFamily, textDirection: textDirection,
            prompt_content: promptContent, use_textbox_prompt: useTextboxPrompt,
            textbox_prompt_content: textboxPromptContent, use_inpainting: useInpainting,
            use_lama: useLama, blend_edges: blendEdges, inpainting_strength: inpaintingStrength,
<<<<<<< HEAD
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
=======
            fill_color: fillColor, 
            text_color: $('#textColor').val(),
            rotation_angle: parseFloat($('#rotationAngle').val() || '0'),
            skip_translation: false, skip_ocr: false, remove_only: false,
        };

        api.translateImageApi(data)
            .then(response => {
                ui.hideTranslatingIndicator(currentIndex);
                // 更新图片状态
                imageData.translatedDataURL = 'data:image/png;base64,' + response.translated_image;
                imageData.cleanImageData = response.clean_image;
                imageData.bubbleTexts = response.bubble_texts;
                imageData.bubbleCoords = response.bubble_coords;
                imageData.originalTexts = response.original_texts;
                imageData.textboxTexts = response.textbox_texts;
                imageData.fontSize = fontSize; imageData.autoFontSize = isAutoFontSize;
                imageData.fontFamily = fontFamily; imageData.layoutDirection = textDirection;
                imageData.showOriginal = false; imageData.translationFailed = false;
                imageData.originalUseInpainting = useInpainting; imageData.originalUseLama = useLama;
                imageData.inpaintingStrength = inpaintingStrength; imageData.blendEdges = blendEdges;
                imageData.fillColor = fillColor;
                imageData.textColor = $('#textColor').val();
                imageData.bubbleSettings = null; // 清空旧设置

                ui.renderThumbnails(); // 更新缩略图
                if (currentIndex === state.currentImageIndex) {
                    switchImage(state.currentImageIndex); // 更新当前显示
                }
            })
            .catch(error => {
                ui.hideTranslatingIndicator(currentIndex);
                console.error(`图片 ${currentIndex} 翻译失败:`, error);
                imageData.translationFailed = true;
                ui.renderThumbnails(); // 更新缩略图显示失败
            })
            .finally(() => {
                currentIndex++;
                processNextImage(); // 处理下一张
            });
    }
    processNextImage(); // 开始处理
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
}

// --- 其他需要导出的函数 ---
// (downloadCurrentImage, downloadAllImages, applySettingsToAll, removeBubbleTextOnly 等)
// 需要将它们的实现从 script.js 移到这里，并添加 export

/**
<<<<<<< HEAD
 * 下载当前图片（翻译后或原始图片）
 */
export function downloadCurrentImage() {
    const currentImage = state.getCurrentImage();
    // 修改：优先使用翻译后图片，如无则使用原始图片
    const imageDataURL = currentImage?.translatedDataURL || currentImage?.originalDataURL;
    
    if (currentImage && imageDataURL) {
        ui.showDownloadingMessage(true);
        try {
            const base64Data = imageDataURL.split(',')[1];
=======
 * 下载当前翻译后的图片
 */
export function downloadCurrentImage() {
    const currentImage = state.getCurrentImage();
    if (currentImage && currentImage.translatedDataURL) {
        ui.showDownloadingMessage(true);
        try {
            const base64Data = currentImage.translatedDataURL.split(',')[1];
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
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
<<<<<<< HEAD
            // 为已翻译和未翻译的图片使用不同前缀
            const prefix = currentImage.translatedDataURL ? 'translated' : 'original';
            fileName = `${prefix}_${fileName.replace(/\.[^/.]+$/, "")}.png`;
=======
            fileName = `translated_${fileName.replace(/\.[^/.]+$/, "")}.png`;
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (e) {
            console.error("下载图片失败:", e);
            ui.showGeneralMessage("下载图片失败", "error");
        } finally {
            ui.showDownloadingMessage(false);
        }
    } else {
<<<<<<< HEAD
        ui.showGeneralMessage("没有可下载的图片", "warning");
=======
        ui.showGeneralMessage("没有可下载的翻译图片", "warning");
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
    }
}

/**
 * 下载所有翻译后的图片
 */
export function downloadAllImages() {
    const selectedFormat = $('#downloadFormat').val();
<<<<<<< HEAD

    // --- 新增代码：立即显示提示信息 ---
    ui.showGeneralMessage("下载中...下载打包需要一定时间，请耐心等待...", "info", false, 0);
    // ---------------------------------

    ui.showDownloadingMessage(true); // 显示下载中并禁用按钮

    // 延迟执行下载，给 UI 更新时间 (保持不变)
=======
    ui.showDownloadingMessage(true); // 显示下载中

    // 延迟执行下载，给 UI 更新时间
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
    setTimeout(() => {
        try {
            switch(selectedFormat) {
                case 'zip': downloadAllAsZip(); break;
                case 'pdf': downloadAllAsPDF(); break;
                case 'cbz': downloadAllAsCBZ(); break;
                default: downloadAllAsZip();
            }
        } catch (e) {
            console.error("下载所有图片时出错:", e);
            ui.showGeneralMessage("下载失败", "error");
<<<<<<< HEAD
            // --- 新增：移除提示信息 ---
            $(".message.info").fadeOut(300, function() { $(this).remove(); });
            // ------------------------
            ui.showDownloadingMessage(false); // 确保出错时也解除禁用
        } finally {
             // 注意：ZIP/PDF/CBZ 函数内部会隐藏进度条和消息，并调用 showDownloadingMessage(false)
             console.log("Download All Images finally block executing...");
             // ui.showDownloadingMessage(false); // 这句可以移到 helper 函数的 finally 中确保执行
=======
        } finally {
             // 注意：ZIP/PDF/CBZ 函数内部会隐藏进度条和消息
             // ui.showDownloadingMessage(false); // 不在这里隐藏
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
        }
    }, 100); // 短暂延迟
}

<<<<<<< HEAD
// --- 修改 downloadAllAsZip, downloadAllAsPDF, downloadAllAsCBZ ---

function downloadAllAsZip() {
    ui.updateProgressBar(0, "准备 ZIP 下载...");
    $("#translationProgressBar").show();
    const zip = new JSZip();
    
    // 修改：使用所有图片，不仅是已翻译的
    const allImages = state.images;
    if (allImages.length === 0) {
        ui.showGeneralMessage("没有可下载的图片", "warning");
        $("#translationProgressBar").hide();
        // --- 新增：移除提示信息 ---
        $(".message.info").fadeOut(300, function() { $(this).remove(); });
        // ------------------------
        ui.showDownloadingMessage(false);
        return;
    }
    
    let processed = 0;
    allImages.forEach((imgData, i) => {
        // 优先使用翻译后的图片，如果没有则使用原始图片
        const imageDataURL = imgData.translatedDataURL || imgData.originalDataURL;
        // 跳过空/无效图片数据
        if (!imageDataURL) return;
        
        const base64Data = imageDataURL.split(',')[1];
        const fileName = imgData.fileName || `image_${i}.png`;
        // 已翻译图片使用translated前缀，未翻译使用original前缀
        const prefix = imgData.translatedDataURL ? 'translated' : 'original';
        const safeFileName = `${prefix}_${String(i).padStart(3, '0')}_${fileName.replace(/[^a-zA-Z0-9.]/g, '_')}`;
        
        zip.file(safeFileName + '.png', base64Data, {base64: true});
        processed++;
        ui.updateProgressBar((processed / allImages.length) * 100, `压缩进度: ${processed}/${allImages.length}`);
    });
    
=======
// --- ZIP/PDF/CBZ 下载辅助函数 (私有) ---
function downloadAllAsZip() {
    ui.updateProgressBar(0, "准备 ZIP 下载...");
    $("#translationProgressBar").show();
    const zip = new JSZip(); // 确保 JSZip 已加载
    const translatedImages = state.images.filter(img => img.translatedDataURL);
    if (translatedImages.length === 0) { /* ... */ $("#translationProgressBar").hide(); return; }
    let processed = 0;
    translatedImages.forEach((imgData, i) => {
        const base64Data = imgData.translatedDataURL.split(',')[1];
        const fileName = imgData.fileName || `image_${i}.png`;
        const safeFileName = `translated_${String(i).padStart(3, '0')}_${fileName.replace(/[^a-zA-Z0-9.]/g, '_')}`; // 清理文件名
        zip.file(safeFileName + '.png', base64Data, {base64: true});
        processed++;
        ui.updateProgressBar((processed / translatedImages.length) * 100, `压缩进度: ${processed}/${translatedImages.length}`);
    });
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
    zip.generateAsync({type:"blob"})
        .then(content => {
            const link = document.createElement('a');
            link.href = URL.createObjectURL(content);
<<<<<<< HEAD
            link.download = "comic_translator_images.zip";
            link.click();
            URL.revokeObjectURL(link.href);
            $("#translationProgressBar").hide();
            // --- 新增：移除提示信息 ---
            $(".message.info").fadeOut(300, function() { $(this).remove(); });
            // ------------------------
            ui.showGeneralMessage(`已成功下载 ${processed} 张图片 (ZIP)`, "success");
        })
        .catch(error => {
            console.error("ZIP 生成失败:", error);
            $("#translationProgressBar").hide();
            // --- 新增：移除提示信息 ---
            $(".message.info").fadeOut(300, function() { $(this).remove(); });
            // ------------------------
            ui.showGeneralMessage(`ZIP 文件生成失败: ${error.message || error}`, "error");
        })
        .finally(() => { // 确保按钮总是被重新启用
             ui.showDownloadingMessage(false);
        });
}

function downloadAllAsPDF() {
    ui.updateProgressBar(0, "准备 PDF 下载...");
    $("#translationProgressBar").show();
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF({orientation: 'p', unit: 'px', format: 'a4'});
    
    // 修改：使用所有图片，不仅是已翻译的
    const allImages = state.images;
    if (allImages.length === 0) {
        ui.showGeneralMessage("没有可下载的图片", "warning");
        $("#translationProgressBar").hide();
        // --- 新增：移除提示信息 ---
        $(".message.info").fadeOut(300, function() { $(this).remove(); });
        // ------------------------
        ui.showDownloadingMessage(false);
        return;
    }
    
    let processed = 0;
    const processImage = (index) => {
        if (index >= allImages.length) {
            try {
                pdf.save('comic_translator_images.pdf');
                $("#translationProgressBar").hide();
                // --- 新增：移除提示信息 ---
                $(".message.info").fadeOut(300, function() { $(this).remove(); });
                // ------------------------
                ui.showGeneralMessage(`已成功创建 PDF 文件`, "success");
            } catch (e) {
                console.error("PDF 保存失败:", e);
                $("#translationProgressBar").hide();
                 // --- 新增：移除提示信息 ---
                $(".message.info").fadeOut(300, function() { $(this).remove(); });
                // ------------------------
                ui.showGeneralMessage(`PDF 文件保存失败: ${e.message || e}`, "error");
            } finally {
                ui.showDownloadingMessage(false); // 完成后启用按钮
            }
            return;
        }
        
        // 如果不是第一页，添加新页面
        if (index > 0 && index < allImages.length) {
             pdf.addPage();
        }
        
        // 优先使用翻译后的图片，如果没有则使用原始图片
        const imageDataURL = allImages[index].translatedDataURL || allImages[index].originalDataURL;
        
        // 如果没有有效的图片数据，跳到下一张
        if (!imageDataURL) {
            processed++;
            ui.updateProgressBar((processed / allImages.length) * 100, `PDF 创建进度: ${processed}/${allImages.length}`);
            processImage(index + 1);
            return;
        }
        
        const img = new Image();
        img.onload = function() {
            try {
                const pdfWidth = pdf.internal.pageSize.getWidth();
                const pdfHeight = pdf.internal.pageSize.getHeight();
                const ratio = Math.min(pdfWidth / img.naturalWidth, pdfHeight / img.naturalHeight);
                const imgWidth = img.naturalWidth * ratio;
                const imgHeight = img.naturalHeight * ratio;
                const x = (pdfWidth - imgWidth) / 2;
                const y = (pdfHeight - imgHeight) / 2;
                pdf.addImage(img, 'PNG', x, y, imgWidth, imgHeight);
                processed++;
                ui.updateProgressBar((processed / allImages.length) * 100, `PDF 创建进度: ${processed}/${allImages.length}`);
                processImage(index + 1); // 递归
            } catch (e) {
                 console.error(`处理图片 ${index} 到 PDF 时出错:`, e);
                 ui.showGeneralMessage(`处理图片 ${allImages[index].fileName || index} 到 PDF 时出错: ${e.message || e}`, "error");
                 processed++;
                 ui.updateProgressBar((processed / allImages.length) * 100, `PDF 创建进度: ${processed}/${allImages.length}`);
                 processImage(index + 1); // 继续下一张
            }
        };
        img.onerror = () => {
            console.error(`加载图片 ${index} 失败，跳过`);
            ui.showGeneralMessage(`加载图片 ${allImages[index].fileName || index} 失败，已跳过`, "warning");
            processed++;
            ui.updateProgressBar((processed / allImages.length) * 100, `PDF 创建进度: ${processed}/${allImages.length}`);
            processImage(index + 1); // 继续下一张
        }
        img.src = imageDataURL;
    };
    processImage(0); // 开始处理
}

=======
            link.download = "translated_images.zip";
            link.click();
            URL.revokeObjectURL(link.href);
            $("#translationProgressBar").hide();
            ui.showGeneralMessage(`已成功下载 ${translatedImages.length} 张翻译图片 (ZIP)`, "success");
        })
        .catch(error => { /* ... */ $("#translationProgressBar").hide(); });
}
function downloadAllAsPDF() {
    ui.updateProgressBar(0, "准备 PDF 下载...");
    $("#translationProgressBar").show();
    const { jsPDF } = window.jspdf; // 确保 jsPDF 已加载
    const pdf = new jsPDF({orientation: 'p', unit: 'px', format: 'a4'}); // 使用像素单位可能更直观
    const translatedImages = state.images.filter(img => img.translatedDataURL);
    if (translatedImages.length === 0) { /* ... */ $("#translationProgressBar").hide(); return; }
    let processed = 0;
    const processImage = (index) => {
        if (index >= translatedImages.length) {
            pdf.save('translated_images.pdf');
            $("#translationProgressBar").hide();
            ui.showGeneralMessage(`已成功创建 PDF 文件`, "success");
            return;
        }
        if (index > 0) pdf.addPage();
        const img = new Image();
        img.onload = function() {
            const pdfWidth = pdf.internal.pageSize.getWidth();
            const pdfHeight = pdf.internal.pageSize.getHeight();
            const ratio = Math.min(pdfWidth / img.naturalWidth, pdfHeight / img.naturalHeight);
            const imgWidth = img.naturalWidth * ratio;
            const imgHeight = img.naturalHeight * ratio;
            const x = (pdfWidth - imgWidth) / 2;
            const y = (pdfHeight - imgHeight) / 2;
            pdf.addImage(img, 'PNG', x, y, imgWidth, imgHeight);
            processed++;
            ui.updateProgressBar((processed / translatedImages.length) * 100, `PDF 创建进度: ${processed}/${translatedImages.length}`);
            processImage(index + 1);
        };
        img.onerror = () => { // 处理图片加载失败
             console.error(`加载图片 ${index} 失败，跳过`);
             processed++;
             ui.updateProgressBar((processed / translatedImages.length) * 100, `PDF 创建进度: ${processed}/${translatedImages.length}`);
             processImage(index + 1);
        }
        img.src = translatedImages[index].translatedDataURL;
    };
    processImage(0);
}
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
function downloadAllAsCBZ() {
    ui.updateProgressBar(0, "准备 CBZ 下载...");
    $("#translationProgressBar").show();
    const zip = new JSZip();
<<<<<<< HEAD
    
    // 修改：使用所有图片，不仅是已翻译的
    const allImages = state.images;
    if (allImages.length === 0) {
        ui.showGeneralMessage("没有可下载的图片", "warning");
        $("#translationProgressBar").hide();
        // --- 新增：移除提示信息 ---
        $(".message.info").fadeOut(300, function() { $(this).remove(); });
        // ------------------------
        ui.showDownloadingMessage(false);
        return;
    }
    
    let processed = 0;
    allImages.forEach((imgData, i) => {
        // 优先使用翻译后的图片，如果没有则使用原始图片
        const imageDataURL = imgData.translatedDataURL || imgData.originalDataURL;
        // 跳过空/无效图片数据
        if (!imageDataURL) return;
        
        const base64Data = imageDataURL.split(',')[1];
        const fileName = imgData.fileName || `image_${i}.png`;
        // 为已翻译和未翻译的图片使用不同前缀（仅在文件名中，不影响CBZ排序）
        const prefix = imgData.translatedDataURL ? 't' : 'o';
        const safeFileName = `${String(i).padStart(3, '0')}_${prefix}_${fileName.replace(/[^a-zA-Z0-9.]/g, '_')}`;
        
        zip.file(safeFileName + '.png', base64Data, {base64: true});
        processed++;
        ui.updateProgressBar((processed / allImages.length) * 100, `CBZ 创建进度: ${processed}/${allImages.length}`);
    });
    
=======
    const translatedImages = state.images.filter(img => img.translatedDataURL);
    if (translatedImages.length === 0) { /* ... */ $("#translationProgressBar").hide(); return; }
    let processed = 0;
    translatedImages.forEach((imgData, i) => {
        const base64Data = imgData.translatedDataURL.split(',')[1];
        const fileName = imgData.fileName || `image_${i}.png`;
        const safeFileName = `${String(i).padStart(3, '0')}_${fileName.replace(/[^a-zA-Z0-9.]/g, '_')}`;
        zip.file(safeFileName + '.png', base64Data, {base64: true}); // 确保是 png
        processed++;
        ui.updateProgressBar((processed / translatedImages.length) * 100, `CBZ 创建进度: ${processed}/${translatedImages.length}`);
    });
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
    zip.generateAsync({type:"blob"})
        .then(content => {
            const link = document.createElement('a');
            link.href = URL.createObjectURL(content);
<<<<<<< HEAD
            link.download = "comic_translator_images.cbz";
            link.click();
            URL.revokeObjectURL(link.href);
            $("#translationProgressBar").hide();
             // --- 新增：移除提示信息 ---
            $(".message.info").fadeOut(300, function() { $(this).remove(); });
            // ------------------------
            ui.showGeneralMessage(`已成功创建 CBZ 文件`, "success");
        })
        .catch(error => {
            console.error("CBZ 生成失败:", error);
            $("#translationProgressBar").hide();
            // --- 新增：移除提示信息 ---
            $(".message.info").fadeOut(300, function() { $(this).remove(); });
            // ------------------------
            ui.showGeneralMessage(`CBZ 文件生成失败: ${error.message || error}`, "error");
        })
        .finally(() => { // 确保按钮总是被重新启用
            ui.showDownloadingMessage(false);
        });
=======
            link.download = "translated_images.cbz";
            link.click();
            URL.revokeObjectURL(link.href);
            $("#translationProgressBar").hide();
            ui.showGeneralMessage(`已成功创建 CBZ 文件`, "success");
        })
        .catch(error => { /* ... */ $("#translationProgressBar").hide(); });
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
}

/**
 * 将当前字体设置应用到所有图片
 */
export function applySettingsToAll() { // 导出
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
        rotationAngle: 0 // 全局应用时不应用旋转
    };

    ui.showLoading("应用设置到所有图片...");

    // 准备数据给后端（如果需要后端重新渲染）
    // 或者在前端直接修改状态并调用 reRenderFullImage (如果性能允许)

    // --- 前端直接修改状态并渲染 ---
    state.images.forEach((img, index) => {
        if (img.translatedDataURL) { // 只修改已翻译的图片
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
                    rotationAngle: settingsToApply.rotationAngle
                }));
            }
        }
    });

    // 重新渲染当前图片以立即看到效果
    if (state.getCurrentImage()?.translatedDataURL) {
<<<<<<< HEAD
        editMode.reRenderFullImage(); // 不再传递全局变更标记
=======
        editMode.reRenderFullImage(true); // 标记为全局变更
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
    }
    ui.hideLoading();
    ui.showGeneralMessage("设置已应用到所有已翻译图片（下次查看时生效）", "success");
    // 注意：这种前端修改方式不会重新渲染所有图片，只在切换到对应图片时生效
    // 如果需要立即重新渲染所有图片，需要调用后端 API
}

/**
 * 仅消除当前图片的气泡文字
 */
export function removeBubbleTextOnly() { // 导出
    const currentImage = state.getCurrentImage();
    if (!currentImage) return;

    // 移除showLoading调用，events.js已处理显示消息
    ui.showTranslatingIndicator(state.currentImageIndex);

    const repairSettings = ui.getRepairSettings();
    const isAutoFontSize = $('#autoFontSize').is(':checked');
    const fontSize = isAutoFontSize ? 'auto' : $('#fontSize').val();
<<<<<<< HEAD
    const ocr_engine = $('#ocrEngine').val();

    // --- 关键修改：检查并使用已保存的手动坐标 ---
    let coordsToUse = null; // 默认不传递，让后端自动检测
    let usedManualCoords = false; // 标记是否使用了手动坐标
    if (currentImage.savedManualCoords && currentImage.savedManualCoords.length > 0) {
        coordsToUse = currentImage.savedManualCoords;
        usedManualCoords = true;
        console.log(`消除文字 ${state.currentImageIndex}: 将使用 ${coordsToUse.length} 个已保存的手动标注框。`);
        ui.showGeneralMessage("检测到手动标注框，将优先使用...", "info", false, 3000);
    } else {
        console.log(`消除文字 ${state.currentImageIndex}: 未找到手动标注框，将进行自动检测。`);
    }
    // ------------------------------------------
=======
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915

    const params = {
        image: currentImage.originalDataURL.split(',')[1],
        target_language: $('#targetLanguage').val(),
        source_language: $('#sourceLanguage').val(),
        fontSize: fontSize, autoFontSize: isAutoFontSize,
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
<<<<<<< HEAD
        skip_translation: true,
        remove_only: true,
        use_json_format_translation: false,
        use_json_format_ai_vision_ocr: state.isAiVisionOcrJsonMode,
        bubble_coords: coordsToUse,
        ocr_engine: ocr_engine,
        baidu_api_key: ocr_engine === 'baidu_ocr' ? $('#baiduApiKey').val() : null,
        baidu_secret_key: ocr_engine === 'baidu_ocr' ? $('#baiduSecretKey').val() : null,
        baidu_version: ocr_engine === 'baidu_ocr' ? $('#baiduVersion').val() : 'standard',
        ai_vision_provider: ocr_engine === 'ai_vision' ? $('#aiVisionProvider').val() : null,
        ai_vision_api_key: ocr_engine === 'ai_vision' ? $('#aiVisionApiKey').val() : null,
        ai_vision_model_name: ocr_engine === 'ai_vision' ? $('#aiVisionModelName').val() : null,
        ai_vision_ocr_prompt: ocr_engine === 'ai_vision' ? $('#aiVisionOcrPrompt').val() : null
    };

    // 检查百度OCR配置
    if (ocr_engine === 'baidu_ocr' && (!params.baidu_api_key || !params.baidu_secret_key)) {
        ui.showGeneralMessage("使用百度OCR时必须提供API Key和Secret Key", "error");
        ui.hideTranslatingIndicator(state.currentImageIndex);
        return;
    }
    
    // 检查AI视觉OCR配置
    if (ocr_engine === 'ai_vision' && (!params.ai_vision_api_key || !params.ai_vision_model_name)) {
        ui.showGeneralMessage("使用AI视觉OCR时必须提供API Key和模型名称", "error");
        ui.hideTranslatingIndicator(state.currentImageIndex);
        return;
    }

    // 返回一个Promise
    return new Promise((resolve, reject) => {
        api.translateImageApi(params)
            .then(response => {
                ui.hideTranslatingIndicator(state.currentImageIndex);

                // 更新当前图片对象
                currentImage.translatedDataURL = 'data:image/png;base64,' + response.translated_image;
                currentImage.cleanImageData = response.clean_image;
                currentImage.bubbleTexts = response.bubble_texts || [];  // 可能为空
                currentImage.bubbleCoords = response.bubble_coords || [];
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
                
                // --- 如果使用了手动坐标，保留标注状态 ---
                if (usedManualCoords) {
                    // 不再清除标注状态，确保可以继续使用手动标注框
                    // state.clearSavedManualCoords(); 
                    ui.renderThumbnails(); // 更新缩略图，保留标注状态
                }
                // --------------------------------------------
                
                // --- 关键修改：初始化气泡设置 ---
                const bubbleCoords = response.bubble_coords || [];
                const bubbleTexts = response.bubble_texts || [];
                
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
=======
        skip_translation: true, // 告诉后端跳过翻译步骤
        skip_ocr: false, // 仍然需要OCR来定位文本
        remove_only: true, // 仅消除文字
    };

    api.translateImageApi(params)
        .then(response => {
            // 移除hideLoading调用
            ui.hideTranslatingIndicator(state.currentImageIndex);

            // 更新当前图片对象
            currentImage.translatedDataURL = 'data:image/png;base64,' + response.translated_image;
            currentImage.cleanImageData = response.clean_image;
            currentImage.bubbleTexts = response.bubble_texts || [];  // 可能为空
            currentImage.bubbleCoords = response.bubble_coords || [];
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
            currentImage.bubbleSettings = null;

            switchImage(state.currentImageIndex); // 重新加载以更新所有 UI
            ui.showGeneralMessage("文字已消除", "success");
            ui.updateButtonStates();
            ui.updateRetranslateButton();
            ui.updateDetectedTextDisplay(); // 显示检测到的文本
        })
        .catch(error => {
            // 移除hideLoading调用
            ui.hideTranslatingIndicator(state.currentImageIndex);
            ui.showGeneralMessage(`操作失败: ${error.message}`, "error");
            ui.updateButtonStates();
        });
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
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

<<<<<<< HEAD
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
=======
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915

// --- 应用启动 ---
$(document).ready(initializeApp);
