// src/app/static/js/main.js

// 引入所有需要的模块
import * as state from './state.js';
import * as ui from './ui.js';
import * as api from './api.js';
import * as events from './events.js';
import * as editMode from './edit_mode.js';
import * as constants from './constants.js'; // 导入前端常量
import * as session from './session.js'; // 导入session模块，用于书架模式保存/加载
import * as hqTranslation from './high_quality_translation.js'; // 导入高质量翻译模块
import * as settingsModal from './settings_modal.js'; // 导入设置模态框模块
import * as translationValidator from './translation_validator.js'; // 导入翻译配置验证模块
import * as bubbleStateModule from './bubble_state.js'; // 导入统一气泡状态模块
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

    // 4. 高质量翻译模块现在由设置模态框初始化

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

    // 5. 初始化主题模式
    initializeThemeMode();

    // 6. 翻译服务和OCR设置现在由设置模态框处理，不再需要初始化侧边栏UI

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
    const isLamaMethod = initialRepairMethod === 'lama_mpe' || initialRepairMethod === 'litelama';
    ui.toggleInpaintingOptions(
        initialRepairMethod === 'true' || isLamaMethod,
        initialRepairMethod === 'false'
    );

    // 11. 初始化 UI 显示
    ui.updateTranslatePromptUI(); // 更新漫画翻译提示词UI
    ui.updateAiVisionOcrPromptUI(); // 更新AI视觉OCR提示词UI

    // 初始化选择器的默认值
    $('#translatePromptModeSelect').val(state.isTranslateJsonMode ? 'json' : 'normal');
    $('#aiVisionPromptModeSelect').val(state.isAiVisionOcrJsonMode ? 'json' : 'normal');

    state.setDefaultStrokeSettings(
        $('#strokeEnabled').is(':checked'),
        $('#strokeColor').val(),
        parseInt($('#strokeWidth').val()) || 1
    );
    state.setStrokeEnabled(state.defaultStrokeEnabled);
    state.setStrokeColor(state.defaultStrokeColor);
    state.setStrokeWidth(state.defaultStrokeWidth);

    $("#strokeOptions").toggle(state.strokeEnabled);

    // 12. 处理书籍/章节 URL 参数
    initializeBookChapterContext();

    // 13. 初始化设置模态框
    try {
        settingsModal.initSettingsModal();
        settingsModal.bindTestButtons();
        console.log("设置模态框初始化完成");
    } catch (error) {
        console.error("初始化设置模态框失败:", error);
    }

    // 14. 初始化翻译配置验证器（显示首次使用提醒）
    try {
        translationValidator.initTranslationValidator();
        console.log("翻译配置验证器初始化完成");
    } catch (error) {
        console.error("初始化翻译配置验证器失败:", error);
    }

    console.log("应用程序初始化完成。");
}

/**
 * 初始化书籍/章节上下文
 * 从 URL 参数中读取 book 和 chapter，加载对应的会话数据
 */
async function initializeBookChapterContext() {
    const urlParams = new URLSearchParams(window.location.search);
    const bookId = urlParams.get('book');
    const chapterId = urlParams.get('chapter');
    
    if (!bookId || !chapterId) {
        console.log('未指定书籍/章节参数，使用独立模式');
        return;
    }
    
    console.log(`检测到书籍/章节参数: book=${bookId}, chapter=${chapterId}`);
    
    try {
        // 获取书籍和章节信息
        const bookResponse = await fetch(`/api/bookshelf/books/${bookId}`);
        const bookData = await bookResponse.json();
        
        if (!bookData.success) {
            console.warn('书籍不存在:', bookId);
            ui.showGeneralMessage('书籍不存在', 'warning');
            return;
        }
        
        const book = bookData.book;
        const chapter = book.chapters?.find(c => c.id === chapterId);
        
        if (!chapter) {
            console.warn('章节不存在:', chapterId);
            ui.showGeneralMessage('章节不存在', 'warning');
            return;
        }
        
        // 设置书籍/章节上下文
        state.setBookChapterContext(bookId, chapterId, book.title, chapter.title);
        
        // 更新页面标题
        document.title = `${chapter.title} - ${book.title} - Saber-Translator`;
        
        // 书架模式下显示保存按钮（保存按钮默认隐藏，仅在书架模式下显示）
        const saveBtn = document.getElementById('saveCurrentSessionButton');
        if (saveBtn) saveBtn.style.display = 'flex';
        
        // 尝试加载章节的会话数据
        if (chapter.session_path) {
            console.log(`尝试加载章节会话: ${chapter.session_path}`);
            // 使用现有的会话加载机制
            try {
                const loadResult = await session.loadSessionByPath(chapter.session_path);
                if (loadResult) {
                    ui.showGeneralMessage(`已加载章节: ${chapter.title}`, 'success');
                }
            } catch (e) {
                console.log('章节会话数据不存在或加载失败，将创建新会话');
            }
        }
        
    } catch (error) {
        console.error('加载书籍/章节信息失败:', error);
        ui.showGeneralMessage('加载书籍信息失败', 'error');
    }
}

// --- 辅助函数 (从原始 script.js 迁移) ---

/**
 * 初始化漫画翻译提示词设置
 * 提示词下拉管理功能已移至设置模态框的"提示词管理"页面
 */
export function initializePromptSettings() {
    api.getPromptsApi()
        .then(response => {
            state.setPromptState(
                state.isTranslateJsonMode ? state.defaultTranslateJsonPrompt : response.default_prompt_content,
                response.default_prompt_content,
                response.prompt_names || [],
                state.defaultTranslateJsonPrompt
            );
            ui.updateTranslatePromptUI();
        })
        .catch(error => {
            console.error("获取提示词信息失败:", error);
            const errorMsg = "获取默认提示词失败";
            state.setPromptState(errorMsg, errorMsg, []);
            ui.updateTranslatePromptUI();
        });
}

/**
 * 初始化文本框提示词设置
 * 提示词下拉管理功能已移至设置模态框的"提示词管理"页面
 */
export function initializeTextboxPromptSettings() {
    api.getTextboxPromptsApi()
        .then(response => {
            state.setTextboxPromptState(response.default_prompt_content, response.default_prompt_content, response.prompt_names || []);
            $('#textboxPromptContent').val(state.currentTextboxPromptContent);
        })
        .catch(error => {
            console.error("获取文本框提示词信息失败:", error);
            const errorMsg = "获取默认文本框提示词失败";
            state.setTextboxPromptState(errorMsg, errorMsg, []);
            $('#textboxPromptContent').val(errorMsg);
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

// 旧的loadPromptContent、deletePrompt、loadTextboxPromptContent、deleteTextboxPrompt函数
// 已移至设置模态框的提示词管理功能中


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
    // 默认所有配置都展开
    // collapsibleHeaders.each(function(index) {
    //     if (index > 0) {
    //         const header = $(this);
    //         header.addClass("collapsed");
    //         header.next(".collapsible-content").addClass("collapsed");
    //         header.find(".toggle-icon").text("▶");
    //     }
    // });
}

/**
 * 初始化亮暗模式切换
 */
function initializeThemeMode() { // 私有辅助函数
    const savedTheme = localStorage.getItem('themeMode');
    // 只有明确保存为 'dark' 时才使用深色模式，否则默认浅色模式
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
        document.body.classList.remove('light-mode');
    } else {
        // 默认使用浅色模式，并保存设置
        document.body.classList.add('light-mode');
        document.body.classList.remove('dark-mode');
        if (!savedTheme) {
            localStorage.setItem('themeMode', 'light');
        }
    }
    // 事件绑定在 events.js 中处理
}

// checkInitialModelProvider, fetchOllamaModels, fetchSakuraModels 已移除
// 这些功能现在由设置模态框(settings_modal.js)处理

/**
 * 处理文件（图片或 PDF）
 * @param {FileList} files - 用户选择或拖放的文件列表
 */
export function handleFiles(files) { // 导出
    if (!files || files.length === 0) return;

    // 注意：不再自动清除书籍/章节上下文
    // 用户在书架模式下上传图片时应保持在书架模式，以便继续编辑当前章节
    // 如需退出书架模式，用户可以通过"清空图片"或返回书架页面来实现

    ui.showLoading("处理文件中...");
    ui.hideError();

    const imagePromises = [];
    const pdfFiles = [];
    const mobiFiles = [];  // MOBI/AZW 电子书文件

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const ext = file.name.split('.').pop().toLowerCase();
        
        if (file.type.startsWith('image/')) {
            imagePromises.push(processImageFile(file));
        } else if (file.type === 'application/pdf') {
            pdfFiles.push(file);
        } else if (['mobi', 'azw', 'azw3'].includes(ext)) {
            // MOBI/AZW 电子书格式
            mobiFiles.push(file);
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
            if (mobiFiles.length > 0) {
                return processMobiFiles(mobiFiles);
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
        })
        .catch(error => {
            ui.hideLoading();
            console.error("处理文件失败:", error);
            ui.showError(`处理文件时出错: ${error.message || error}`);
            ui.updateButtonStates();
        }).finally(() => {
            // 最后配置页数选择上下限
            $("#pageStart").attr('min', 1).attr('max', state.images.length)
            $("#pageEnd").attr('min', 1).attr('max', state.images.length)
        })
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
                bubbleStates: null, fileName: file.name,
                fontSize: state.defaultFontSize, autoFontSize: $('#autoFontSize').is(':checked'),
                fontFamily: state.defaultFontFamily, layoutDirection: state.defaultLayoutDirection,
                showOriginal: false, translationFailed: false,
            });
            resolve();
        };
        reader.onerror = (error) => reject(error);
        reader.readAsDataURL(file);
    });
}

/**
 * 处理 PDF 文件列表 (浏览器端解析，无需服务器)
 * @param {Array<File>} pdfFiles - PDF 文件数组
 * @returns {Promise<void>}
 */
async function processPDFFiles(pdfFiles) { // 私有
    for (const file of pdfFiles) {
        try {
            ui.showLoading(`正在解析 PDF: ${file.name}...`);
            
            // 读取文件为 ArrayBuffer
            const arrayBuffer = await file.arrayBuffer();
            
            // 使用 pdf.js 加载 PDF（浏览器端解析）
            const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
            const numPages = pdf.numPages;
            
            console.log(`PDF ${file.name} 共 ${numPages} 页，开始本地渲染...`);
            
            // 检测是否支持 OffscreenCanvas（后台渲染不受页面可见性影响）
            const useOffscreen = typeof OffscreenCanvas !== 'undefined';
            if (useOffscreen) {
                console.log('使用 OffscreenCanvas 后台渲染模式');
            }
            
            // 逐页渲染为图片
            for (let pageNum = 1; pageNum <= numPages; pageNum++) {
                ui.showLoading(`处理 PDF: ${file.name} (${pageNum}/${numPages})...`);
                
                try {
                    const page = await pdf.getPage(pageNum);
                    
                    // 设置渲染比例，2.0 可以获得较高清晰度
                    const scale = 2.0;
                    const viewport = page.getViewport({ scale });
                    
                    let originalDataURL;
                    
                    if (useOffscreen) {
                        // 使用 OffscreenCanvas - 后台也能继续渲染
                        const offscreen = new OffscreenCanvas(viewport.width, viewport.height);
                        const context = offscreen.getContext('2d');
                        
                        await page.render({
                            canvasContext: context,
                            viewport: viewport
                        }).promise;
                        
                        // OffscreenCanvas 转 Blob 再转 DataURL (JPEG 最高质量)
                        const blob = await offscreen.convertToBlob({ type: 'image/jpeg', quality: 1.0 });
                        originalDataURL = await blobToDataURL(blob);
                    } else {
                        // 回退：使用普通 Canvas
                        const canvas = document.createElement('canvas');
                        const context = canvas.getContext('2d');
                        canvas.width = viewport.width;
                        canvas.height = viewport.height;
                        
                        await page.render({
                            canvasContext: context,
                            viewport: viewport
                        }).promise;
                        
                        originalDataURL = canvas.toDataURL('image/jpeg', 1.0);
                    }
                    
                    const pdfFileName = `${file.name}_页面${pageNum}`;
                    
                    // 添加到状态（跟图片上传一样）
                    state.addImage({
                        originalDataURL: originalDataURL,
                        translatedDataURL: null, 
                        cleanImageData: null,
                        bubbleTexts: [], 
                        bubbleCoords: [], 
                        originalTexts: [], 
                        textboxTexts: [],
                        bubbleStates: null, 
                        fileName: pdfFileName,
                        fontSize: state.defaultFontSize, 
                        autoFontSize: $('#autoFontSize').is(':checked'),
                        fontFamily: state.defaultFontFamily, 
                        layoutDirection: state.defaultLayoutDirection,
                        showOriginal: false, 
                        translationFailed: false,
                    });
                    
                    console.log(`  页面 ${pageNum}/${numPages} 处理完成`);
                    
                } catch (pageError) {
                    console.warn(`PDF ${file.name} 第 ${pageNum} 页渲染失败:`, pageError);
                }
            }
            
            console.log(`PDF ${file.name} 全部 ${numPages} 页处理完成`);
            
        } catch (error) {
            console.error(`处理PDF文件 ${file.name} 失败:`, error);
            ui.showGeneralMessage(`处理PDF文件 ${file.name} 失败: ${error.message}`, "error");
        }
    }
}

/**
 * 处理 MOBI/AZW 电子书文件（上传到后端解析，分批获取图片）
 * @param {Array<File>} mobiFiles - MOBI 文件数组
 * @returns {Promise<void>}
 */
async function processMobiFiles(mobiFiles) {
    const BATCH_SIZE = 5;  // 每批获取的页数
    
    for (const file of mobiFiles) {
        let sessionId = null;
        
        try {
            ui.showLoading(`正在解析电子书: ${file.name}...`);
            
            // 步骤1: 上传文件并创建解析会话
            const formData = new FormData();
            formData.append('file', file);
            
            const startResponse = await fetch('/api/parse_mobi_start', {
                method: 'POST',
                body: formData
            });
            
            const startResult = await startResponse.json();
            
            if (!startResult.success) {
                throw new Error(startResult.error || '解析失败');
            }
            
            sessionId = startResult.session_id;
            const totalPages = startResult.total_pages;
            
            console.log(`MOBI ${file.name} 共 ${totalPages} 页，开始分批获取...`);
            
            // 步骤2: 分批获取图片
            let loadedCount = 0;
            
            for (let startIndex = 0; startIndex < totalPages; startIndex += BATCH_SIZE) {
                ui.showLoading(`处理电子书: ${file.name} (${Math.min(startIndex + BATCH_SIZE, totalPages)}/${totalPages})...`);
                
                const batchResponse = await fetch('/api/parse_mobi_batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        start_index: startIndex,
                        count: BATCH_SIZE
                    })
                });
                
                const batchResult = await batchResponse.json();
                
                if (!batchResult.success) {
                    console.warn(`批次 ${startIndex} 获取失败:`, batchResult.error);
                    continue;
                }
                
                // 将本批图片添加到状态
                for (const imgData of batchResult.images) {
                    const mobiFileName = `${file.name}_页面${String(imgData.page_index + 1).padStart(4, '0')}`;
                    
                    state.addImage({
                        originalDataURL: imgData.data_url,
                        translatedDataURL: null,
                        cleanImageData: null,
                        bubbleTexts: [],
                        bubbleCoords: [],
                        originalTexts: [],
                        textboxTexts: [],
                        bubbleStates: null,
                        fileName: mobiFileName,
                        fontSize: state.defaultFontSize,
                        autoFontSize: $('#autoFontSize').is(':checked'),
                        fontFamily: state.defaultFontFamily,
                        layoutDirection: state.defaultLayoutDirection,
                        showOriginal: false,
                        translationFailed: false,
                    });
                    
                    loadedCount++;
                }
                
                console.log(`  已加载 ${loadedCount}/${totalPages} 页`);
            }
            
            console.log(`MOBI ${file.name} 全部 ${loadedCount} 页处理完成`);
            
        } catch (error) {
            console.error(`处理MOBI文件 ${file.name} 失败:`, error);
            ui.showGeneralMessage(`处理电子书 ${file.name} 失败: ${error.message}`, "error");
        } finally {
            // 步骤3: 清理会话
            if (sessionId) {
                try {
                    await fetch(`/api/parse_mobi_cleanup/${sessionId}`, { method: 'POST' });
                } catch (e) {
                    console.warn('清理 MOBI 会话失败:', e);
                }
            }
        }
    }
}

/**
 * 将 Blob 转换为 DataURL
 * @param {Blob} blob
 * @returns {Promise<string>}
 */
function blobToDataURL(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
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

    // 设置一个全局标记，表示当前正在进行切换图片操作
    // 这个标记将在 handleGlobalSettingChange 中被检测到，以避免重渲染
    window._isChangingFromSwitchImage = true;

    // --- 退出当前模式 (如果需要) ---
    // 如果在编辑模式，保存当前图片的 bubbleStates，但不触发重渲染
    if (state.editModeActive) {
        // 使用新的函数退出编辑模式但不触发重渲染
        editMode.exitEditModeWithoutRender();
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
        // 全局 rotationAngle 已移除
        $('#strokeEnabled').prop('checked', state.defaultStrokeEnabled);
        $('#strokeColor').val(state.defaultStrokeColor);
        $('#strokeWidth').val(state.defaultStrokeWidth);
        $("#strokeOptions").toggle(state.defaultStrokeEnabled);
        
        state.setStrokeEnabled(state.defaultStrokeEnabled);
        state.setStrokeColor(state.defaultStrokeColor);
        state.setStrokeWidth(state.defaultStrokeWidth);
    } else if (!imageData.translatedDataURL && imageData.originalDataURL) {
        // 通常我们希望保留用户在翻译前的全局设置，或者也可以重置
        // 这里我们选择保留当前UI控件的值（通常是上一个图片的或全局默认的）
        // 但要确保 fillColor 反映图片自身的记录，如果没有，则用全局的
        $('#fillColor').val(imageData.fillColor || state.defaultFillColor);
        
        const strokeEnabled = imageData.strokeEnabled === undefined ? state.strokeEnabled : imageData.strokeEnabled;
        const strokeColor = imageData.strokeColor || state.strokeColor;
        const strokeWidth = imageData.strokeWidth === undefined ? state.strokeWidth : imageData.strokeWidth;
        
        $('#strokeEnabled').prop('checked', strokeEnabled);
        $('#strokeColor').val(strokeColor);
        $('#strokeWidth').val(strokeWidth);
        $("#strokeOptions").toggle(strokeEnabled);
        
        state.setStrokeEnabled(strokeEnabled);
        state.setStrokeColor(strokeColor);
        state.setStrokeWidth(strokeWidth);
    } else { // 图片已翻译或处理过
        // 简化：从 bubbleStates[0] 读取设置，否则使用默认值
        const s = imageData.bubbleStates?.[0] || {};
        
        $('#fontSize').val(s.fontSize || state.defaultFontSize);
        // autoFontSize 从图片级别状态读取（bubbleStates 中已移除此字段）
        $('#autoFontSize').prop('checked', imageData.autoFontSize || false);
        
        // 设置字体
        const fontToSet = s.fontFamily || state.defaultFontFamily;
        const fontSelect = $('#fontFamily');
        if (fontSelect.find(`option[value="${fontToSet}"]`).length > 0) {
            fontSelect.val(fontToSet);
        } else {
            fontSelect.val(state.defaultFontFamily);
        }
        
        // 【重要】优先使用图片保存的用户选择（包括 'auto'），否则使用气泡状态的方向
        $('#layoutDirection').val(imageData.userLayoutDirection || s.textDirection || state.defaultLayoutDirection);
        $('#textColor').val(s.textColor || state.defaultTextColor);
        $('#fillColor').val(s.fillColor || state.defaultFillColor);
        
        const strokeEnabled = s.strokeEnabled !== undefined ? s.strokeEnabled : state.defaultStrokeEnabled;
        $('#strokeEnabled').prop('checked', strokeEnabled);
        $('#strokeColor').val(s.strokeColor || state.defaultStrokeColor);
        $('#strokeWidth').val(s.strokeWidth || state.defaultStrokeWidth);
        $("#strokeOptions").toggle(strokeEnabled);
        
        state.setStrokeEnabled(strokeEnabled);
        state.setStrokeColor(s.strokeColor || state.defaultStrokeColor);
        state.setStrokeWidth(s.strokeWidth || state.defaultStrokeWidth);
    }

    // 触发 change 以更新依赖 UI (比如修复选项的显隐)
    $('#useInpainting').trigger('change'); // 这个会调用 toggleInpaintingOptions
    $('#autoFontSize').trigger('change');

    ui.updateButtonStates();
    $('.thumbnail-item').removeClass('active');
    $(`.thumbnail-item[data-index="${index}"]`).addClass('active');
    ui.scrollToActiveThumbnail();
    
    // 重置切换图片操作的标记
    setTimeout(() => {
        window._isChangingFromSwitchImage = false;
        console.log("已重置切换图片操作标记");
    }, 100); // 短暂延迟以确保所有设置更改事件都能检测到这个标记
}

/**
 * 翻译当前图片
 */
export function translateCurrentImage(config) {
    const translateMerge = config?.translate_merge || false
    const chatSession = config?.chat_session || null

    const currentImage = state.getCurrentImage();
    if (!currentImage) return Promise.reject("No current image"); // 返回一个被拒绝的Promise

    // 验证翻译配置是否完整
    if (!translationValidator.validateBeforeTranslation('normal')) {
        return Promise.reject("Translation config validation failed");
    }

    ui.showGeneralMessage("翻译中...", "info", false, 0);
    ui.showTranslatingIndicator(state.currentImageIndex);

    const repairSettings = ui.getRepairSettings();
    const isAutoFontSize = $('#autoFontSize').is(':checked');
    const fontSize = isAutoFontSize ? 'auto' : $('#fontSize').val();
    const ocr_engine = $('#ocrEngine').val();
    const modelProvider = $('#modelProvider').val(); // 获取当前选中的服务商

    // --- 关键修改：检查并使用已有的坐标和角度（优先级：savedManualCoords > bubbleCoords > 自动检测）---
    // 角度优先从 bubbleStates 提取（唯一状态来源），回退到 bubbleAngles（检测结果）
    let coordsToUse = null;
    let anglesToUse = null;
    let usedExistingCoords = false;
    
    // 辅助函数：从 bubbleStates 提取角度
    const extractAngles = (states) => states && states.length > 0 ? states.map(s => s.rotationAngle || 0) : null;
    
    if (currentImage.savedManualCoords && currentImage.savedManualCoords.length > 0) {
        coordsToUse = currentImage.savedManualCoords;
        anglesToUse = currentImage.savedManualAngles || null;
        usedExistingCoords = true;
        console.log(`翻译当前图片 ${state.currentImageIndex}: 将使用 ${coordsToUse.length} 个已保存的手动标注框。`);
        ui.showGeneralMessage("检测到手动标注框，将优先使用...", "info", false, 3000);
    } else if (currentImage.bubbleCoords && currentImage.bubbleCoords.length > 0) {
        coordsToUse = currentImage.bubbleCoords;
        // 优先从 bubbleStates 提取角度（用户可能已调整），回退到 bubbleAngles（检测结果）
        anglesToUse = extractAngles(currentImage.bubbleStates) || currentImage.bubbleAngles || null;
        usedExistingCoords = true;
        console.log(`翻译当前图片 ${state.currentImageIndex}: 将使用 ${coordsToUse.length} 个已有的文本框，角度来源: ${currentImage.bubbleStates ? 'bubbleStates' : 'bubbleAngles'}`);
        ui.showGeneralMessage("使用已有的文本框进行翻译...", "info", false, 3000);
    } else {
        console.log(`翻译当前图片 ${state.currentImageIndex}: 未找到已有文本框，将进行自动检测。`);
    }
    // ------------------------------------------

    // 处理排版方向：如果是 "auto" 则启用自动排版
    const layoutDirectionValue = $('#layoutDirection').val();
    const isAutoTextDirection = layoutDirectionValue === 'auto';
    // 自动排版时默认使用 vertical，实际方向由后端检测决定
    const textDirectionToSend = isAutoTextDirection ? 'vertical' : layoutDirectionValue;
    
    const params = {
        image: currentImage.originalDataURL.split(',')[1],
        target_language: $('#targetLanguage').val(),
        source_language: $('#sourceLanguage').val(),
        fontSize: fontSize, autoFontSize: isAutoFontSize,
        api_key: $('#apiKey').val(),
        model_name: $('#modelName').val(),
        model_provider: modelProvider, // 使用获取到的服务商
        fontFamily: $('#fontFamily').val(),
        textDirection: textDirectionToSend,
        autoTextDirection: isAutoTextDirection,  // 自动排版开关
        prompt_content: $('#promptContent').val(),
        use_textbox_prompt: $('#enableTextboxPrompt').prop('checked'),
        textbox_prompt_content: $('#textboxPromptContent').val(),
        use_inpainting: repairSettings.useInpainting,
        use_lama: repairSettings.useLama,
        lamaModel: repairSettings.lamaModel,
        fillColor: $('#fillColor').val(),
        textColor: $('#textColor').val(),
        rotationAngle: 0,  // 全局角度已移除，使用每个气泡独立的旋转角度
        skip_ocr: false,
        skip_translation: false,
        bubble_coords: coordsToUse,
        bubble_angles: anglesToUse,  // 传递角度信息
        ocr_engine: ocr_engine,
        baidu_api_key: ocr_engine === 'baidu_ocr' ? $('#baiduApiKey').val() : null,
        baidu_secret_key: ocr_engine === 'baidu_ocr' ? $('#baiduSecretKey').val() : null,
        baidu_version: ocr_engine === 'baidu_ocr' ? $('#baiduVersion').val() : null,
        baidu_ocr_language: ocr_engine === 'baidu_ocr' ? (() => {
            const baiduOcrLang = state.getBaiduOcrSourceLanguage();
            // 如果百度OCR语言设置为"无"或空，则使用用户选择的源语言
            return (baiduOcrLang === '无' || !baiduOcrLang || baiduOcrLang === '') ? $('#sourceLanguage').val() : baiduOcrLang;
        })() : null, // 添加百度OCR源语言
        ai_vision_provider: ocr_engine === 'ai_vision' ? $('#aiVisionProvider').val() : null,
        ai_vision_api_key: ocr_engine === 'ai_vision' ? $('#aiVisionApiKey').val() : null,
        ai_vision_model_name: ocr_engine === 'ai_vision' ? $('#aiVisionModelName').val() : null,
        ai_vision_ocr_prompt: ocr_engine === 'ai_vision' ? $('#aiVisionOcrPrompt').val() : null,

        strokeEnabled: state.strokeEnabled,
        strokeColor: state.strokeColor,
        strokeWidth: state.strokeWidth,
        
        rpm_limit_translation: state.rpmLimitTranslation,
        rpm_limit_ai_vision_ocr: state.rpmLimitAiVisionOcr,
        use_json_format_translation: state.isTranslateJsonMode,
        use_json_format_ai_vision_ocr: state.isAiVisionOcrJsonMode,
        
        // 调试选项
        showDetectionDebug: state.showDetectionDebug,

        // 合并多轮翻译
        translate_merge: translateMerge,
        chat_session: chatSession
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

                // 简化：只保存核心数据和 bubbleStates
                state.updateCurrentImageProperty('translatedDataURL', 'data:image/png;base64,' + response.translated_image);
                state.updateCurrentImageProperty('cleanImageData', response.clean_image);
                state.updateCurrentImageProperty('bubbleCoords', response.bubble_coords);
                state.updateCurrentImageProperty('bubbleAngles', response.bubble_angles || []);
                state.updateCurrentImageProperty('originalTexts', response.original_texts);
                state.updateCurrentImageProperty('textboxTexts', response.textbox_texts || []);
                state.updateCurrentImageProperty('translationFailed', false);
                state.updateCurrentImageProperty('showOriginal', false);
                state.updateCurrentImageProperty('_lama_inpainted', repairSettings.useLama);
                
                // 使用统一的 bubbleStates 保存所有设置
                // 注意：后端已在首次翻译时计算好字号并保存到 fontSize，不再需要 autoFontSize
                const bubbleStates = bubbleStateModule.createBubbleStatesFromResponse(response, {
                    fontSize: fontSize,
                    fontFamily: params.fontFamily,
                    textDirection: params.textDirection,
                    textColor: params.textColor,
                    fillColor: params.fillColor,
                    inpaintMethod: repairSettings.useLama ? repairSettings.lamaModel : 'solid',
                    strokeEnabled: params.strokeEnabled,
                    strokeColor: params.strokeColor,
                    strokeWidth: params.strokeWidth
                });
                state.updateCurrentImageProperty('bubbleStates', bubbleStates);
                state.updateCurrentImageProperty('bubbleTexts', bubbleStates.map(s => s.translatedText || ""));
                
                // 【重要】保存用户选择的排版方向（包括 'auto'），以便切换图片时恢复
                state.updateCurrentImageProperty('userLayoutDirection', $('#layoutDirection').val());

                // 如果使用了已有坐标，保留状态
                if (usedExistingCoords) {
                    // 不再清除标注状态，确保可以继续使用已有文本框
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
 * 获取页数选择器中选取的张数
 * @returns {number}
 */
function getPageSelectCount() {
  const $pageStart = $("#pageStart")
  const $pageEnd = $("#pageEnd")

  const start = parseFloat($pageStart.val() || 1)
  const end = parseFloat($pageEnd.val() || $pageEnd.attr('max'))
  return end - start + 1
}

/**
 * 翻译所有已加载的图片
 * 按照每张图片的当前状态（包括手动标注框）批量翻译
 */
export function translateAllImages(config) {
    const translateMerge = config?.translate_merge || false
    let chatSession = config?.translate_merge ? window.crypto.randomUUID() : null

    if (state.images.length === 0) {
        ui.showGeneralMessage("请先添加图片", "warning");
        return;
    }

    // 验证翻译配置是否完整
    if (!translationValidator.validateBeforeTranslation('normal')) {
        return;
    }
    
    // 立即显示进度条（移到前面来）
    $("#translationProgressBar").show();
    $("#pauseTranslationButton").show(); // 显示暂停按钮
    ui.updatePauseButton(false); // 初始化为非暂停状态
    ui.updateProgressBar(0, `0/${state.images.length}`);
    ui.showGeneralMessage("批量翻译中...", "info", false, 0); // 显示模态提示，不自动消失

    // 设置批量翻译状态为进行中
    state.setBatchTranslationInProgress(true);
    state.setBatchTranslationPaused(false); // 确保暂停状态重置
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
    // 处理排版方向：如果是 "auto" 则启用自动排版
    const layoutDirectionValue = $('#layoutDirection').val();
    const isAutoTextDirection = layoutDirectionValue === 'auto';
    const textDirection = isAutoTextDirection ? 'vertical' : layoutDirectionValue;
    const useTextboxPrompt = $('#enableTextboxPrompt').prop('checked');
    const textboxPromptContent = $('#textboxPromptContent').val();
    const fillColor = $('#fillColor').val();
    const repairSettings = ui.getRepairSettings(); // ui.js 获取修复设置
    const useInpainting = repairSettings.useInpainting;
    const useLama = repairSettings.useLama;
    const lamaModel = repairSettings.lamaModel;
    const promptContent = $('#promptContent').val();
    const textColor = $('#textColor').val();
    const rotationAngle = 0;  // 全局角度已移除，使用每个气泡独立的旋转角度
    const ocr_engine = $('#ocrEngine').val();
    
    const strokeEnabled = state.strokeEnabled;
    const strokeColor = state.strokeColor;
    const strokeWidth = state.strokeWidth;

    // 百度OCR相关参数
    const baiduApiKey = ocr_engine === 'baidu_ocr' ? $('#baiduApiKey').val() : null;
    const baiduSecretKey = ocr_engine === 'baidu_ocr' ? $('#baiduSecretKey').val() : null;
    const baiduVersion = ocr_engine === 'baidu_ocr' ? $('#baiduVersion').val() : null;
    const baiduOcrLanguage = ocr_engine === 'baidu_ocr' ? state.getBaiduOcrSourceLanguage() : null; // 添加百度OCR源语言
    
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
        $("#pauseTranslationButton").hide(); // 隐藏暂停按钮
        ui.updateButtonStates(); // 恢复按钮状态
        return;
    }
    
    // 检查AI视觉OCR配置
    if (ocr_engine === 'ai_vision' && (!aiVisionApiKey || !aiVisionModelName)) {
        ui.showGeneralMessage("使用AI视觉OCR时必须提供API Key和模型名称", "error");
        state.setBatchTranslationInProgress(false); // 确保错误时重置状态
        $("#translationProgressBar").hide(); // 隐藏进度条
        $("#pauseTranslationButton").hide(); // 隐藏暂停按钮
        ui.updateButtonStates(); // 恢复按钮状态
        return;
    }

    // 在循环外获取一次JSON模式状态
    const aktuellenTranslateJsonMode = state.isTranslateJsonMode;
    const aktuellenAiVisionOcrJsonMode = state.isAiVisionOcrJsonMode;

    // let currentIndex = 0;
    let currentIndex = translateMerge ? parseFloat($('#pageStart').val() || 1) - 1 : 0;
    const totalImages = translateMerge ? parseFloat($('#pageEnd').val() || state.images.length) : state.images.length;
    let failCount = 0; // 记录失败数量

    let customBaseUrlForAll = null;
    if (modelProvider === 'custom_openai') {
        customBaseUrlForAll = $('#customBaseUrl').val().trim();
        if (!customBaseUrlForAll) {
            ui.showGeneralMessage("批量翻译自定义 OpenAI 服务需要填写 Base URL！", "error");
            state.setBatchTranslationInProgress(false); // 确保错误时重置状态
            $("#translationProgressBar").hide(); // 隐藏进度条
            $("#pauseTranslationButton").hide(); // 隐藏暂停按钮
            ui.updateButtonStates(); // 更新按钮状态
            return; // 或返回 Promise.reject
        }
    }

    function processNextImage() {
        // 检查是否暂停
        if (state.isBatchTranslationPaused) {
            // 设置继续翻译的回调
            state.setBatchTranslationResumeCallback(() => {
                ui.updatePauseButton(false);
                ui.showGeneralMessage("继续翻译...", "info", false, 2000);
                processNextImage(); // 继续处理
            });
            ui.showGeneralMessage("翻译已暂停", "warning", false, 0);
            return; // 暂停，等待用户继续
        }
        
        if (currentIndex >= totalImages) {
            ui.updateProgressBar(100, `${totalImages - failCount}/${totalImages}`);
            $("#pauseTranslationButton").hide(); // 隐藏暂停按钮
            $(".message.info, .message.warning").fadeOut(300, function() { $(this).remove(); }); // 移除加载消息
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
            
            // 批量翻译完成后，刷新当前显示的图片
            // const originalImageIndex = state.currentImageIndex; // 获取当前显示的图片索引
            // 如果选择合并翻译，使用图片选择器的数据，否则使用当前的图片索引
            const originalImageIndex = translateMerge ? parseFloat($("pageStart").val() || 1) - 1 : state.currentImageIndex;
            if (originalImageIndex >= 0) {
                // 重新显示当前图片，触发UI刷新显示最新渲染结果
                switchImage(originalImageIndex);
            }
            return;
        }

        ui.updateProgressBar((currentIndex / totalImages) * 100, `${currentIndex}/${totalImages}`);
        ui.showTranslatingIndicator(currentIndex);
        const imageData = state.images[currentIndex]; // 获取当前循环索引对应的图片数据

        // --- 关键修改：检查并使用已有的坐标和角度（优先级：savedManualCoords > bubbleCoords > 自动检测）---
        // 角度优先从 bubbleStates 提取（唯一状态来源），回退到 bubbleAngles（检测结果）
        let coordsToUse = null;
        let anglesToUse = null;
        let usedExistingCoordsThisImage = false;
        
        const extractAngles = (states) => states && states.length > 0 ? states.map(s => s.rotationAngle || 0) : null;
        
        if (imageData.savedManualCoords && imageData.savedManualCoords.length > 0) {
            coordsToUse = imageData.savedManualCoords;
            anglesToUse = imageData.savedManualAngles || null;
            usedExistingCoordsThisImage = true;
            console.log(`批量翻译图片 ${currentIndex}: 将使用 ${coordsToUse.length} 个已保存的手动标注框。`);
        } else if (imageData.bubbleCoords && imageData.bubbleCoords.length > 0) {
            coordsToUse = imageData.bubbleCoords;
            anglesToUse = extractAngles(imageData.bubbleStates) || imageData.bubbleAngles || null;
            usedExistingCoordsThisImage = true;
            console.log(`批量翻译图片 ${currentIndex}: 将使用 ${coordsToUse.length} 个已有的文本框。`);
        } else {
            console.log(`批量翻译图片 ${currentIndex}: 未找到已有文本框，将进行自动检测。`);
        }
        // --- 调用api配置查章节chat_session
        if (chapterCheckIndexHasStartActive(currentIndex)) {
            chatSession = window.crypto.randomUUID()
            console.log('刷新会话Session,新Session -> ', chatSession)
        }
        // ----------------------------------------------

        const data = { // 准备 API 请求数据
            image: imageData.originalDataURL.split(',')[1],
            target_language: targetLanguage, source_language: sourceLanguage,
            fontSize: fontSize, autoFontSize: isAutoFontSize,
            api_key: apiKey, model_name: modelName, model_provider: modelProvider,
            fontFamily: fontFamily, textDirection: textDirection,
            autoTextDirection: isAutoTextDirection,  // 自动排版开关
            prompt_content: promptContent, use_textbox_prompt: useTextboxPrompt,
            textbox_prompt_content: textboxPromptContent, use_inpainting: useInpainting,
            use_lama: useLama,
            lamaModel: lamaModel,
            fillColor: fillColor,
            textColor: textColor,
            rotationAngle: rotationAngle,
            skip_translation: false, skip_ocr: false, remove_only: false,
            bubble_coords: coordsToUse, // 传递坐标
            bubble_angles: anglesToUse, // 传递角度
            ocr_engine: ocr_engine,
            baidu_api_key: baiduApiKey,
            baidu_secret_key: baiduSecretKey,
            baidu_version: baiduVersion,
            baidu_ocr_language: baiduOcrLanguage, // 添加百度OCR源语言
            ai_vision_provider: aiVisionProvider,
            ai_vision_api_key: aiVisionApiKey,
            ai_vision_model_name: aiVisionModelName,
            ai_vision_ocr_prompt: aiVisionOcrPrompt,
            custom_base_url: customBaseUrlForAll,
            
            strokeEnabled: strokeEnabled,
            strokeColor: strokeColor,
            strokeWidth: strokeWidth,
            
            use_json_format_translation: aktuellenTranslateJsonMode,
            use_json_format_ai_vision_ocr: aktuellenAiVisionOcrJsonMode,
            
            // 调试选项
            showDetectionDebug: state.showDetectionDebug,

            // 合并多轮翻译
            translate_merge: translateMerge,
            chat_session: chatSession,
        };

        // --- 核心修改：直接调用 API，而不是 translateCurrentImage ---
        api.translateImageApi(data)
            .then(response => {
                ui.hideTranslatingIndicator(currentIndex);

                // 简化：只保存核心数据和 bubbleStates
                state.updateImagePropertyByIndex(currentIndex, 'translatedDataURL', 'data:image/png;base64,' + response.translated_image);
                state.updateImagePropertyByIndex(currentIndex, 'cleanImageData', response.clean_image);
                state.updateImagePropertyByIndex(currentIndex, 'bubbleCoords', response.bubble_coords);
                state.updateImagePropertyByIndex(currentIndex, 'bubbleAngles', response.bubble_angles || []);
                state.updateImagePropertyByIndex(currentIndex, 'originalTexts', response.original_texts);
                state.updateImagePropertyByIndex(currentIndex, 'textboxTexts', response.textbox_texts || []);
                state.updateImagePropertyByIndex(currentIndex, 'translationFailed', false);
                state.updateImagePropertyByIndex(currentIndex, 'showOriginal', false);
                state.updateImagePropertyByIndex(currentIndex, '_lama_inpainted', useLama);
                
                // 使用统一的 bubbleStates 保存所有设置
                // 注意：后端已在首次翻译时计算好字号并保存到 fontSize
                const batchBubbleStates = bubbleStateModule.createBubbleStatesFromResponse(response, {
                    fontSize: fontSize,
                    fontFamily: fontFamily,
                    textDirection: textDirection,
                    textColor: textColor,
                    fillColor: fillColor,
                    inpaintMethod: useLama ? lamaModel : 'solid',
                    strokeEnabled: strokeEnabled,
                    strokeColor: strokeColor,
                    strokeWidth: strokeWidth
                });
                state.updateImagePropertyByIndex(currentIndex, 'bubbleStates', batchBubbleStates);
                state.updateImagePropertyByIndex(currentIndex, 'bubbleTexts', batchBubbleStates.map(s => s.translatedText || ""));
                
                // 【重要】保存用户选择的排版方向（包括 'auto'）
                state.updateImagePropertyByIndex(currentIndex, 'userLayoutDirection', layoutDirectionValue);

                if (usedExistingCoordsThisImage) {
                    state.updateImagePropertyByIndex(currentIndex, 'hasUnsavedChanges', false);
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
                // 章节检查结束标志
                if (chapterCheckIndexHasEndActive(currentIndex)) {
                    chatSession = null
                }
                currentIndex++;
                processNextImage(); // 处理下一张图片
            });
        // --- 结束核心修改 ---
    }
    processNextImage(); // 开始处理第一张图片
}

/**
 * 检查索引图像章节开始标记是否激活
 * @param index
 * @returns boolean
 */
function chapterCheckIndexHasStartActive(index) {
    const wrapper = $('.thumbnail-wrapper')[index]
    if (!wrapper) return false
    const mark = $(wrapper).find('.mark-left')[0]
    if (!mark) return false
    return $(mark).hasClass('active')
}

/**
 * 检查索引图像章节结束标记是否激活
 * @param index
 * @returns boolean
 */
function chapterCheckIndexHasEndActive(index) {
    const wrapper = $('.thumbnail-wrapper')[index]
    if (!wrapper) return false
    const mark = $(wrapper).find('.mark-right')[0]
    if (!mark) return false
    return $(mark).hasClass('active')
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
 * 下载所有翻译后的图片（逐张上传到后端，避免大数据量导致的字符串长度限制）
 */
export async function downloadAllImages() {
    const selectedFormat = $('#downloadFormat').val();

    // 立即显示进度条
    $("#translationProgressBar").show();
    ui.updateProgressBar(0, "准备下载...");

    // --- 显示提示信息 ---
    ui.showGeneralMessage("下载中...处理可能需要一定时间，请耐心等待...", "info", false, 0);
    ui.showDownloadingMessage(true); // 显示下载中并禁用按钮

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
        
        // 收集需要发送的图像数据（只记录索引和类型，不一次性收集所有数据）
        const imageInfoList = [];
        let translatedCount = 0;
        let originalCount = 0;
        
        ui.updateProgressBar(5, "检查图片数据...");
        
        for (let i = 0; i < allImages.length; i++) {
            const imgData = allImages[i];
            // 优先使用翻译后的图片，如果没有则使用原始图片
            if (imgData.translatedDataURL) {
                imageInfoList.push({ index: i, type: 'translated' });
                translatedCount++;
            } else if (imgData.originalDataURL) {
                imageInfoList.push({ index: i, type: 'original' });
                originalCount++;
            }
        }
        
        if (imageInfoList.length === 0) {
            ui.showGeneralMessage("没有可下载的图片", "warning");
            $("#translationProgressBar").hide();
            $(".message.info").fadeOut(300, function() { $(this).remove(); });
            ui.showDownloadingMessage(false);
            return;
        }
        
        // 步骤1: 创建下载会话
        ui.updateProgressBar(10, "创建下载会话...");
        
        let sessionId;
        try {
            const sessionResponse = await $.ajax({
                url: '/api/download_start_session',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ total_images: imageInfoList.length })
            });
            
            if (!sessionResponse.success || !sessionResponse.session_id) {
                throw new Error(sessionResponse.error || '创建会话失败');
            }
            sessionId = sessionResponse.session_id;
        } catch (e) {
            throw new Error(`创建下载会话失败: ${e.message || e}`);
        }
        
        // 步骤2: 逐张上传图片
        const totalImages = imageInfoList.length;
        let uploadedCount = 0;
        let failedCount = 0;
        
        for (let i = 0; i < imageInfoList.length; i++) {
            const info = imageInfoList[i];
            const imgData = allImages[info.index];
            const imageDataURL = info.type === 'translated' ? imgData.translatedDataURL : imgData.originalDataURL;
            
            // 更新进度条
            const progress = 10 + (i / totalImages) * 70; // 10% - 80%
            ui.updateProgressBar(progress, `上传图片 ${i + 1}/${totalImages}...`);
            
            try {
                const uploadResponse = await $.ajax({
                    url: '/api/download_upload_image',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({
                        session_id: sessionId,
                        image_index: i,
                        image_data: imageDataURL
                    })
                });
                
                if (uploadResponse.success) {
                    uploadedCount++;
                } else {
                    console.error(`上传图片 ${i} 失败:`, uploadResponse.error);
                    failedCount++;
                }
            } catch (e) {
                console.error(`上传图片 ${i} 出错:`, e);
                failedCount++;
            }
        }
        
        if (uploadedCount === 0) {
            throw new Error('所有图片上传失败');
        }
        
        // 步骤3: 请求打包
        ui.updateProgressBar(85, "打包文件...");
        
        const finalizeResponse = await $.ajax({
            url: '/api/download_finalize',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                session_id: sessionId,
                format: selectedFormat
            })
        });
        
        if (!finalizeResponse.success || !finalizeResponse.file_id) {
            throw new Error(finalizeResponse.error || '打包失败');
        }
        
        // 步骤4: 触发下载
        ui.updateProgressBar(95, "准备下载...");
        
        const downloadUrl = `/api/download_file/${finalizeResponse.file_id}?format=${finalizeResponse.format}`;
        const link = document.createElement('a');
        link.href = downloadUrl;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        ui.updateProgressBar(100, "下载已开始");
        
        // 更新下载成功信息
        let successMessage = `已成功处理 ${uploadedCount} 张图片`;
        if (failedCount > 0) {
            successMessage += `（${failedCount} 张失败）`;
        }
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
        
    } catch (e) {
        console.error("下载所有图片时出错:", e);
        ui.showGeneralMessage(`下载失败: ${e.message || e}`, "error");
    } finally {
        setTimeout(() => {
            $("#translationProgressBar").hide();
            $(".message.info").fadeOut(300, function() { $(this).remove(); });
        }, 2000);
        ui.showDownloadingMessage(false);
    }
}

/**
 * 将当前设置应用到所有图片（简化版）
 * 核心逻辑：从当前图片的 bubbleStates[0] 读取设置，应用到所有图片的 bubbleStates
 */
export async function applySettingsToAll() {
    const currentImage = state.getCurrentImage();
    if (!currentImage || !currentImage.bubbleStates?.[0]) {
        ui.showGeneralMessage("请先选择一张已翻译的图片", "warning");
        return;
    }
    if (state.images.length <= 1) {
        ui.showGeneralMessage("只有一张图片，无需应用到全部", "info");
        return;
    }

    // 从当前图片的第一个气泡读取设置
    const source = currentImage.bubbleStates[0];
    
    // 根据复选框选择要应用的参数
    const settingsToApply = {};
    if ($('#apply_fontSize').is(':checked')) {
        settingsToApply.fontSize = source.fontSize;
    }
    if ($('#apply_fontFamily').is(':checked')) {
        settingsToApply.fontFamily = source.fontFamily;
    }
    if ($('#apply_textDirection').is(':checked')) {
        settingsToApply.textDirection = source.textDirection === 'auto' ? 'vertical' : source.textDirection;
    }
    if ($('#apply_textColor').is(':checked')) {
        settingsToApply.textColor = source.textColor;
    }
    if ($('#apply_fillColor').is(':checked')) {
        settingsToApply.fillColor = source.fillColor;
    }
    if ($('#apply_strokeEnabled').is(':checked')) {
        settingsToApply.strokeEnabled = source.strokeEnabled;
    }
    if ($('#apply_strokeColor').is(':checked')) {
        settingsToApply.strokeColor = source.strokeColor;
    }
    if ($('#apply_strokeWidth').is(':checked')) {
        settingsToApply.strokeWidth = source.strokeWidth;
    }
    
    // 检查是否至少选择了一个参数
    if (Object.keys(settingsToApply).length === 0) {
        ui.showGeneralMessage("请至少选择一个要应用的参数", "warning");
        return;
    }

    ui.showLoading("应用设置到所有图片...");
    const originalImageIndex = state.currentImageIndex;
    
    try {
        for (let i = 0; i < state.images.length; i++) {
            const img = state.images[i];
            if (!img.translatedDataURL || !img.bubbleStates) continue;
            
            ui.updateLoadingMessage(`应用设置到图片 ${i + 1}/${state.images.length}...`);
            
            // 更新所有气泡的设置
            img.bubbleStates = img.bubbleStates.map(bs => ({
                ...bs,
                ...settingsToApply
            }));
            img.bubbleTexts = img.bubbleStates.map(bs => bs.translatedText || bs.text || "");
            
            // 切换到目标图片并重渲染
            state.setCurrentImageIndex(i);
            const editMode = await import('./edit_mode.js');
            await editMode.reRenderFullImage(false, true);
            
            await new Promise(r => setTimeout(r, 100));
        }
        
        switchImage(originalImageIndex);
        ui.hideLoading();
        $("#applySettingsDropdown").removeClass('show'); // 关闭下拉菜单
        ui.showGeneralMessage("设置已应用到所有图片", "success");
    } catch (error) {
        console.error("应用设置时出错:", error);
        ui.hideLoading();
        $("#applySettingsDropdown").removeClass('show'); // 关闭下拉菜单
        ui.showGeneralMessage("应用设置时出错: " + error.message, "error");
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
    let anglesToUse = null;
    let usedExistingCoords = false;
    // --- 关键修改：检查并使用已有的坐标和角度（优先级：savedManualCoords > bubbleCoords > 自动检测）---
    // 角度优先从 bubbleStates 提取（唯一状态来源），回退到 bubbleAngles（检测结果）
    const extractAngles = (states) => states && states.length > 0 ? states.map(s => s.rotationAngle || 0) : null;
    
    if (currentImage.savedManualCoords && currentImage.savedManualCoords.length > 0) {
        coordsToUse = currentImage.savedManualCoords;
        anglesToUse = currentImage.savedManualAngles || null;
        usedExistingCoords = true;
        console.log(`消除文字 ${state.currentImageIndex}: 将使用 ${coordsToUse.length} 个已保存的手动标注框。`);
        ui.showGeneralMessage("检测到手动标注框，将优先使用...", "info", false, 3000);
    } else if (currentImage.bubbleCoords && currentImage.bubbleCoords.length > 0) {
        coordsToUse = currentImage.bubbleCoords;
        anglesToUse = extractAngles(currentImage.bubbleStates) || currentImage.bubbleAngles || null;
        usedExistingCoords = true;
        console.log(`消除文字 ${state.currentImageIndex}: 将使用 ${coordsToUse.length} 个已有的文本框。`);
        ui.showGeneralMessage("使用已有的文本框消除文字...", "info", false, 3000);
    } else {
        console.log(`消除文字 ${state.currentImageIndex}: 未找到已有文本框，将进行自动检测。`);
    }
    // ------------------------------------------

    // 处理排版方向：消除文字模式下如果是 'auto' 则默认使用 'vertical'
    const layoutDirectionValue = $('#layoutDirection').val();
    const textDirectionForRemove = layoutDirectionValue === 'auto' ? 'vertical' : layoutDirectionValue;
    
    const params = {
        image: currentImage.originalDataURL.split(',')[1],
        target_language: $('#targetLanguage').val(),
        source_language: $('#sourceLanguage').val(),
        fontSize: fontSize,
        autoFontSize: isAutoFontSize,
        api_key: null,  // 仅消除文字模式不需要API Key
        model_name: null,
        model_provider: null,
        fontFamily: $('#fontFamily').val(),
        textDirection: textDirectionForRemove,
        prompt_content: null,
        use_textbox_prompt: false,
        textbox_prompt_content: null,
        use_inpainting: repairSettings.useInpainting,
        use_lama: repairSettings.useLama,
        lamaModel: repairSettings.lamaModel,
        fillColor: $('#fillColor').val(),
        textColor: $('#textColor').val(),
        rotationAngle: 0,  // 默认旋转角度
        skip_ocr: false,
        skip_translation: true,  // 跳过翻译
        remove_only: true,  // 标记仅消除文字模式
        bubble_coords: coordsToUse,
        bubble_angles: anglesToUse,  // 传递角度
        ocr_engine: ocr_engine,
        baidu_api_key: ocr_engine === 'baidu_ocr' ? $('#baiduApiKey').val() : null,
        baidu_secret_key: ocr_engine === 'baidu_ocr' ? $('#baiduSecretKey').val() : null,
        baidu_version: ocr_engine === 'baidu_ocr' ? $('#baiduVersion').val() : null,
        baidu_ocr_language: ocr_engine === 'baidu_ocr' ? state.getBaiduOcrSourceLanguage() : null, // 添加百度OCR源语言
        ai_vision_provider: ocr_engine === 'ai_vision' ? $('#aiVisionProvider').val() : null,
        ai_vision_api_key: ocr_engine === 'ai_vision' ? $('#aiVisionApiKey').val() : null,
        ai_vision_model_name: ocr_engine === 'ai_vision' ? $('#aiVisionModelName').val() : null,
        ai_vision_ocr_prompt: ocr_engine === 'ai_vision' ? $('#aiVisionOcrPrompt').val() : null,
        // 描边参数
        strokeEnabled: state.strokeEnabled,
        strokeColor: state.strokeColor,
        strokeWidth: state.strokeWidth,
        // 调试选项
        showDetectionDebug: state.showDetectionDebug
    };

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
                ui.hideTranslatingIndicator(state.currentImageIndex);

                // 简化：只保存核心数据和 bubbleStates
                currentImage.translatedDataURL = 'data:image/png;base64,' + response.translated_image;
                currentImage.cleanImageData = response.clean_image;
                currentImage.bubbleCoords = response.bubble_coords || [];
                currentImage.bubbleAngles = response.bubble_angles || [];
                currentImage.originalTexts = response.original_texts || [];
                currentImage.textboxTexts = response.textbox_texts || [];
                currentImage.translationFailed = false;
                currentImage.showOriginal = false;
                currentImage._lama_inpainted = params.use_lama;
                
                // 使用统一的 bubbleStates
                const layoutDir = $('#layoutDirection').val();
                const bubbleStates = bubbleStateModule.createBubbleStatesFromResponse(response, {
                    fontSize: fontSize,
                    fontFamily: fontFamily,
                    textDirection: layoutDir === 'auto' ? 'vertical' : layoutDir,
                    textColor: params.textColor,
                    fillColor: params.fillColor,
                    inpaintMethod: repairSettings.useLama ? repairSettings.lamaModel : 'solid',
                    strokeEnabled: state.strokeEnabled,
                    strokeColor: state.strokeColor,
                    strokeWidth: state.strokeWidth
                });
                currentImage.bubbleStates = bubbleStates;
                currentImage.bubbleTexts = bubbleStates.map(s => s.translatedText || "");
                
                // 【重要】保存用户选择的排版方向（包括 'auto'）
                currentImage.userLayoutDirection = layoutDir;
                
                if (usedExistingCoords) {
                    ui.renderThumbnails();
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

// initializeOcrEngineSettings 已移除 - OCR设置现在由设置模态框处理

/**
 * 消除所有图片中的文字
 * 按照每张图片的当前状态（包括手动标注框）批量消除文字
 */
export function removeAllBubblesText() {
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
    // 处理排版方向：消除文字模式下如果是 'auto' 则默认使用 'vertical'
    const layoutDirectionValue = $('#layoutDirection').val();
    const textDirection = layoutDirectionValue === 'auto' ? 'vertical' : layoutDirectionValue;
    const fillColor = $('#fillColor').val();
    const repairSettings = ui.getRepairSettings(); // ui.js 获取修复设置
    const useInpainting = repairSettings.useInpainting;
    const useLama = repairSettings.useLama;
    const lamaModel = repairSettings.lamaModel;
    const textColor = $('#textColor').val();
    const rotationAngle = 0;  // 全局角度已移除，使用每个气泡独立的旋转角度
    const ocr_engine = $('#ocrEngine').val();

    // 百度OCR相关参数
    const baiduApiKey = ocr_engine === 'baidu_ocr' ? $('#baiduApiKey').val() : null;
    const baiduSecretKey = ocr_engine === 'baidu_ocr' ? $('#baiduSecretKey').val() : null;
    const baiduVersion = ocr_engine === 'baidu_ocr' ? $('#baiduVersion').val() : 'standard';
    // 如果百度OCR语言设置为"无"或空，则使用用户选择的源语言
    const sourceLang = $('#sourceLanguage').val();
    const baiduOcrLang = state.getBaiduOcrSourceLanguage();
    const baiduOcrLanguage = (baiduOcrLang === '无' || !baiduOcrLang || baiduOcrLang === '') ? sourceLang : baiduOcrLang;
    
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
            return;
        }

        ui.updateProgressBar((currentIndex / totalImages) * 100, `${currentIndex}/${totalImages}`);
        ui.showTranslatingIndicator(currentIndex);
        const imageData = state.images[currentIndex];

        // --- 检查并使用已有的坐标和角度（优先级：savedManualCoords > bubbleCoords > 自动检测）---
        // 角度优先从 bubbleStates 提取（唯一状态来源），回退到 bubbleAngles（检测结果）
        let coordsToUse = null;
        let anglesToUse = null;
        let usedExistingCoordsThisImage = false;
        
        const extractAngles = (states) => states && states.length > 0 ? states.map(s => s.rotationAngle || 0) : null;
        
        if (imageData.savedManualCoords && imageData.savedManualCoords.length > 0) {
            coordsToUse = imageData.savedManualCoords;
            anglesToUse = imageData.savedManualAngles || null;
            usedExistingCoordsThisImage = true;
            console.log(`批量消除文字 ${currentIndex}: 将使用 ${coordsToUse.length} 个已保存的手动标注框。`);
        } else if (imageData.bubbleCoords && imageData.bubbleCoords.length > 0) {
            coordsToUse = imageData.bubbleCoords;
            anglesToUse = extractAngles(imageData.bubbleStates) || imageData.bubbleAngles || null;
            usedExistingCoordsThisImage = true;
            console.log(`批量消除文字 ${currentIndex}: 将使用 ${coordsToUse.length} 个已有的文本框。`);
        } else {
            console.log(`批量消除文字 ${currentIndex}: 未找到已有文本框，将进行自动检测。`);
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
            lamaModel: lamaModel,
            fillColor: fillColor,
            textColor: textColor,
            rotationAngle: rotationAngle,
            skip_translation: true, // 关键设置：跳过翻译
            skip_ocr: false,
            remove_only: true, // 关键设置：仅消除文字
            bubble_coords: coordsToUse, // 传递坐标
            bubble_angles: anglesToUse, // 传递角度
            ocr_engine: ocr_engine,
            baidu_api_key: baiduApiKey,
            baidu_secret_key: baiduSecretKey,
            baidu_version: baiduVersion,
            baidu_ocr_language: baiduOcrLanguage, // 使用根据条件确定的语言设置
            ai_vision_provider: aiVisionProvider,
            ai_vision_api_key: aiVisionApiKey,
            ai_vision_model_name: aiVisionModelName,
            ai_vision_ocr_prompt: aiVisionOcrPrompt,
            use_json_format_translation: false,
            use_json_format_ai_vision_ocr: aktuellenAiVisionOcrJsonMode,
            // 描边参数
            strokeEnabled: state.strokeEnabled,
            strokeColor: state.strokeColor,
            strokeWidth: state.strokeWidth,
            // 调试选项
            showDetectionDebug: state.showDetectionDebug
        };

        // --- 调用API执行文字消除 ---
        api.translateImageApi(data)
            .then(response => {
                ui.hideTranslatingIndicator(currentIndex);

                // 简化：只保存核心数据和 bubbleStates
                state.updateImagePropertyByIndex(currentIndex, 'translatedDataURL', 'data:image/png;base64,' + response.translated_image);
                state.updateImagePropertyByIndex(currentIndex, 'cleanImageData', response.clean_image);
                state.updateImagePropertyByIndex(currentIndex, 'bubbleCoords', response.bubble_coords || []);
                state.updateImagePropertyByIndex(currentIndex, 'bubbleAngles', response.bubble_angles || []);
                state.updateImagePropertyByIndex(currentIndex, 'originalTexts', response.original_texts || []);
                state.updateImagePropertyByIndex(currentIndex, 'textboxTexts', response.textbox_texts || []);
                state.updateImagePropertyByIndex(currentIndex, 'translationFailed', false);
                state.updateImagePropertyByIndex(currentIndex, 'showOriginal', false);
                state.updateImagePropertyByIndex(currentIndex, '_lama_inpainted', useLama);
                
                // 使用统一的 bubbleStates
                const removeBubbleStates = bubbleStateModule.createBubbleStatesFromResponse(response, {
                    fontSize: fontSize,
                    fontFamily: fontFamily,
                    textDirection: textDirection,
                    textColor: textColor,
                    fillColor: fillColor,
                    inpaintMethod: useLama ? lamaModel : 'solid',
                    strokeEnabled: state.strokeEnabled,
                    strokeColor: state.strokeColor,
                    strokeWidth: state.strokeWidth
                });
                state.updateImagePropertyByIndex(currentIndex, 'bubbleStates', removeBubbleStates);
                state.updateImagePropertyByIndex(currentIndex, 'bubbleTexts', removeBubbleStates.map(s => s.translatedText || ""));
                
                // 【重要】保存用户选择的排版方向（包括 'auto'）
                state.updateImagePropertyByIndex(currentIndex, 'userLayoutDirection', layoutDirectionValue);

                if (usedExistingCoordsThisImage) {
                    state.updateImagePropertyByIndex(currentIndex, 'hasUnsavedChanges', false);
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
            if (image.bubbleStates && Array.isArray(image.bubbleStates) && 
                bubbleIndex < image.bubbleStates.length && 
                image.bubbleStates[bubbleIndex]) {
                const bubbleDir = image.bubbleStates[bubbleIndex].textDirection;
                // 确保不传递 'auto'
                textDirection = (bubbleDir && bubbleDir !== 'auto') ? bubbleDir : textDirection;
            } else if (image.layoutDirection && image.layoutDirection !== 'auto') {
                // 如果没有特定气泡设置，使用图片整体布局方向（但不使用 'auto'）
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
            const currentFontFamily = $('#fontFamily').val();
            const currentTextColor = $('#textColor').val();
            const currentFillColor = $('#fillColor').val();
            const currentRotationAngle = 0;  // 全局角度已移除，使用每个气泡独立的旋转角度
            
            ui.updateProgressBar(20, "开始更新图片...");
            
            // 遍历导入的数据
            const totalImages = importedData.length;
            let processedImages = 0;

            // 创建一个队列，用于存储所有渲染任务
            const renderTasks = [];
            
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
                    const original = bubbleData.original;
                    const translated = bubbleData.translated;
                    // 获取排版方向，确保不是 'auto'
                    const rawDir = bubbleData.textDirection;
                    const textDirection = (rawDir && rawDir !== 'auto') ? rawDir : 'vertical';
                    
                    // 确保数组索引存在
                    while (image.bubbleTexts.length <= bubbleIndex) {
                        image.bubbleTexts.push("");
                    }
                    while (image.originalTexts.length <= bubbleIndex) {
                        image.originalTexts.push("");
                    }
                    
                    // 更新文本
                    if (original) image.originalTexts[bubbleIndex] = original;
                    if (translated) image.bubbleTexts[bubbleIndex] = translated;
                    
                    // 更新气泡设置
                    if (!image.bubbleStates) image.bubbleStates = [];
                    while (image.bubbleStates.length <= bubbleIndex) {
                        image.bubbleStates.push(null);
                    }
                    
                    if (!image.bubbleStates[bubbleIndex]) {
                        // 优先使用检测到的角度
                        const detectedAngle = (image.bubbleAngles && image.bubbleAngles[bubbleIndex]) || currentRotationAngle;
                        // 计算自动排版方向
                        let autoDir = textDirection;
                        if (image.bubbleCoords && image.bubbleCoords[bubbleIndex] && image.bubbleCoords[bubbleIndex].length >= 4) {
                            const [x1, y1, x2, y2] = image.bubbleCoords[bubbleIndex];
                            autoDir = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal';
                        }
                        image.bubbleStates[bubbleIndex] = {
                            translatedText: translated || "",
                            fontSize: currentFontSize,
                            fontFamily: currentFontFamily,
                            textDirection: textDirection,
                            autoTextDirection: autoDir,  // 自动检测的排版方向
                            textColor: currentTextColor,
                            fillColor: currentFillColor,
                            rotationAngle: detectedAngle
                        };
                    } else {
                        // 更新现有设置中的文本和排版方向
                        image.bubbleStates[bubbleIndex].translatedText = translated || "";
                        image.bubbleStates[bubbleIndex].textDirection = textDirection;
                    }
                    
                    imageUpdated = true;
                    updatedBubbles++;
                }
                
                // 确保 bubbleTexts 与 bubbleStates 同步
                if (imageUpdated && image.bubbleStates) {
                    image.bubbleTexts = image.bubbleStates.map(bs => bs.translatedText || "");
                }
                
                if (imageUpdated) {
                    updatedImages++;
                    
                    // 不切换图片，而是将渲染任务添加到队列中
                    if (image.translatedDataURL) {
                        renderTasks.push(async () => {
                            // 创建一个离屏画布，避免切换图片
                            const tempImage = await loadImage(image.originalDataURL);
                            const canvas = document.createElement('canvas');
                            canvas.width = tempImage.width;
                            canvas.height = tempImage.height;
                            
                            // 借用edit_mode.js中的渲染逻辑，但不切换图片
                            const editMode = await import('./edit_mode.js');
                            
                            // 保存当前索引
                            const currentIndex = state.currentImageIndex;
                            
                            // 临时切换到目标图片（但不更新UI）
                            state.setCurrentImageIndex(imageIndex);
                            
                            try {
                                // 重新渲染图片
                                await editMode.reRenderFullImage(false, true); // 传入silentMode=true参数，表示静默渲染
                                
                                // 图片已在reRenderFullImage中更新到state中
                                console.log(`已完成图片 ${imageIndex} 的渲染`);
                            } finally {
                                // 恢复原始索引（但不更新UI）
                                state.setCurrentImageIndex(currentIndex);
                            }
                        });
                    }
                }
            }
            
            // 开始执行渲染队列
            ui.updateProgressBar(90, "开始渲染图片...");
            ui.showGeneralMessage("正在渲染图片，请稍候...", "info", false);
            
            // 执行所有渲染任务
            for (let i = 0; i < renderTasks.length; i++) {
                ui.updateProgressBar(90 + (i / renderTasks.length * 10), `渲染图片 ${i+1}/${renderTasks.length}`);
                await renderTasks[i]();
            }
            
            ui.updateProgressBar(100, "完成图片更新");
            
            // 全部导入完成后，回到最初的图片
            switchImage(originalImageIndex);
            
            // 更新UI
            ui.renderThumbnails();
            
            // 显示导入结果
            ui.showGeneralMessage(`导入成功！更新了${updatedImages}张图片中的${updatedBubbles}个气泡文本`, "success");
            
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
