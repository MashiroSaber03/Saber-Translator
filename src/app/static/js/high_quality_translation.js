/**
 * 高质量翻译模式 - 利用AI上下文增强翻译质量
 */
import * as api from './api.js';
import * as state from './state.js';
import * as ui from './ui.js';
import * as translationValidator from './translation_validator.js';
import * as bubbleStateModule from './bubble_state.js';

// 保存翻译状态
let isHqTranslationInProgress = false;
let currentJsonData = null;
let allImageBase64 = [];
let allBatchResults = [];
// 保存开始翻译前的文本样式设置
let savedFontFamily = null;
let savedFontSize = null;
let savedAutoFontSize = false;
let savedTextDirection = null;
let savedAutoTextDirection = false;  // 自动排版开关
let savedFillColor = null;
let savedTextColor = null;
let savedRotationAngle = 0;
let savedStrokeEnabled = true;   // 新增: 保存描边启用状态
let savedStrokeColor = '#FFFFFF'; // 新增: 保存描边颜色
let savedStrokeWidth = 3;        // 新增: 保存描边宽度

/**
 * 开始高质量翻译流程
 */
export async function startHqTranslation() {
    // 检查是否有图片
    if (state.images.length === 0) {
        ui.showGeneralMessage("请先添加图片", "warning");
        return;
    }
    
    // 检查是否正在进行其他批量操作
    if (state.isBatchTranslationInProgress) {
        ui.showGeneralMessage("请等待当前批量操作完成", "warning");
        return;
    }
    
    // 验证高质量翻译配置是否完整
    if (!translationValidator.validateBeforeTranslation('hq')) {
        return;
    }
    
    // 从state获取配置参数
    const provider = state.hqTranslateProvider;
    const apiKey = state.hqApiKey;
    const modelName = state.hqModelName;
    const customBaseUrl = state.hqCustomBaseUrl;
    
    // 保存当前选择的所有文本样式设置，用于翻译过程中保持一致
    savedFontFamily = $('#fontFamily').val();
    savedFontSize = parseInt($('#fontSize').val()) || state.defaultFontSize;
    savedAutoFontSize = $('#autoFontSize').prop('checked');
    // 处理排版方向：如果是 "auto" 则启用自动排版
    const layoutDirectionValue = $('#layoutDirection').val();
    savedAutoTextDirection = layoutDirectionValue === 'auto';
    savedTextDirection = savedAutoTextDirection ? 'vertical' : layoutDirectionValue;
    savedFillColor = $('#fillColor').val();
    savedTextColor = $('#textColor').val();
    savedRotationAngle = 0;  // 全局角度已移除，使用每个气泡独立的旋转角度
    savedStrokeEnabled = state.strokeEnabled;   // 新增: 保存描边启用状态
    savedStrokeColor = state.strokeColor;       // 新增: 保存描边颜色
    savedStrokeWidth = state.strokeWidth;       // 新增: 保存描边宽度
    console.log("高质量翻译前保存的文本样式设置:", {
        fontFamily: savedFontFamily,
        fontSize: savedFontSize,
        autoFontSize: savedAutoFontSize,
        textDirection: savedTextDirection,
        autoTextDirection: savedAutoTextDirection,
        fillColor: savedFillColor,
        textColor: savedTextColor,
        rotationAngle: savedRotationAngle,
        strokeEnabled: savedStrokeEnabled,
        strokeColor: savedStrokeColor,
        strokeWidth: savedStrokeWidth
    });
    
    // 立即显示进度条
    $("#translationProgressBar").show();
    ui.updateProgressBar(0, '准备翻译...');
    ui.showGeneralMessage("步骤1/4: 消除所有图片文字...", "info", false);
    
    // 设置翻译状态
    isHqTranslationInProgress = true;
    state.setBatchTranslationInProgress(true);
    ui.updateButtonStates();
    
    try {
        // 1. 消除所有图片文字
        await removeAllImagesText();
        
        // 2. 导出文本为JSON
        ui.showGeneralMessage("步骤2/4: 导出文本数据...", "info", false);
        ui.updateProgressBar(25, '导出文本数据...');
        currentJsonData = exportTextToJson();
        if (!currentJsonData) {
            throw new Error("导出文本失败");
        }
        
        // 3. 收集所有图片的Base64数据
        ui.showGeneralMessage("步骤3/4: 准备图片数据...", "info", false);
        ui.updateProgressBar(40, '准备图片数据...');
        allImageBase64 = collectAllImageBase64();
        
        // 4. 分批发送给AI翻译
        ui.showGeneralMessage("步骤4/4: 发送到AI进行翻译...", "info", false);
        ui.updateProgressBar(50, '开始发送到AI...');
        
        // 从state获取参数
        const batchSize = state.hqBatchSize;
        const sessionResetFrequency = state.hqSessionReset;
        const rpmLimit = state.hqRpmLimit;
        const lowReasoning = state.hqLowReasoning;
        const prompt = state.hqPrompt;
        const forceJsonOutput = state.hqForceJsonOutput;
        
        // 重置批次结果
        allBatchResults = [];
        
        // 执行分批翻译
        await processBatchTranslation(
            currentJsonData, 
            allImageBase64, 
            batchSize, 
            sessionResetFrequency,
            provider,
            apiKey,
            modelName,
            customBaseUrl,
            rpmLimit,
            lowReasoning,
            prompt,
            forceJsonOutput
        );
        
        // 5. 解析合并的JSON结果并导入
        ui.showGeneralMessage("翻译完成，正在导入翻译结果...", "info", false);
        ui.updateProgressBar(90, '导入翻译结果...');
        await importTranslationResult(mergeJsonResults(allBatchResults));
        
        // 完成
        ui.updateProgressBar(100, '翻译完成！');
        ui.showGeneralMessage("高质量翻译完成！", "success");
    } catch (error) {
        console.error("高质量翻译过程出错:", error);
        ui.showGeneralMessage(`翻译失败: ${error.message}`, "error");
    } finally {
        // 恢复状态
        isHqTranslationInProgress = false;
        state.setBatchTranslationInProgress(false);
        ui.updateButtonStates();
    }
}

/**
 * 消除所有图片文字并获取原文
 */
async function removeAllImagesText() {
    return new Promise((resolve, reject) => {
        // 这里可以直接调用main.js中已有的removeAllBubblesText函数
        // 但需要修改为返回Promise，所以这里重新实现简化版
        
        const totalImages = state.images.length;
        let currentIndex = 0;
        let failCount = 0;
        
        // 更新进度条（不需要显示进度条，因为在startHqTranslation中已经显示了）
        ui.updateProgressBar(0, `消除文字: 0/${totalImages}`);
        
        // 从DOM获取全局设置（与普通翻译保持一致）
        const sourceLanguage = $('#sourceLanguage').val();
        const ocr_engine = $('#ocrEngine').val();
        // 使用保存的文本样式设置，保持与普通翻译一致的逻辑
        const fontSize = savedFontSize || parseInt($('#fontSize').val()) || state.defaultFontSize;
        const isAutoFontSize = savedAutoFontSize !== null ? savedAutoFontSize : $('#autoFontSize').prop('checked');
        const fontFamily = savedFontFamily || $('#fontFamily').val();
        const textDirection = savedTextDirection || $('#layoutDirection').val();
        const repairSettings = ui.getRepairSettings();
        const useInpainting = repairSettings.useInpainting;
        const useLama = repairSettings.useLama;
        const lamaModel = repairSettings.lamaModel;
        const fillColor = savedFillColor || $('#fillColor').val();
        const textColor = savedTextColor || $('#textColor').val();
        const rotationAngle = savedRotationAngle || 0;  // 全局角度已移除
        
        // 从DOM获取OCR相关参数
        let baiduApiKey = null;
        let baiduSecretKey = null;
        let baiduVersion = 'standard';
        let aiVisionProvider = null;
        let aiVisionApiKey = null;
        let aiVisionModelName = null;
        let aiVisionOcrPrompt = null;
        
        if (ocr_engine === 'baidu_ocr') {
            baiduApiKey = $('#baiduApiKey').val();
            baiduSecretKey = $('#baiduSecretKey').val();
            baiduVersion = $('#baiduVersion').val() || 'standard';
            // 检查百度OCR源语言设置，如果为"无"或空值，则使用所选的源语言
            const baiduOcrLang = state.getBaiduOcrSourceLanguage ? state.getBaiduOcrSourceLanguage() : $('#baiduOcrSourceLanguage').val();
            if (baiduOcrLang === '无' || !baiduOcrLang || baiduOcrLang === '') {
                console.log("高质量翻译: 百度OCR语言设置为空或'无'，将使用选择的源语言:", sourceLanguage);
                state.setBaiduOcrSourceLanguage(sourceLanguage);
            }
        } else if (ocr_engine === 'ai_vision') {
            aiVisionProvider = $('#aiVisionProvider').val();
            aiVisionApiKey = $('#aiVisionApiKey').val();
            aiVisionModelName = $('#aiVisionModelName').val();
            aiVisionOcrPrompt = state.aiVisionOcrPrompt;
        }
        
        const aiVisionOcrJsonMode = state.isAiVisionOcrJsonMode;
        
        function processNextImage() {
            if (currentIndex >= totalImages) {
                ui.updateProgressBar(25, `消除文字完成`); // 表示这一阶段已完成，进入下一阶段
                if (failCount > 0) {
                    reject(new Error(`消除文字完成，但有 ${failCount} 张图片失败`));
                } else {
                    resolve();
                }
                return;
            }
            
            const progressPercent = Math.floor((currentIndex / totalImages) * 25); // 最多占总进度的25%
            ui.updateProgressBar(progressPercent, `消除文字: ${currentIndex + 1}/${totalImages}`);
            ui.showTranslatingIndicator(currentIndex);
            
            const imageData = state.images[currentIndex];
            
            // 添加日志，显示当前使用的修复方式
            console.log(`高质量翻译-消除文字[${currentIndex + 1}/${totalImages}]: 使用修复方式: ${useLama ? 'LAMA' : (useInpainting ? 'MI-GAN' : '纯色填充')}`);
            
            // --- 检查并使用已有的坐标和角度（优先级：savedManualCoords > bubbleCoords > 自动检测）---
            // 角度优先从 bubbleStates 提取（唯一状态来源），回退到 bubbleAngles（检测结果）
            let coordsToUse = null;
            let anglesToUse = null;
            const extractAngles = (states) => states && states.length > 0 ? states.map(s => s.rotationAngle || 0) : null;
            
            if (imageData.savedManualCoords && imageData.savedManualCoords.length > 0) {
                coordsToUse = imageData.savedManualCoords;
                anglesToUse = imageData.savedManualAngles || null;
                console.log(`高质量翻译[${currentIndex}]: 使用手动标注框 ${coordsToUse.length} 个`);
            } else if (imageData.bubbleCoords && imageData.bubbleCoords.length > 0) {
                coordsToUse = imageData.bubbleCoords;
                anglesToUse = extractAngles(imageData.bubbleStates) || imageData.bubbleAngles || null;
                console.log(`高质量翻译[${currentIndex}]: 使用已有文本框 ${coordsToUse.length} 个`);
            }
            
            // 准备API请求参数
            const params = {
                image: imageData.originalDataURL.split(',')[1],
                source_language: sourceLanguage,
                target_language: $('#targetLanguage').val(),
                fontSize: fontSize, 
                autoFontSize: isAutoFontSize,
                fontFamily: fontFamily, 
                textDirection: textDirection,
                autoTextDirection: savedAutoTextDirection,  // 自动排版开关
                use_inpainting: useInpainting,
                use_lama: useLama,
                lamaModel: lamaModel,
                fillColor: fillColor,
                textColor: textColor,
                rotationAngle: rotationAngle,
                skip_translation: true,
                remove_only: true,
                skip_ocr: false,
                use_json_format_translation: false,
                use_json_format_ai_vision_ocr: aiVisionOcrJsonMode,
                bubble_coords: coordsToUse,
                bubble_angles: anglesToUse,
                ocr_engine: ocr_engine,
                baidu_api_key: baiduApiKey,
                baidu_secret_key: baiduSecretKey,
                baidu_version: baiduVersion,
                baidu_ocr_language: ocr_engine === 'baidu_ocr' ? state.getBaiduOcrSourceLanguage() : null,
                ai_vision_provider: aiVisionProvider,
                ai_vision_api_key: aiVisionApiKey,
                ai_vision_model_name: aiVisionModelName,
                ai_vision_ocr_prompt: aiVisionOcrPrompt,
                // 添加描边参数，使用保存的值以保持一致性
                strokeEnabled: savedStrokeEnabled,
                strokeColor: savedStrokeColor,
                strokeWidth: savedStrokeWidth,
                // 调试选项
                showDetectionDebug: state.showDetectionDebug
            };
            
            // 调用API
            api.translateImageApi(params)
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
                    
                    // 使用统一的 bubbleStates 保存所有设置
                    const hqBubbleStates = bubbleStateModule.createBubbleStatesFromResponse(response, {
                        fontSize: fontSize,
                        fontFamily: fontFamily,
                        textDirection: textDirection,
                        textColor: textColor,
                        fillColor: fillColor,
                        inpaintMethod: useLama ? lamaModel : 'solid',
                        strokeEnabled: savedStrokeEnabled,
                        strokeColor: savedStrokeColor,
                        strokeWidth: savedStrokeWidth
                    });
                    state.updateImagePropertyByIndex(currentIndex, 'bubbleStates', hqBubbleStates);
                    state.updateImagePropertyByIndex(currentIndex, 'bubbleTexts', hqBubbleStates.map(s => s.translatedText || ""));
                    
                    // 更新缩略图
                    ui.renderThumbnails();
                    console.log(`高质量翻译-消除文字[${currentIndex + 1}/${totalImages}]: 处理完成`);
                })
                .catch(error => {
                    ui.hideTranslatingIndicator(currentIndex);
                    console.error(`图片 ${currentIndex} 消除文字失败:`, error);
                    failCount++;
                    state.updateImagePropertyByIndex(currentIndex, 'translationFailed', true);
                    ui.renderThumbnails();
                })
                .finally(() => {
                    currentIndex++;
                    processNextImage();
                });
        }
        
        // 开始处理
        processNextImage();
    });
}

/**
 * 导出文本为JSON
 */
function exportTextToJson() {
    const allImages = state.images;
    if (allImages.length === 0) return null;
    
    // 准备导出数据
    const exportData = [];
    
    // 遍历所有图片
    for (let imageIndex = 0; imageIndex < allImages.length; imageIndex++) {
        const image = allImages[imageIndex];
        const originalTexts = image.originalTexts || [];
        
        // 构建该图片的文本数据
        const imageTextData = {
            imageIndex: imageIndex,
            bubbles: []
        };
        
        // 构建每个气泡的文本数据
        for (let bubbleIndex = 0; bubbleIndex < originalTexts.length; bubbleIndex++) {
            const original = originalTexts[bubbleIndex] || '';
            
            // 获取气泡的排版方向
            let textDirection = 'vertical'; // 默认为竖排
            
            // 优先使用每个气泡独立的排版方向（自动检测结果）
            if (image.bubbleStates && image.bubbleStates[bubbleIndex] && image.bubbleStates[bubbleIndex].textDirection) {
                const bubbleDir = image.bubbleStates[bubbleIndex].textDirection;
                // 确保不传递 'auto'，如果是 'auto' 则使用默认的 'vertical'
                textDirection = (bubbleDir === 'auto') ? 'vertical' : bubbleDir;
            } else if (image.layoutDirection && image.layoutDirection !== 'auto') {
                // 如果没有独立设置，使用全局设置（但不使用 'auto'）
                textDirection = image.layoutDirection;
            }
            
            imageTextData.bubbles.push({
                bubbleIndex: bubbleIndex,
                original: original,
                translated: "", // 初始译文为空
                textDirection: textDirection
            });
        }
        
        exportData.push(imageTextData);
    }
    
    return exportData;
}

/**
 * 收集所有图片的Base64数据
 */
function collectAllImageBase64() {
    return state.images.map(image => image.originalDataURL.split(',')[1]);
}

/**
 * 分批处理翻译
 */
async function processBatchTranslation(jsonData, imageBase64Array, batchSize, sessionResetFrequency, provider, apiKey, modelName, customBaseUrl, rpmLimit, lowReasoning, prompt, forceJsonOutput) {
    const totalImages = imageBase64Array.length;
    const totalBatches = Math.ceil(totalImages / batchSize);
    
    // 显示批次进度
    ui.updateProgressBar(0, '0/' + totalBatches);
    
    // 创建限流器
    const rateLimiter = createRateLimiter(rpmLimit);
    
    // 跟踪批次计数，用于决定何时重置会话
    let batchCount = 0;
    let sessionId = generateSessionId();
    
    for (let batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
        // 更新进度
        ui.updateProgressBar((batchIndex / totalBatches) * 100, `${batchIndex + 1}/${totalBatches}`);
        
        // 检查是否需要重置会话
        if (batchCount >= sessionResetFrequency) {
            console.log("重置会话上下文");
            sessionId = generateSessionId();
            batchCount = 0;
        }
        
        // 准备这一批次的图片和JSON数据
        const startIdx = batchIndex * batchSize;
        const endIdx = Math.min(startIdx + batchSize, totalImages);
        const batchImages = imageBase64Array.slice(startIdx, endIdx);
        const batchJsonData = filterJsonForBatch(jsonData, startIdx, endIdx);
        
        // 重试逻辑
        const maxRetries = state.hqTranslationMaxRetries || 2;
        let retryCount = 0;
        let success = false;
        
        while (retryCount <= maxRetries && !success) {
            try {
                // 等待速率限制
                await rateLimiter.waitForTurn();
                
                // 发送批次到AI
                const result = await callAiForTranslation(
                    batchImages,
                    batchJsonData,
                    provider,
                    apiKey,
                    modelName,
                    customBaseUrl,
                    lowReasoning,
                    prompt,
                    sessionId,
                    forceJsonOutput
                );
                
                // 解析并保存结果
                if (result) {
                    allBatchResults.push(result);
                    success = true;
                }
                
                // 增加批次计数
                batchCount++;
                
            } catch (error) {
                retryCount++;
                if (retryCount <= maxRetries) {
                    console.log(`批次 ${batchIndex + 1} 翻译失败，第 ${retryCount}/${maxRetries} 次重试...`);
                    ui.showGeneralMessage(`批次 ${batchIndex + 1} 失败，正在重试 (${retryCount}/${maxRetries})...`, "warning", true);
                    await new Promise(r => setTimeout(r, 1000)); // 等待1秒后重试
                } else {
                    console.error(`批次 ${batchIndex + 1} 翻译最终失败:`, error);
                    ui.showGeneralMessage(`批次 ${batchIndex + 1} 翻译失败: ${error.message}`, "error", true);
                    // 继续处理下一批次
                }
            }
        }
    }
    
    // 完成所有批次
    ui.updateProgressBar(100, `${totalBatches}/${totalBatches}`);
}

/**
 * 为特定批次过滤JSON数据
 */
function filterJsonForBatch(jsonData, startIdx, endIdx) {
    return jsonData.filter(item => item.imageIndex >= startIdx && item.imageIndex < endIdx);
}

/**
 * 创建简单的速率限制器
 */
function createRateLimiter(rpm) {
    // 修复 Bug #2: 处理 rpm 为 0 或无效值的情况，避免除零错误
    if (!rpm || rpm <= 0) {
        return {
            waitForTurn: async function() {
                // 无限制时不等待
            }
        };
    }
    
    const intervalMs = 60000 / rpm; // 计算请求间隔
    let lastRequestTime = 0;
    
    return {
        waitForTurn: async function() {
            const now = Date.now();
            const timeToWait = Math.max(0, intervalMs - (now - lastRequestTime));
            
            if (timeToWait > 0) {
                await new Promise(resolve => setTimeout(resolve, timeToWait));
            }
            
            lastRequestTime = Date.now();
        }
    };
}

/**
 * 生成会话ID
 */
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 9);
}

/**
 * 调用AI进行翻译 - 通过后端代理，避免 CORS 问题
 */
async function callAiForTranslation(imageBase64Array, jsonData, provider, apiKey, modelName, customBaseUrl, lowReasoning, prompt, sessionId, forceJsonOutput) {
    // 构建提示词和图片
    const jsonString = JSON.stringify(jsonData, null, 2);
    const messages = [
        {
            role: "system",
            content: "你是一个专业的漫画翻译助手，能够根据漫画图像内容和上下文提供高质量的翻译。"
        },
        {
            role: "user",
            content: [
                {
                    type: "text",
                    text: prompt + "\n\n以下是JSON数据:\n```json\n" + jsonString + "\n```"
                }
            ]
        }
    ];
    
    // 添加图片到消息中
    for (const imgBase64 of imageBase64Array) {
        messages[1].content.push({
            type: "image_url",
            image_url: {
                url: `data:image/png;base64,${imgBase64}`
            }
        });
    }
    
    // 获取当前取消思考方法设置
    const noThinkingMethod = state.hqNoThinkingMethod || 'gemini';
    
    // 通过后端 API 代理调用，避免 CORS 问题
    try {
        console.log(`高质量翻译: 通过后端代理调用 ${provider} API...`);
        
        const response = await api.hqTranslateBatchApi({
            provider: provider,
            api_key: apiKey,
            model_name: modelName,
            custom_base_url: customBaseUrl,
            messages: messages,
            low_reasoning: lowReasoning,
            force_json_output: forceJsonOutput,
            no_thinking_method: noThinkingMethod
        });
        
        if (!response.success) {
            throw new Error(response.error || 'API 调用失败');
        }
        
        let content = response.content;
        
        // 如果是强制JSON输出，则内容应该已经是JSON了
        if (forceJsonOutput) {
            try {
                // 直接解析AI返回的JSON
                return JSON.parse(content);
            } catch (e) {
                console.error("解析AI强制JSON返回的内容失败:", e);
                console.log("原始内容:", content);
                throw new Error("解析AI返回的JSON结果失败，请检查服务商是否支持response_format参数");
            }
        } else {
            // 使用原来的代码处理非强制JSON输出的情况
            // 尝试从内容中提取JSON
            const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
            if (jsonMatch && jsonMatch[1]) {
                content = jsonMatch[1];
            }
            
            try {
                // 尝试解析JSON
                return JSON.parse(content);
            } catch (e) {
                console.error("解析AI返回的JSON失败:", e);
                console.log("原始内容:", content);
                throw new Error("解析AI返回的翻译结果失败");
            }
        }
    } catch (error) {
        console.error("调用AI翻译API失败:", error);
        throw error;
    }
}

/**
 * 合并所有批次的JSON结果
 */
function mergeJsonResults(batchResults) {
    if (!batchResults || batchResults.length === 0) {
        return [];
    }
    
    // 合并所有批次结果
    const mergedResult = [];
    
    // 遍历每个批次结果
    for (const batchResult of batchResults) {
        // 修复 Bug #5: 添加类型检查，确保 batchResult 是数组
        if (!batchResult) {
            console.warn('mergeJsonResults: 跳过空的批次结果');
            continue;
        }
        
        // 如果 batchResult 不是数组，尝试将其包装成数组
        const batchArray = Array.isArray(batchResult) ? batchResult : [batchResult];
        
        // 遍历批次中的每个图片数据
        for (const imageData of batchArray) {
            // 检查 imageData 是否有效
            if (imageData && typeof imageData === 'object' && 'imageIndex' in imageData) {
                mergedResult.push(imageData);
            } else {
                console.warn('mergeJsonResults: 跳过无效的图片数据', imageData);
            }
        }
    }
    
    // 按imageIndex排序
    mergedResult.sort((a, b) => a.imageIndex - b.imageIndex);
    
    return mergedResult;
}

/**
 * 导入翻译结果
 */
async function importTranslationResult(importedData) {
    if (!importedData || importedData.length === 0) {
        throw new Error("没有有效的翻译数据可导入");
    }
    
    // 保存当前图片索引，以便导入完成后返回
    const originalImageIndex = state.currentImageIndex;
    
    // 获取当前的全局设置作为默认值，使用保存的文本样式设置
    const currentFontSize = savedFontSize || parseInt($('#fontSize').val());
    const currentAutoFontSize = savedAutoFontSize !== null ? savedAutoFontSize : $('#autoFontSize').prop('checked');
    const currentFontFamily = savedFontFamily || $('#fontFamily').val();
    // 确保 currentTextDirection 不是 'auto'
    const rawTextDirection = savedTextDirection || $('#layoutDirection').val();
    const currentTextDirection = (rawTextDirection === 'auto') ? 'vertical' : rawTextDirection;
    const currentTextColor = savedTextColor || $('#textColor').val();
    const currentFillColor = savedFillColor || $('#fillColor').val();
    const currentRotationAngle = savedRotationAngle || 0;  // 全局角度已移除
    // 描边设置 - 使用保存的值
    const currentStrokeEnabled = savedStrokeEnabled !== null ? savedStrokeEnabled : state.strokeEnabled;
    const currentStrokeColor = savedStrokeColor || state.strokeColor;
    const currentStrokeWidth = savedStrokeWidth !== null ? savedStrokeWidth : state.strokeWidth;
    
    console.log("高质量翻译导入结果使用的文本样式:", {
        fontFamily: currentFontFamily,
        fontSize: currentFontSize,
        textDirection: currentTextDirection,
        strokeEnabled: currentStrokeEnabled
    });
    
    ui.updateProgressBar(90, "更新图片数据...");
    
    // 创建一个队列，用于存储所有渲染任务
    const renderTasks = [];
    
    // 遍历导入的数据
    const totalImages = importedData.length;
    let processedImages = 0;
    
    for (const imageData of importedData) {
        processedImages++;
        ui.updateProgressBar(90 + (processedImages / totalImages * 5), `处理图片 ${processedImages}/${totalImages}`);
        
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
        
        // 遍历该图片的所有气泡数据
        for (const bubbleData of imageData.bubbles || []) {
            const bubbleIndex = bubbleData.bubbleIndex;
            
            // 检查气泡索引是否有效
            if (bubbleIndex < 0 || bubbleIndex >= image.bubbleCoords.length) {
                console.warn(`图片 ${imageIndex}: 跳过无效的气泡索引 ${bubbleIndex}`);
                continue;
            }
            
            // 获取翻译文本和排版方向
            const translatedText = bubbleData.translated;
            // 确保 textDirection 不是 'auto'，如果是则使用默认的 currentTextDirection
            let textDirection = bubbleData.textDirection;
            if (textDirection === 'auto') {
                textDirection = currentTextDirection;
            }
            
            // 更新翻译文本
            image.bubbleTexts[bubbleIndex] = translatedText;
            
            // 确定要使用的排版方向
            const effectiveTextDirection = (textDirection && textDirection !== 'auto') ? textDirection : currentTextDirection;
            
            // 始终更新 bubbleStates（确保 translatedText 被更新）
            // 如果图片没有bubbleStates或长度不匹配，则初始化它
            if (!image.bubbleStates || 
                !Array.isArray(image.bubbleStates) || 
                image.bubbleStates.length !== image.bubbleCoords.length) {
                // 创建新的气泡设置
                const detectedAngles = image.bubbleAngles || [];
                const newSettings = [];
                for (let i = 0; i < image.bubbleCoords.length; i++) {
                    const bubbleTextDirection = (i === bubbleIndex) ? effectiveTextDirection : currentTextDirection;
                    // 计算自动排版方向（根据宽高比）
                    let autoDir = bubbleTextDirection;
                    if (image.bubbleCoords[i] && image.bubbleCoords[i].length >= 4) {
                        const [x1, y1, x2, y2] = image.bubbleCoords[i];
                        autoDir = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal';
                    }
                    newSettings.push({
                        translatedText: image.bubbleTexts[i] || "",
                        fontSize: currentFontSize,
                        autoFontSize: currentAutoFontSize,
                        fontFamily: currentFontFamily,
                        textDirection: bubbleTextDirection,
                        autoTextDirection: autoDir,  // 自动检测的排版方向
                        position: { x: 0, y: 0 },
                        textColor: currentTextColor,
                        rotationAngle: detectedAngles[i] || currentRotationAngle,
                        fillColor: currentFillColor,
                        strokeEnabled: currentStrokeEnabled,
                        strokeColor: currentStrokeColor,
                        strokeWidth: currentStrokeWidth
                    });
                }
                image.bubbleStates = newSettings;
            } else if (image.bubbleStates[bubbleIndex]) {
                // 更新现有的 bubbleState
                image.bubbleStates[bubbleIndex].translatedText = translatedText;
                if (textDirection && textDirection !== 'auto') {
                    image.bubbleStates[bubbleIndex].textDirection = effectiveTextDirection;
                }
            } else {
                // 创建新的 bubbleState
                const bubbleDetectedAngle = (image.bubbleAngles && image.bubbleAngles[bubbleIndex]) || currentRotationAngle;
                // 计算自动排版方向
                let autoDir = effectiveTextDirection;
                if (image.bubbleCoords[bubbleIndex] && image.bubbleCoords[bubbleIndex].length >= 4) {
                    const [x1, y1, x2, y2] = image.bubbleCoords[bubbleIndex];
                    autoDir = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal';
                }
                image.bubbleStates[bubbleIndex] = {
                    translatedText: translatedText,
                    fontSize: currentFontSize,
                    autoFontSize: currentAutoFontSize,
                    fontFamily: currentFontFamily,
                    textDirection: effectiveTextDirection,
                    autoTextDirection: autoDir,  // 自动检测的排版方向
                    position: { x: 0, y: 0 },
                    textColor: currentTextColor,
                    rotationAngle: bubbleDetectedAngle,
                    fillColor: currentFillColor,
                    strokeEnabled: currentStrokeEnabled,
                    strokeColor: currentStrokeColor,
                    strokeWidth: currentStrokeWidth
                };
            }
            
            imageUpdated = true;
        }
        
        // 如果图片有更新，添加到渲染队列
        if (imageUpdated) {
            // bubbleStates 已更新，switchImage 会从 bubbleStates[0] 读取设置
            // 同步 bubbleTexts
            image.bubbleTexts = image.bubbleStates.map(bs => bs.translatedText || bs.text || "");
            
            // 添加到渲染队列
            if (image.translatedDataURL) {
                renderTasks.push(async () => {
                    const editMode = await import('./edit_mode.js');
                    
                    // 保存当前索引
                    const currentIndex = state.currentImageIndex;
                    
                    // 临时切换到目标图片（但不更新UI）
                    state.setCurrentImageIndex(imageIndex);
                    
                    try {
                        // 重新渲染图片，传递 savedAutoFontSize 以启用自动字号计算
                        await editMode.reRenderFullImage(false, true, savedAutoFontSize);
                        
                        // 图片已在reRenderFullImage中更新到state中
                        console.log(`已完成图片 ${imageIndex} 的渲染 (autoFontSize=${savedAutoFontSize})`);
                    } finally {
                        // 恢复原始索引（但不更新UI）
                        state.setCurrentImageIndex(currentIndex);
                    }
                });
            }
        }
    }
    
    // 开始执行渲染队列
    ui.updateProgressBar(95, "开始渲染图片...");
    ui.showGeneralMessage("正在渲染图片，请稍候...", "info", false);
    
    // 执行所有渲染任务
    for (let i = 0; i < renderTasks.length; i++) {
        ui.updateProgressBar(95 + (i / renderTasks.length * 5), `渲染图片 ${i+1}/${renderTasks.length}`);
        await renderTasks[i]();
    }
    
    ui.updateProgressBar(100, "完成图片更新");
    
    // 全部导入完成后，回到最初的图片并刷新UI
    // switchImage 会从 bubbleStates[0] 读取设置并显示到 UI
    const main = await import('./main.js');
    main.switchImage(originalImageIndex);
}

/**
 * 初始化高质量翻译设置UI
 * 高质量翻译设置已移至顶部设置模态框(settings_modal.js)，此函数保留为空
 */
export function initHqTranslationUI() {
    // 高质量翻译设置UI已移至设置模态框，此处无需初始化
} 