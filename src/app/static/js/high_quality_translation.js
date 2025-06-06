/**
 * 高质量翻译模式 - 利用AI上下文增强翻译质量
 */
import * as api from './api.js';
import * as state from './state.js';
import * as ui from './ui.js';
import * as session from './session.js';

// 保存翻译状态
let isHqTranslationInProgress = false;
let currentJsonData = null;
let allImageBase64 = [];
let allBatchResults = [];
// 保存开始翻译前的字体设置
let savedFontFamily = null;

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
    
    // 从state获取配置参数，而不是直接从DOM元素获取
    const provider = state.hqTranslateProvider;
    const apiKey = state.hqApiKey;
    const modelName = state.hqModelName;
    const customBaseUrl = state.hqCustomBaseUrl;
    
    if (!apiKey || !modelName) {
        ui.showGeneralMessage("请填写API Key和模型名称", "error");
        return;
    }
    
    if (provider === 'custom_openai' && !customBaseUrl) {
        ui.showGeneralMessage("使用自定义服务时必须填写Base URL", "error");
        return;
    }
    
    // 保存当前选择的字体，用于翻译后恢复
    savedFontFamily = $('#fontFamily').val();
    console.log("高质量翻译前保存的字体设置:", savedFontFamily);
    
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
        
        // 触发自动保存
        session.triggerAutoSave();
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
        
        // 从state获取全局设置
        const sourceLanguage = state.sourceLanguage || $('#sourceLanguage').val();
        const ocr_engine = state.ocrEngine || $('#ocrEngine').val();
        const fontSize = state.defaultFontSize;
        const isAutoFontSize = state.autoFontSize !== undefined ? state.autoFontSize : $('#autoFontSize').prop('checked');
        const fontFamily = state.defaultFontFamily;
        const textDirection = state.defaultLayoutDirection;
        const repairSettings = ui.getRepairSettings();
        const useInpainting = repairSettings.useInpainting;
        const useLama = repairSettings.useLama;
        const blendEdges = state.blendEdges !== undefined ? state.blendEdges : $('#blendEdges').prop('checked');
        const inpaintingStrength = state.inpaintingStrength !== undefined ? state.inpaintingStrength : parseFloat($('#inpaintingStrength').val());
        const fillColor = state.defaultFillColor;
        const textColor = state.defaultTextColor;
        const rotationAngle = state.rotationAngle !== undefined ? state.rotationAngle : parseFloat($('#rotationAngle').val() || '0');
        
        // 从state获取OCR相关参数
        let baiduApiKey = null;
        let baiduSecretKey = null;
        let baiduVersion = 'standard';
        let aiVisionProvider = null;
        let aiVisionApiKey = null;
        let aiVisionModelName = null;
        let aiVisionOcrPrompt = null;
        
        if (ocr_engine === 'baidu_ocr') {
            baiduApiKey = state.baiduApiKey || $('#baiduApiKey').val();
            baiduSecretKey = state.baiduSecretKey || $('#baiduSecretKey').val();
            baiduVersion = state.baiduVersion || $('#baiduVersion').val() || 'standard';
        } else if (ocr_engine === 'ai_vision') {
            aiVisionProvider = state.aiVisionProvider || $('#aiVisionProvider').val();
            aiVisionApiKey = state.aiVisionApiKey || $('#aiVisionApiKey').val();
            aiVisionModelName = state.aiVisionModelName || $('#aiVisionModelName').val();
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
            
            // 准备API请求参数
            const params = {
                image: imageData.originalDataURL.split(',')[1],
                source_language: sourceLanguage,
                target_language: state.targetLanguage || $('#targetLanguage').val(),
                fontSize: fontSize, 
                autoFontSize: isAutoFontSize,
                fontFamily: fontFamily, 
                textDirection: textDirection,
                use_inpainting: useInpainting,
                use_lama: useLama,
                blend_edges: blendEdges,
                inpainting_strength: inpaintingStrength,
                fill_color: fillColor,
                text_color: textColor,
                rotation_angle: rotationAngle,
                skip_translation: true,
                remove_only: true,
                skip_ocr: false,
                use_json_format_translation: false,
                use_json_format_ai_vision_ocr: aiVisionOcrJsonMode,
                bubble_coords: imageData.savedManualCoords && imageData.savedManualCoords.length > 0 
                               ? imageData.savedManualCoords : null,
                ocr_engine: ocr_engine,
                baidu_api_key: baiduApiKey,
                baidu_secret_key: baiduSecretKey,
                baidu_version: baiduVersion,
                ai_vision_provider: aiVisionProvider,
                ai_vision_api_key: aiVisionApiKey,
                ai_vision_model_name: aiVisionModelName,
                ai_vision_ocr_prompt: aiVisionOcrPrompt
            };
            
            // 调用API
            api.translateImageApi(params)
                .then(response => {
                    ui.hideTranslatingIndicator(currentIndex);
                    
                    // 更新图片状态
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
                            } else {
                                // 文本过多，截断
                                bubbleTexts = bubbleTexts.slice(0, bubbleCoords.length);
                            }
                        }
                    }
                    
                    state.updateImagePropertyByIndex(currentIndex, 'bubbleTexts', bubbleTexts);
                    state.updateImagePropertyByIndex(currentIndex, 'bubbleCoords', bubbleCoords);
                    state.updateImagePropertyByIndex(currentIndex, 'originalTexts', response.original_texts || []);
                    state.updateImagePropertyByIndex(currentIndex, 'textboxTexts', response.textbox_texts || []);
                    state.updateImagePropertyByIndex(currentIndex, 'translationFailed', false);
                    state.updateImagePropertyByIndex(currentIndex, 'showOriginal', false);
                    
                    // 保存处理使用的设置
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
                    
                    // 添加标记信息，标记使用了哪种修复方式
                    if (useLama) {
                        state.updateImagePropertyByIndex(currentIndex, '_lama_inpainted', true);
                        console.log(`高质量翻译-消除文字[${currentIndex + 1}/${totalImages}]: 已标记使用LAMA修复`);
                    } else {
                        state.updateImagePropertyByIndex(currentIndex, '_lama_inpainted', false);
                    }
                    
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
            
            if (image.layoutDirection) {
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
            }
            
            // 增加批次计数
            batchCount++;
            
        } catch (error) {
            console.error(`批次 ${batchIndex + 1} 翻译失败:`, error);
            ui.showGeneralMessage(`批次 ${batchIndex + 1} 翻译失败: ${error.message}`, "error", true);
            // 继续处理下一批次
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
 * 调用AI进行翻译
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
    
    // 构建API请求参数
    const apiParams = {
        model: modelName,
        messages: messages
    };
    
    // 如果强制JSON输出，添加response_format参数
    if (forceJsonOutput) {
        apiParams.response_format = { type: "json_object" };
        console.log("已启用强制JSON输出模式");
    }
    
    // 获取当前取消思考方法设置
    const noThinkingMethod = state.hqNoThinkingMethod || 'gemini';
    
    // 根据不同取消思考方法添加参数
    if (lowReasoning) {
        if (noThinkingMethod === 'gemini') {
            // Gemini风格：使用reasoning_effort参数
            apiParams.reasoning_effort = "low";
            console.log("使用Gemini方式取消思考: reasoning_effort=low");
        } else if (noThinkingMethod === 'volcano' && provider === 'volcano') {
            // 火山引擎风格：设置thinking=null
            apiParams.thinking = null;
            console.log("使用火山引擎方式取消思考: thinking=null");
        } else {
            // 默认使用Gemini风格
            apiParams.reasoning_effort = "low";
            console.log("使用默认方式取消思考: reasoning_effort=low");
        }
    }
    
    // 根据不同服务商设置不同的endpoint
    let baseUrl = "";
    switch (provider) {
        case 'siliconflow':
            baseUrl = "https://api.siliconflow.cn/v1/chat/completions";
            break;
        case 'deepseek':
            baseUrl = "https://api.deepseek.com/v1/chat/completions";
            break;
        case 'volcano':
            baseUrl = "https://ark.cn-beijing.volces.com/api/v3/chat/completions";
            break;
        case 'gemini':
            baseUrl = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions";
            break;
        case 'custom_openai':
            baseUrl = customBaseUrl + "/chat/completions";
            break;
        default:
            baseUrl = "https://api.siliconflow.cn/v1/chat/completions";
    }
    
    // 发送请求
    try {
        const response = await fetch(baseUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify(apiParams)
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API请求失败: ${response.status} ${errorText}`);
        }
        
        const result = await response.json();
        
        // 提取AI返回的文本
        if (result.choices && result.choices.length > 0) {
            let content = result.choices[0].message.content;
            
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
        } else {
            throw new Error("AI未返回有效内容");
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
        // 遍历批次中的每个图片数据
        for (const imageData of batchResult) {
            // 将图片数据添加到合并结果中
            mergedResult.push(imageData);
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
    
    // 保存当前图片索引
    const originalImageIndex = state.currentImageIndex;
    
    // 获取当前的全局设置作为默认值，但使用保存的字体设置
    const currentFontSize = parseInt($('#fontSize').val());
    const currentAutoFontSize = $('#autoFontSize').prop('checked');
    // 使用保存的字体设置，如果没有则使用当前UI中的字体
    const currentFontFamily = savedFontFamily || $('#fontFamily').val();
    const currentTextColor = $('#textColor').val();
    const currentFillColor = $('#fillColor').val();
    const currentRotationAngle = parseFloat($('#rotationAngle').val() || '0');
    
    console.log("高质量翻译导入结果使用的字体:", currentFontFamily);
    
    // 遍历导入的数据
    for (const imageData of importedData) {
        const imageIndex = imageData.imageIndex;
        
        // 检查图片索引是否有效
        if (imageIndex < 0 || imageIndex >= state.images.length) {
            console.warn(`跳过无效的图片索引: ${imageIndex}`);
            continue;
        }
        
        // 切换到该图片
        await switchToImage(imageIndex);
        
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
            const textDirection = bubbleData.textDirection;
            
            // 更新翻译文本
            image.bubbleTexts[bubbleIndex] = translatedText;
            
            // 更新排版方向 (需要创建或更新bubbleSettings)
            if (textDirection) {
                // 如果图片没有bubbleSettings或长度不匹配，则初始化它
                if (!image.bubbleSettings || 
                    !Array.isArray(image.bubbleSettings) || 
                    image.bubbleSettings.length !== image.bubbleCoords.length) {
                    // 创建新的气泡设置，使用当前左侧边栏的设置
                    const newSettings = [];
                    for (let i = 0; i < image.bubbleCoords.length; i++) {
                        const bubbleTextDirection = (i === bubbleIndex) ? textDirection : $('#layoutDirection').val();
                        newSettings.push({
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
                    // 更新现有bubbleSettings中的textDirection，保持其他设置不变
                    if (!image.bubbleSettings[bubbleIndex]) {
                        image.bubbleSettings[bubbleIndex] = {
                            text: image.bubbleTexts[bubbleIndex] || "",
                            fontSize: currentFontSize,
                            autoFontSize: currentAutoFontSize,
                            fontFamily: currentFontFamily,
                            textDirection: textDirection,
                            position: { x: 0, y: 0 },
                            textColor: currentTextColor,
                            rotationAngle: currentRotationAngle,
                            fillColor: currentFillColor
                        };
                    } else {
                        // 只更新textDirection，保持其他设置（特别是fontFamily）不变
                        image.bubbleSettings[bubbleIndex].textDirection = textDirection;
                    }
                }
            }
            
            imageUpdated = true;
        }
        
        // 如果图片有更新，重新渲染
        if (imageUpdated) {
            // 更新图片的fontFamily属性，确保切换图片后能保持字体设置
            image.fontFamily = currentFontFamily;
            
            // 使用已有的reRenderFullImage函数重新渲染
            await new Promise(resolve => {
                if (image.translatedDataURL) {
                    // 如果已经有翻译图像，重新渲染
                    import('./edit_mode.js').then(editMode => {
                        editMode.reRenderFullImage().then(resolve);
                    });
                } else {
                    resolve();
                }
            });
            
            // 短暂停顿，以便用户可以看到进度
            await new Promise(resolve => setTimeout(resolve, 300));
        }
    }
    
    // 全部导入完成后，回到最初的图片
    await switchToImage(originalImageIndex);
    
    // 确保在完成导入后，字体选择器还原为保存的字体
    if (savedFontFamily) {
        $('#fontFamily').val(savedFontFamily);
        
        // 同时更新当前图片的fontFamily属性，确保切换图片后仍然保持选择的字体
        const currentImage = state.images[originalImageIndex];
        if (currentImage) {
            currentImage.fontFamily = savedFontFamily;
        }
    }
}

/**
 * 切换到指定索引的图片
 */
async function switchToImage(index) {
    return new Promise(resolve => {
        import('./main.js').then(main => {
            main.switchImage(index);
            resolve();
        });
    });
}

/**
 * 初始化高质量翻译设置UI
 */
export function initHqTranslationUI() {
    // 服务商选择
    $('#hqTranslateProvider').val(state.hqTranslateProvider);
    
    // API Key
    $('#hqApiKey').val(state.hqApiKey);
    
    // 模型名称
    $('#hqModelName').val(state.hqModelName);
    
    // 自定义Base URL(仅当选择自定义OpenAI兼容服务时可见)
    $('#hqCustomBaseUrl').val(state.hqCustomBaseUrl);
    $('.custom-base-url-div').toggle($('#hqTranslateProvider').val() === 'custom_openai');
    
    // 批次大小
    $('#hqBatchSize').val(state.hqBatchSize);
    
    // 会话重置频率
    $('#hqSessionReset').val(state.hqSessionReset);
    
    // RPM限制
    $('#hqRpmLimit').val(state.hqRpmLimit);
    
    // 低推理模式
    $('#hqLowReasoning').prop('checked', state.hqLowReasoning);
    
    // 强制JSON输出
    $('#hqForceJsonOutput').prop('checked', state.hqForceJsonOutput);
    
    // 取消思考方法选择
    let noThinkingMethodSelector = `
    <div>
        <label>取消思考方法:</label>
        <select id="hqNoThinkingMethod">
            <option value="gemini" ${state.hqNoThinkingMethod === 'gemini' ? 'selected' : ''}>Gemini (reasoning_effort)</option>
            <option value="volcano" ${state.hqNoThinkingMethod === 'volcano' ? 'selected' : ''}>火山引擎 (thinking: null)</option>
        </select>
    </div>`;
    
    // 在低推理模式选项后面插入取消思考方法选择器
    $('#hqLowReasoning').closest('div').after(noThinkingMethodSelector);
    
    // 翻译提示词
    $('#hqPrompt').val(state.hqPrompt);
    
    // 绑定事件处理程序
    $('#hqTranslateProvider').on('change', function() {
        const provider = $(this).val();
        state.setHqTranslateProvider(provider);
        $('.custom-base-url-div').toggle(provider === 'custom_openai');
    });
    
    $('#hqApiKey').on('change', function() {
        state.setHqApiKey($(this).val());
    });
    
    $('#hqModelName').on('change', function() {
        state.setHqModelName($(this).val());
    });
    
    $('#hqCustomBaseUrl').on('change', function() {
        state.setHqCustomBaseUrl($(this).val());
    });
    
    $('#hqBatchSize').on('change', function() {
        state.setHqBatchSize($(this).val());
    });
    
    $('#hqSessionReset').on('change', function() {
        state.setHqSessionReset($(this).val());
    });
    
    $('#hqRpmLimit').on('change', function() {
        state.setHqRpmLimit($(this).val());
    });
    
    $('#hqLowReasoning').on('change', function() {
        state.setHqLowReasoning($(this).is(':checked'));
    });
    
    // 添加对取消思考方法选择器的事件处理
    $('#hqNoThinkingMethod').on('change', function() {
        state.setHqNoThinkingMethod($(this).val());
    });
    
    $('#hqForceJsonOutput').on('change', function() {
        state.setHqForceJsonOutput($(this).is(':checked'));
    });
    
    $('#hqPrompt').on('change', function() {
        state.setHqPrompt($(this).val());
    });
} 