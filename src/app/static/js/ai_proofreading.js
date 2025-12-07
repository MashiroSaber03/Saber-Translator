/**
 * AI校对模式 - 多轮AI校对提升翻译质量
 */
import * as api from './api.js';
import * as state from './state.js';
import * as ui from './ui.js';
import * as constants from './constants.js';
import * as translationValidator from './translation_validator.js';
import * as bubbleStateModule from './bubble_state.js';

// 保存校对状态
let isProofreadingInProgress = false;
let currentRound = 0;
let totalRounds = 0;
let proofreadingConfig = [];
let currentJsonData = null;
let allImageBase64 = [];
let allBatchResults = [];
// 保存开始校对前的文本样式设置
let savedFontFamily = null;
let savedFontSize = null;
let savedAutoFontSize = false;
let savedTextDirection = null;
let savedAutoTextDirection = false;  // 自动排版开关
let savedFillColor = null;
let savedTextColor = null;
let savedRotationAngle = 0;

/**
 * 开始AI校对流程
 */
export async function startProofreading() {
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
    
    // 检查是否启用了校对功能
    if (!state.isProofreadingEnabled) {
        ui.showGeneralMessage("请先在顶部 ⚙️ 设置菜单中启用校对功能", "warning");
        return;
    }
    
    // 从state获取校对配置
    proofreadingConfig = [...state.proofreadingRounds];
    
    // 验证校对配置是否完整
    if (!translationValidator.validateBeforeTranslation('proofread', { proofreadingRounds: proofreadingConfig })) {
        return;
    }
    
    // 设置轮次计数
    currentRound = 0;
    totalRounds = proofreadingConfig.length;
    
    // 保存当前选择的所有文本样式设置，用于校对过程中保持一致（与高质量翻译保持一致）
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
    console.log("AI校对前保存的文本样式设置:", {
        fontFamily: savedFontFamily,
        fontSize: savedFontSize,
        autoFontSize: savedAutoFontSize,
        textDirection: savedTextDirection,
        autoTextDirection: savedAutoTextDirection,
        fillColor: savedFillColor,
        textColor: savedTextColor,
        rotationAngle: savedRotationAngle
    });
    
    // 立即显示进度条
    $("#translationProgressBar").show();
    ui.updateProgressBar(0, '准备校对...');
    ui.showGeneralMessage(`开始校对，共 ${totalRounds} 轮`, "info", false);
    
    // 设置校对状态
    isProofreadingInProgress = true;
    state.setBatchTranslationInProgress(true);
    ui.updateButtonStates();
    
    try {
        // 主校对循环
        for (currentRound = 0; currentRound < totalRounds; currentRound++) {
            // 获取当前轮次配置
            const roundConfig = proofreadingConfig[currentRound];
            
            ui.showGeneralMessage(`校对第 ${currentRound + 1}/${totalRounds} 轮: ${roundConfig.name || '未命名轮次'}`, "info", false);
            ui.updateProgressBar((currentRound / totalRounds) * 100, `轮次 ${currentRound + 1}/${totalRounds}`);
            
            // 1. 导出文本为JSON
            ui.showGeneralMessage(`轮次 ${currentRound + 1}/${totalRounds}: 导出文本数据...`, "info", false);
            ui.updateProgressBar((currentRound / totalRounds) * 100 + (1 / totalRounds) * 20, '导出文本...');
            currentJsonData = exportTextToJson();
            if (!currentJsonData) {
                throw new Error("导出文本失败");
            }
            
            // 2. 收集所有图片的Base64数据
            ui.showGeneralMessage(`轮次 ${currentRound + 1}/${totalRounds}: 准备图片数据...`, "info", false);
            ui.updateProgressBar((currentRound / totalRounds) * 100 + (1 / totalRounds) * 40, '准备图片数据...');
            allImageBase64 = collectAllImageBase64();
            
            // 3. 分批发送给AI校对
            ui.showGeneralMessage(`轮次 ${currentRound + 1}/${totalRounds}: 发送到AI进行校对...`, "info", false);
            ui.updateProgressBar((currentRound / totalRounds) * 100 + (1 / totalRounds) * 50, '开始发送到AI...');
            
            // 从当前轮次配置获取参数
            const provider = roundConfig.provider;
            const apiKey = roundConfig.apiKey;
            const modelName = roundConfig.modelName;
            const customBaseUrl = roundConfig.customBaseUrl;
            const batchSize = roundConfig.batchSize;
            const sessionResetFrequency = roundConfig.sessionReset;
            const rpmLimit = roundConfig.rpmLimit;
            const lowReasoning = roundConfig.lowReasoning;
            const prompt = roundConfig.prompt;
            const forceJsonOutput = roundConfig.forceJsonOutput;
            
            // 重置批次结果
            allBatchResults = [];
            
            // 执行分批校对
            await processBatchProofreading(
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
            
            // 4. 解析合并的JSON结果并导入
            ui.showGeneralMessage(`轮次 ${currentRound + 1}/${totalRounds}: 导入校对结果...`, "info", false);
            ui.updateProgressBar((currentRound / totalRounds) * 100 + (1 / totalRounds) * 90, '导入校对结果...');
            await importProofreadingResult(mergeJsonResults(allBatchResults));
        }
        
        // 完成
        ui.updateProgressBar(100, '校对完成！');
        ui.showGeneralMessage(`AI校对完成，共 ${totalRounds} 轮`, "success");
    } catch (error) {
        console.error("AI校对过程出错:", error);
        ui.showGeneralMessage(`校对失败: ${error.message}`, "error");
        
        // 重置进度条
        ui.updateProgressBar(0, '校对已取消');
        setTimeout(() => {
            $("#translationProgressBar").hide();
        }, 1000); // 给用户一秒时间看到进度条重置
    } finally {
        // 恢复状态
        isProofreadingInProgress = false;
        state.setBatchTranslationInProgress(false);
        ui.updateButtonStates();
    }
}

/**
 * 导出文本为JSON
 * 与高质量翻译相同但导出已翻译文本
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
        const translatedTexts = image.bubbleTexts || [];
        
        // 构建该图片的文本数据
        const imageTextData = {
            imageIndex: imageIndex,
            bubbles: []
        };
        
        // 构建每个气泡的文本数据
        for (let bubbleIndex = 0; bubbleIndex < originalTexts.length; bubbleIndex++) {
            const original = originalTexts[bubbleIndex] || '';
            const translated = bubbleIndex < translatedTexts.length ? translatedTexts[bubbleIndex] : '';
            
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
                translated: translated,
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
    const result = [];
    const allImages = state.images;
    
    for (let i = 0; i < allImages.length; i++) {
        // 使用原始图像数据，不使用翻译后的图像
        const base64Data = allImages[i].originalDataURL.split(',')[1];
        result.push(base64Data);
    }
    
    return result;
}

/**
 * 分批处理校对
 */
async function processBatchProofreading(jsonData, imageBase64Array, batchSize, sessionResetFrequency, provider, apiKey, modelName, customBaseUrl, rpmLimit, lowReasoning, prompt, forceJsonOutput) {
    const totalImages = imageBase64Array.length;
    const totalBatches = Math.ceil(totalImages / batchSize);
    
    // 显示批次进度
    ui.updateProgressBar(
        (currentRound / totalRounds) * 100 + (1 / totalRounds) * 50,
        `轮次 ${currentRound + 1}/${totalRounds}: 0/${totalBatches}`
    );
    
    // 创建限流器
    const rateLimiter = createRateLimiter(rpmLimit);
    
    // 跟踪批次计数，用于决定何时重置会话
    let batchCount = 0;
    let sessionId = generateSessionId();
    let successCount = 0; // 添加成功计数
    
    for (let batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
        // 更新进度
        const progressPercent = (currentRound / totalRounds) * 100 + (1 / totalRounds) * 50 + (1 / totalRounds) * 40 * (batchIndex / totalBatches);
        ui.updateProgressBar(
            progressPercent,
            `轮次 ${currentRound + 1}/${totalRounds}: ${batchIndex + 1}/${totalBatches}`
        );
        
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
        const maxRetries = state.proofreadingMaxRetries || 2;
        let retryCount = 0;
        let success = false;
        
        while (retryCount <= maxRetries && !success) {
            try {
                // 等待速率限制
                await rateLimiter.waitForTurn();
                
                // 发送批次到AI
                const result = await callAiForProofreading(
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
                    successCount++; // 增加成功计数
                    success = true;
                }
                
                // 增加批次计数
                batchCount++;
                
            } catch (error) {
                retryCount++;
                if (retryCount <= maxRetries) {
                    console.log(`轮次 ${currentRound + 1}, 批次 ${batchIndex + 1} 校对失败，第 ${retryCount}/${maxRetries} 次重试...`);
                    ui.showGeneralMessage(`轮次 ${currentRound + 1}, 批次 ${batchIndex + 1} 失败，正在重试 (${retryCount}/${maxRetries})...`, "warning", true);
                    await new Promise(r => setTimeout(r, 1000)); // 等待1秒后重试
                } else {
                    console.error(`轮次 ${currentRound + 1}, 批次 ${batchIndex + 1} 校对最终失败:`, error);
                    ui.showGeneralMessage(`轮次 ${currentRound + 1}, 批次 ${batchIndex + 1} 校对失败: ${error.message}`, "error", true);
                    // 继续处理下一批次
                }
            }
        }
    }
    
    // 完成所有批次
    ui.updateProgressBar(
        (currentRound / totalRounds) * 100 + (1 / totalRounds) * 90, 
        `轮次 ${currentRound + 1}/${totalRounds}: ${totalBatches}/${totalBatches}`
    );
    
    // 如果所有批次都失败，抛出错误
    if (successCount === 0) {
        throw new Error(`轮次 ${currentRound + 1} 校对完全失败，请检查API设置或校对提示词`);
    }
}

/**
 * 调用AI进行校对
 */
async function callAiForProofreading(imageBase64Array, jsonData, provider, apiKey, modelName, customBaseUrl, lowReasoning, prompt, sessionId, forceJsonOutput) {
    // 构建提示词和图片
    const jsonString = JSON.stringify(jsonData, null, 2);
    const messages = [
        {
            role: "system",
            content: "你是一个专业的漫画翻译校对助手，能够根据漫画图像内容和上下文对已有翻译进行校对和润色。"
        },
        {
            role: "user",
            content: [
                {
                    type: "text",
                    text: prompt + "\n\n以下是JSON数据，包含原文和已有译文:\n```json\n" + jsonString + "\n```\n请在保持JSON格式的情况下，校对每个bubble的translated字段，使翻译更加准确、自然、符合语境。"
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
    const noThinkingMethod = state.proofreadingNoThinkingMethod || 'gemini';
    
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
    
    console.log(`正在校对批次，使用服务商: ${provider}, 模型: ${modelName}`);
    
    // 发送API请求
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
                    throw new Error("解析AI返回的校对结果失败");
                }
            }
        } else {
            throw new Error("AI未返回有效内容");
        }
    } catch (error) {
        console.error("校对API调用失败:", error);
        throw error;
    }
}

/**
 * 从JSON数据中过滤指定范围的图片数据
 */
function filterJsonForBatch(jsonData, startIdx, endIdx) {
    if (!jsonData || !Array.isArray(jsonData)) {
        return [];
    }
    
    return jsonData.filter(item => 
        item.imageIndex >= startIdx && item.imageIndex < endIdx
    );
}

/**
 * 合并所有批次的JSON结果
 */
function mergeJsonResults(batchResults) {
    if (!batchResults || !Array.isArray(batchResults) || batchResults.length === 0) {
        console.warn("没有批次结果可合并");
        return [];
    }
    
    // 合并所有批次结果
    const mergedResult = [];
    
    // 遍历每个批次结果
    for (const batchResult of batchResults) {
        // 确保批次结果是有效的
        if (!batchResult || !Array.isArray(batchResult)) {
            console.warn("跳过无效的批次结果");
            continue;
        }
        
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
 * 导入校对结果
 */
async function importProofreadingResult(importedData) {
    if (!importedData || !Array.isArray(importedData) || importedData.length === 0) {
        console.warn("没有有效的校对数据可导入");
        ui.showGeneralMessage("没有有效的校对结果可导入，可能是校对失败或返回格式不正确", "warning");
        return; // 直接返回，而不是抛出错误
    }
    
    // 导入main模块
    const main = await import('./main.js');
    
    // 保存当前图片索引
    const originalImageIndex = state.currentImageIndex;
    
    // 获取当前的全局设置作为默认值，使用保存的文本样式设置（与高质量翻译保持一致）
    const currentFontSize = savedFontSize || parseInt($('#fontSize').val());
    const currentAutoFontSize = savedAutoFontSize !== null ? savedAutoFontSize : $('#autoFontSize').prop('checked');
    const currentFontFamily = savedFontFamily || $('#fontFamily').val();
    // 确保 currentTextDirection 不是 'auto'
    const rawTextDirection = savedTextDirection || $('#layoutDirection').val();
    const currentTextDirection = (rawTextDirection === 'auto') ? 'vertical' : rawTextDirection;
    const currentTextColor = savedTextColor || $('#textColor').val();
    const currentFillColor = savedFillColor || $('#fillColor').val();
    const currentRotationAngle = savedRotationAngle || 0;  // 全局角度已移除
    // 描边设置
    const currentStrokeEnabled = state.strokeEnabled;
    const currentStrokeColor = state.strokeColor;
    const currentStrokeWidth = state.strokeWidth;
    
    console.log("AI校对导入结果使用的文本样式:", {
        fontFamily: currentFontFamily,
        fontSize: currentFontSize,
        textDirection: currentTextDirection,
        strokeEnabled: currentStrokeEnabled
    });
    
    console.log(`轮次 ${currentRound + 1}/${totalRounds}: 开始导入校对结果`);
    ui.updateProgressBar((currentRound / totalRounds) * 100 + (1 / totalRounds) * 92, '更新图片数据...');
    
    try {
        // 创建一个队列，用于存储所有渲染任务
        const renderTasks = [];
        
        // 遍历导入的数据
        const totalImages = importedData.length;
        let processedImages = 0;
        
        // 处理每个图片的校对结果
        for (const imageData of importedData) {
            processedImages++;
            ui.updateProgressBar(
                (currentRound / totalRounds) * 100 + (1 / totalRounds) * (92 + processedImages / totalImages * 5), 
                `处理图片 ${processedImages}/${totalImages}`
            );
            
            const imageIndex = imageData.imageIndex;
            
            // 检查图片索引是否有效
            if (imageIndex < 0 || imageIndex >= state.images.length) {
                console.warn(`无效的图片索引: ${imageIndex}, 跳过该图片`);
                continue;
            }
            
            // 检查imageData.bubbles是否存在且为数组
            if (!imageData.bubbles || !Array.isArray(imageData.bubbles) || imageData.bubbles.length === 0) {
                console.warn(`图片 ${imageIndex}: 没有有效的气泡数据，跳过该图片`);
                continue;
            }
            
            // 获取该图片的当前数据
            const image = state.images[imageIndex];
            let imageUpdated = false;
            
            // 确保必要的数组存在
            if (!image.bubbleTexts) image.bubbleTexts = [];
            if (!image.originalTexts) image.originalTexts = [];
            
            // 处理每个气泡的校对结果
            for (const bubble of imageData.bubbles) {
                const bubbleIndex = bubble.bubbleIndex;
                const proofreadText = bubble.translated || '';
                // 确保 textDirection 不是 'auto'，如果是则使用默认的 currentTextDirection
                let textDirection = bubble.textDirection;
                if (!textDirection || textDirection === 'auto') {
                    textDirection = currentTextDirection;
                }
                
                // 检查气泡索引是否有效
                if (bubbleIndex < 0 || bubbleIndex >= image.bubbleCoords.length) {
                    console.warn(`图片 ${imageIndex}: 跳过无效的气泡索引 ${bubbleIndex}`);
                    continue;
                }
                
                // 确保bubbleIndex有效
                while (image.bubbleTexts.length <= bubbleIndex) {
                    image.bubbleTexts.push('');
                }
                
                // 更新翻译文本
                image.bubbleTexts[bubbleIndex] = proofreadText;
                
                // 更新排版方向 (需要创建或更新bubbleStates和bubbleStates)
                if (textDirection && textDirection !== 'auto') {
                    // 如果图片没有bubbleStates或长度不匹配，则初始化它
                    if (!image.bubbleStates || 
                        !Array.isArray(image.bubbleStates) || 
                        image.bubbleStates.length !== image.bubbleCoords.length) {
                        // 创建新的气泡设置，使用保存的文本样式设置
                        const detectedAngles = image.bubbleAngles || [];
                        const newStates = [];
                        for (let i = 0; i < image.bubbleCoords.length; i++) {
                            const bubbleTextDirection = (i === bubbleIndex) ? textDirection : currentTextDirection;
                            // 计算自动排版方向
                            let autoDir = bubbleTextDirection;
                            if (image.bubbleCoords[i] && image.bubbleCoords[i].length >= 4) {
                                const [x1, y1, x2, y2] = image.bubbleCoords[i];
                                autoDir = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal';
                            }
                            newStates.push({
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
                        image.bubbleStates = newStates;
                    } else {
                        // 更新现有bubbleStates中的textDirection，保持其他设置不变
                        if (!image.bubbleStates[bubbleIndex]) {
                            const bubbleDetectedAngle = (image.bubbleAngles && image.bubbleAngles[bubbleIndex]) || currentRotationAngle;
                            // 计算自动排版方向
                            let autoDir = textDirection;
                            if (image.bubbleCoords[bubbleIndex] && image.bubbleCoords[bubbleIndex].length >= 4) {
                                const [x1, y1, x2, y2] = image.bubbleCoords[bubbleIndex];
                                autoDir = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal';
                            }
                            image.bubbleStates[bubbleIndex] = {
                                translatedText: image.bubbleTexts[bubbleIndex] || "",
                                fontSize: currentFontSize,
                                autoFontSize: currentAutoFontSize,
                                fontFamily: currentFontFamily,
                                textDirection: textDirection,
                                autoTextDirection: autoDir,  // 自动检测的排版方向
                                position: { x: 0, y: 0 },
                                textColor: currentTextColor,
                                rotationAngle: bubbleDetectedAngle,
                                fillColor: currentFillColor,
                                strokeEnabled: currentStrokeEnabled,
                                strokeColor: currentStrokeColor,
                                strokeWidth: currentStrokeWidth
                            };
                        } else {
                            // 更新 translatedText 和 textDirection
                            image.bubbleStates[bubbleIndex].translatedText = proofreadText;
                            image.bubbleStates[bubbleIndex].textDirection = textDirection;
                        }
                    }
                }
                
                imageUpdated = true;
            }
            
            // 如果图片有更新，添加到渲染队列
            if (imageUpdated) {
                // 添加到渲染队列
                if (image.translatedDataURL) {
                    renderTasks.push(async () => {
                        // 借用edit_mode.js中的渲染逻辑，但不切换图片
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
        ui.updateProgressBar(
            (currentRound / totalRounds) * 100 + (1 / totalRounds) * 97, 
            "开始渲染图片..."
        );
        ui.showGeneralMessage(`轮次 ${currentRound + 1}/${totalRounds}: 正在渲染图片...`, "info", false);
        
        // 执行所有渲染任务
        for (let i = 0; i < renderTasks.length; i++) {
            ui.updateProgressBar(
                (currentRound / totalRounds) * 100 + (1 / totalRounds) * (97 + i / renderTasks.length * 3), 
                `渲染图片 ${i+1}/${renderTasks.length}`
            );
            await renderTasks[i]();
        }
        
        // 全部导入完成后，回到最初的图片
        main.switchImage(originalImageIndex);
        
        console.log(`轮次 ${currentRound + 1}/${totalRounds}: 校对结果导入完成`);
        
    } catch (error) {
        console.error("导入校对结果时出错:", error);
        ui.showGeneralMessage(`导入校对结果失败: ${error.message}`, "error");
        // 这里不再抛出错误，让流程继续执行
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
 * 创建速率限制器
 * @param {number} rpmLimit - 每分钟请求限制
 */
function createRateLimiter(rpmLimit) {
    // 默认RPM为10
    const rpm = rpmLimit || 10;
    // 计算每请求延迟时间（毫秒）
    const delayMs = Math.ceil(60000 / rpm);
    
    // 上次请求时间
    let lastRequestTime = 0;
    
    return {
        // 等待轮到本次请求
        waitForTurn: async function() {
            const now = Date.now();
            const timeElapsed = now - lastRequestTime;
            
            // 如果距离上次请求的时间不足最小延迟，则等待
            if (timeElapsed < delayMs) {
                const waitTime = delayMs - timeElapsed;
                await new Promise(resolve => setTimeout(resolve, waitTime));
            }
            
            // 更新上次请求时间
            lastRequestTime = Date.now();
        }
    };
}

/**
 * 生成会话ID
 */
function generateSessionId() {
    return Date.now().toString(36) + Math.random().toString(36).substring(2, 15);
}

/**
 * 初始化校对设置UI
 */
export function initProofreadingUI() {
    // 初始化校对设置弹窗中的启用开关
    $('#proofreadingEnabled').prop('checked', state.isProofreadingEnabled);
    
    // 清空并重新生成轮次列表
    const roundsContainer = $('#proofreadingRoundsContainer');
    roundsContainer.empty();
    
    // 添加已有轮次
    state.proofreadingRounds.forEach((round, index) => {
        addRoundToUI(round, index);
    });
    
    // 如果没有轮次，添加一个默认轮次
    if (state.proofreadingRounds.length === 0) {
        addNewRound();
    }
    
    // 绑定"添加轮次"按钮事件
    $('#addRoundButton').off('click').on('click', addNewRound);
}

/**
 * 添加新的校对轮次
 */
function addNewRound() {
    // 创建默认轮次配置
    const newRound = {
        name: `轮次 ${state.proofreadingRounds.length + 1}`,
        provider: 'siliconflow',
        apiKey: '',
        modelName: '',
        customBaseUrl: '',
        batchSize: 3,
        sessionReset: 20,
        rpmLimit: 7,
        lowReasoning: false,
        forceJsonOutput: true,
        prompt: constants.DEFAULT_PROOFREADING_PROMPT
    };
    
    // 添加到state
    state.proofreadingRounds.push(newRound);
    
    // 添加到UI
    addRoundToUI(newRound, state.proofreadingRounds.length - 1);
}

/**
 * 将轮次添加到UI
 */
function addRoundToUI(round, index) {
    const roundsContainer = $('#proofreadingRoundsContainer');
    
    // 创建轮次HTML
    const roundHTML = `
    <div class="proofreading-round" data-index="${index}">
        <div class="round-header">
            <input type="text" class="round-name" value="${round.name}" placeholder="轮次名称">
            <button class="remove-round-button">删除</button>
        </div>
        <div class="round-settings">
            <div>
                <label>AI服务商:</label>
                <select class="round-provider">
                    <option value="siliconflow" ${round.provider === 'siliconflow' ? 'selected' : ''}>SiliconFlow</option>
                    <option value="deepseek" ${round.provider === 'deepseek' ? 'selected' : ''}>DeepSeek</option>
                    <option value="volcano" ${round.provider === 'volcano' ? 'selected' : ''}>火山引擎</option>
                    <option value="gemini" ${round.provider === 'gemini' ? 'selected' : ''}>Google Gemini</option>
                    <option value="custom_openai" ${round.provider === 'custom_openai' ? 'selected' : ''}>自定义OpenAI兼容服务</option>
                </select>
            </div>
            <div>
                <label>API Key:</label>
                <div class="password-input-wrapper">
                    <input type="text" class="round-api-key secure-input" value="${round.apiKey}" placeholder="填写API Key" autocomplete="off">
                    <button type="button" class="password-toggle-btn" tabindex="-1">
                        <span class="eye-icon">👁</span>
                        <span class="eye-off-icon">👁‍🗨</span>
                    </button>
                </div>
            </div>
            <div>
                <label>模型名称:</label>
                <div class="model-input-with-fetch">
                    <input type="text" class="round-model-name" value="${round.modelName}" placeholder="如 gemini-2.5-flash-preview-05-20">
                    <button type="button" class="fetch-models-btn round-fetch-models-btn" title="获取可用模型列表">
                        <span class="fetch-icon">🔍</span>
                        <span class="fetch-text">获取模型</span>
                    </button>
                </div>
                <div class="round-model-select-container model-select-container" style="display:none;">
                    <select class="round-model-select model-select">
                        <option value="">-- 选择模型 --</option>
                    </select>
                    <span class="round-model-count model-count"></span>
                </div>
            </div>
            <div class="custom-base-url-div" style="${round.provider === 'custom_openai' ? '' : 'display:none;'}">
                <label>Base URL:</label>
                <input type="text" class="round-custom-base-url" value="${round.customBaseUrl}" placeholder="如 https://your-api-endpoint.com">
            </div>
            <div>
                <label>批次大小:</label>
                <input type="number" class="round-batch-size" value="${round.batchSize}" min="1" max="10">
            </div>
            <div>
                <label>会话重置频率:</label>
                <input type="number" class="round-session-reset" value="${round.sessionReset}" min="1">
            </div>
            <div>
                <label>RPM限制:</label>
                <input type="number" class="round-rpm-limit" value="${round.rpmLimit}" min="1" max="100">
            </div>
            <div>
                <label>关闭思考功能:</label>
                <input type="checkbox" class="round-low-reasoning" ${round.lowReasoning ? 'checked' : ''}>
                <span class="input-hint">部分模型支持关闭思考以加快响应速度</span>
            </div>
            <div>
                <label>取消思考方法:</label>
                <select class="round-no-thinking-method">
                    <option value="gemini" ${state.proofreadingNoThinkingMethod === 'gemini' ? 'selected' : ''}>Gemini (reasoning_effort)</option>
                    <option value="volcano" ${state.proofreadingNoThinkingMethod === 'volcano' ? 'selected' : ''}>火山引擎 (thinking: null)</option>
                </select>
                <span class="input-hint">选择使用哪种方式取消思考，仅在开启"关闭思考"时有效</span>
            </div>
            <div>
                <label>强制JSON输出:</label>
                <input type="checkbox" class="round-force-json" ${round.forceJsonOutput ? 'checked' : ''}>
                <span class="input-hint">使用API参数强制输出JSON，仅支持OpenAI兼容接口</span>
            </div>
            <div>
                <label>校对提示词:</label>
                <textarea class="round-prompt" rows="4">${round.prompt}</textarea>
            </div>
        </div>
    </div>
    `;
    
    // 添加到容器
    roundsContainer.append(roundHTML);
    
    // 绑定事件
    const $roundElement = roundsContainer.find(`.proofreading-round[data-index="${index}"]`);
    
    // 删除轮次按钮
    $roundElement.find('.remove-round-button').on('click', function() {
        removeRound(index);
    });
    
    // 轮次名称变更
    $roundElement.find('.round-name').on('change', function() {
        updateRoundProperty(index, 'name', $(this).val());
    });
    
    // 服务商变更
    $roundElement.find('.round-provider').on('change', function() {
        const provider = $(this).val();
        updateRoundProperty(index, 'provider', provider);
        
        // 显示/隐藏自定义Base URL输入框
        $roundElement.find('.custom-base-url-div').toggle(provider === 'custom_openai');
    });
    
    // 其他输入字段变更
    $roundElement.find('.round-api-key').on('change', function() {
        updateRoundProperty(index, 'apiKey', $(this).val());
    });
    
    $roundElement.find('.round-model-name').on('change', function() {
        updateRoundProperty(index, 'modelName', $(this).val());
    });
    
    $roundElement.find('.round-custom-base-url').on('change', function() {
        updateRoundProperty(index, 'customBaseUrl', $(this).val());
    });
    
    $roundElement.find('.round-batch-size').on('change', function() {
        updateRoundProperty(index, 'batchSize', parseInt($(this).val()) || 3);
    });
    
    $roundElement.find('.round-session-reset').on('change', function() {
        updateRoundProperty(index, 'sessionReset', parseInt($(this).val()) || 20);
    });
    
    $roundElement.find('.round-rpm-limit').on('change', function() {
        updateRoundProperty(index, 'rpmLimit', parseInt($(this).val()) || 7);
    });
    
    $roundElement.find('.round-low-reasoning').on('change', function() {
        updateRoundProperty(index, 'lowReasoning', $(this).is(':checked'));
    });
    
    $roundElement.find('.round-no-thinking-method').on('change', function() {
        state.setProofreadingNoThinkingMethod($(this).val());
    });
    
    $roundElement.find('.round-force-json').on('change', function() {
        updateRoundProperty(index, 'forceJsonOutput', $(this).is(':checked'));
    });
    
    $roundElement.find('.round-prompt').on('change', function() {
        updateRoundProperty(index, 'prompt', $(this).val());
    });
}

/**
 * 更新轮次属性
 */
function updateRoundProperty(index, property, value) {
    if (index < 0 || index >= state.proofreadingRounds.length) {
        console.error(`无效的轮次索引: ${index}`);
        return;
    }
    
    state.proofreadingRounds[index][property] = value;
}

/**
 * 移除轮次
 */
function removeRound(index) {
    if (index < 0 || index >= state.proofreadingRounds.length) {
        console.error(`无效的轮次索引: ${index}`);
        return;
    }
    
    // 从state中移除
    state.proofreadingRounds.splice(index, 1);
    
    // 重新初始化UI
    initProofreadingUI();
}

/**
 * 保存校对设置
 */
export function saveProofreadingSettings() {
    // 从UI获取启用状态
    const isEnabled = $('#proofreadingEnabled').is(':checked');
    state.setProofreadingEnabled(isEnabled);
    
    // 轮次已经通过各自的事件处理程序更新到state中
    
    ui.showGeneralMessage("校对设置已保存", "success");
    ui.hideProofreadingSettingsModal();
} 