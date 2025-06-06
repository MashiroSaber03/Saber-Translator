// src/app/static/js/session.js

import * as state from './state.js'; // 引入状态模块以获取当前状态
import * as ui from './ui.js';     // 引入 UI 模块以显示消息和加载状态
import * as api from './api.js';     // 引入 API 模块以调用后端接口
import * as main from './main.js'; // 需要 main 来切换图片
import * as editMode from './edit_mode.js'; // 需要退出编辑模式
import * as labelingMode from './labeling_mode.js';// 需要退出标注模式
import * as constants from './constants.js';

// --- 新增：自动存档相关变量 ---
let autoSaveTimeout = null; // 用于节流的 Timeout ID
const AUTO_SAVE_DELAY = 10000; // 自动存档延迟时间 (毫秒)，例如 10 秒
// --------------------------

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

    // 1. 收集 UI 设置
    // 注意：这里需要根据你的 index.html 中的 ID 来获取值
    const uiSettings = {
        targetLanguage: $('#targetLanguage').val(),
        sourceLanguage: $('#sourceLanguage').val(),
        fontSize: $('#fontSize').val(), // 保存原始值（可能是数字或 'auto'）
        autoFontSize: $('#autoFontSize').is(':checked'),
        fontFamily: $('#fontFamily').val(),
        layoutDirection: $('#layoutDirection').val(),
        textColor: $('#textColor').val(),
        rotationAngle: parseFloat($('#rotationAngle').val() || '0'),
        modelProvider: $('#modelProvider').val(),
        // apiKey: $('#apiKey').val(), // 安全考虑：通常不建议保存 API Key 到会话文件
        modelName: $('#modelName').val(),
        promptContent: $('#promptContent').val(), // 保存当前编辑框的内容
        // 可以考虑保存选中的提示词名称
        // selectedPromptName: $('#promptDropdownButton').text().replace(' ▼', ''), // 获取按钮文本作为参考
        enableTextboxPrompt: $('#enableTextboxPrompt').is(':checked'),
        textboxPromptContent: $('#textboxPromptContent').val(),
        // selectedTextboxPromptName: $('#textboxPromptDropdownButton').text().replace(' ▼', ''),
        useInpaintingMethod: $('#useInpainting').val(), // 保存 'false', 'true', 或 'lama'
        fillColor: $('#fillColor').val(),
        inpaintingStrength: parseFloat($('#inpaintingStrength').val()),
        blendEdges: $('#blendEdges').is(':checked'),
        rpmLimitTranslation: state.rpmLimitTranslation,         // <--- 新增
        rpmLimitAiVisionOcr: state.rpmLimitAiVisionOcr,       // <--- 新增

        // === 新增：收集描边设置 START ===
        enableTextStroke: state.enableTextStroke,       // 从 state.js 获取
        textStrokeColor: state.textStrokeColor,         // 从 state.js 获取
        textStrokeWidth: state.textStrokeWidth,         // 从 state.js 获取
        // === 新增：收集描边设置 END ===

        // 可以添加其他需要保存的 UI 状态，如图片缩放比例等
        // imageZoom: $('#imageSize').val()
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
 * 触发"另存为"会话流程。
 * 总是弹出一个输入框让用户输入新会话名称，然后调用 API 保存。
 * @param {string} [defaultName=''] - 可选的默认会话名称（可以基于当前名称）。
 */
export function triggerSaveAsSession(defaultName = '') {
    console.log("触发另存为会话流程...");

    const currentName = state.currentSessionName || defaultName; // 建议使用当前名称作为默认值

    // 1. 弹出输入框获取会话名称
    const sessionName = prompt("请输入新的会话名称:", currentName); // 使用当前名称作为提示

    // 2. 检查用户是否输入了名称或取消了操作
    if (sessionName === null) { // 用户点击了取消
        console.log("用户取消了保存会话。");
        ui.showGeneralMessage("保存已取消。", "info", false, 2000);
        return;
    }
    if (!sessionName.trim()) { // 用户输入了空名称
        ui.showGeneralMessage("会话名称不能为空！", "warning");
        console.warn("保存失败：会话名称为空。");
        // 可以选择重新提示或直接返回
        // triggerSaveAsSession(defaultName); // 重新提示
        return;
    }

    const finalSessionName = sessionName.trim();
    console.log(`用户输入的会话名称: ${finalSessionName}`);

    // 3. 收集当前状态数据
    const currentSessionData = collectCurrentSessionData();
    if (!currentSessionData) {
        // collectCurrentSessionData 内部已经显示了错误消息
        return;
    }

    ui.showLoading(`正在另存为会话 "${finalSessionName}"...`);

    api.saveSessionApi(finalSessionName, currentSessionData)
        .then(response => {
            ui.hideLoading();
            if (response.success) {
                ui.showGeneralMessage(response.message || `会话已成功另存为 "${finalSessionName}"！`, "success");
                // **重要：** 另存为成功后，将新名称设为当前会话名称
                state.setCurrentSessionName(finalSessionName);
            } else {
                throw new Error(response.error || "另存为会话失败，请查看后端日志。");
            }
        })
        .catch(error => {
            ui.hideLoading();
            console.error(`另存为会话 "${finalSessionName}" 失败:`, error);
            ui.showGeneralMessage(`另存为失败: ${error.message || '未知错误'}`, "error");
        });
}

/**
 * 触发"保存"当前会话流程。
 * 如果当前有活动会话名称，则直接覆盖保存；否则，行为同"另存为"。
 */
export function triggerSaveCurrentSession() {
    console.log("触发保存当前会话流程...");

    const sessionNameToSave = state.currentSessionName;

    if (sessionNameToSave) {
        // 当前有活动会话，直接覆盖保存
        console.log(`准备覆盖保存会话: ${sessionNameToSave}`);
        const currentSessionData = collectCurrentSessionData();
        if (!currentSessionData) return;

        ui.showLoading(`正在保存会话 "${sessionNameToSave}"...`);

        api.saveSessionApi(sessionNameToSave, currentSessionData)
            .then(response => {
                ui.hideLoading();
                if (response.success) {
                    ui.showGeneralMessage(response.message || `会话 "${sessionNameToSave}" 已更新！`, "success");
                    // 当前会话名称不需要变
                } else {
                    throw new Error(response.error || "保存会话失败，请查看后端日志。");
                }
            })
            .catch(error => {
                ui.hideLoading();
                console.error(`保存会话 "${sessionNameToSave}" 失败:`, error);
                ui.showGeneralMessage(`保存失败: ${error.message || '未知错误'}`, "error");
            });
    } else {
        // 没有活动会话名称，行为等同于"另存为"
        console.log("没有当前活动会话，执行\"另存为\"...");
        triggerSaveAsSession(); // 调用另存为函数
    }
}

/**
 * 获取会话列表并显示会话管理模态框。
 */
export function showSessionManager() {
    console.log("准备显示会话管理器...");

    // 1. 显示模态框，并显示加载状态
    ui.showSessionManagerModal();
    $("#sessionListContainer").html("<p>正在加载会话列表...</p>"); // 显示加载提示

    // 2. 调用 API 获取会话列表
    api.listSessionsApi()
        .then(response => {
            if (response.success && response.sessions) {
                console.log("成功获取会话列表:", response.sessions);
                // 3. 使用获取到的数据渲染列表
                ui.renderSessionList(response.sessions);
            } else {
                // API 调用成功但后端逻辑失败，或返回格式不对
                throw new Error(response.error || "获取会话列表失败，响应格式错误。");
            }
        })
        .catch(error => {
            console.error("获取会话列表失败:", error);
            // 在模态框内显示错误信息
            $("#sessionListContainer").html(`<p class="error">加载会话列表失败: ${error.message || '未知错误'}</p>`);
            // 不自动关闭模态框，让用户看到错误信息
        });
}

/**
 * 处理加载指定会话的逻辑。
 * @param {string} sessionName - 要加载的会话名称。
 */
export function handleLoadSession(sessionName) {
    console.log(`请求加载会话 (含描边恢复): ${sessionName}`);

    // 0. 退出可能存在的编辑或标注模式
    if (state.editModeActive) {
        editMode.exitEditMode();
    }
    if (state.isLabelingModeActive) {
        // 退出标注模式，但不提示保存（因为即将被加载的内容覆盖）
        labelingMode.exitLabelingMode(false); // 假设 exitLabelingMode 可接受一个参数跳过确认
        // 如果 exitLabelingMode 没有此参数，需要先手动重置状态：
        // state.setLabelingModeActive(false);
        // state.setManualCoords([], true);
        // state.setSelectedManualBoxIndex(-1);
        // ui.clearBoundingBoxes();
        // ui.toggleLabelingModeUI(false);
    }


    // 1. 显示加载状态 (可以使用通用消息，或者在模态框内显示)
    ui.showLoading(`正在加载会话 "${sessionName}"...`);
    ui.hideSessionManagerModal(); // 先关闭选择列表模态框

    // 2. 调用 API 加载会话数据
    api.loadSessionApi(sessionName)
        .then(response => {
            if (response.success && response.session_data) {
                console.log("成功加载会话数据 (含描边):", response.session_data);
                const loadedData = response.session_data;
                const uiSettings = loadedData.ui_settings || {}; // <--- 在这里先获取 uiSettings

                // 3. 恢复状态 (state.js)
                try {
                    // 清除当前状态
                    state.clearImages(); // 使用 state 中的清除函数

                    // 恢复图片列表 (确保 state.setImages 处理所需的默认值)
                    state.setImages(loadedData.images || []);

                    // 恢复当前索引
                    // 确保索引在有效范围内
                    let newIndex = loadedData.currentImageIndex !== undefined ? loadedData.currentImageIndex : -1;
                    if (newIndex >= state.images.length || newIndex < 0) {
                        newIndex = state.images.length > 0 ? 0 : -1; // 索引无效则设为0或-1
                    }
                    state.setCurrentImageIndex(newIndex);
                    state.setCurrentSessionName(sessionName); // 设置当前会话名

                    console.log("前端状态已恢复。");

                } catch (stateError) {
                     console.error("恢复前端状态时出错:", stateError);
                     throw new Error("恢复应用状态失败，加载的数据可能已损坏。");
                }

                // 4. **先刷新界面显示 (让 switchImage 执行)**
                ui.renderThumbnails(); // 重新渲染缩略图
                if (state.currentImageIndex !== -1) {
                    // !!! 调用 switchImage，它可能会根据第一张图片覆盖 UI 设置 !!!
                    main.switchImage(state.currentImageIndex); // 切换到加载时的图片
                    ui.showResultSection(true);
                } else {
                    // 如果加载后没有图片了（理论上不应该发生）
                    ui.showResultSection(false);
                }
                
                // 5. **再次恢复 UI 设置 (覆盖 switchImage 的影响)**
                try {
                    console.log("重新应用加载的 UI 设置以覆盖 switchImage 的影响:", uiSettings);

                    // --- (这里是之前第 4 步的代码块，现在移到这里) ---
                    $('#targetLanguage').val(uiSettings.targetLanguage || 'zh');
                    $('#sourceLanguage').val(uiSettings.sourceLanguage || 'japan');
                    $('#fontSize').val(uiSettings.fontSize || state.defaultFontSize);
                    $('#autoFontSize').prop('checked', uiSettings.autoFontSize || false);
                    $('#fontFamily').val(uiSettings.fontFamily || state.defaultFontFamily);
                    $('#layoutDirection').val(uiSettings.layoutDirection || state.defaultLayoutDirection);
                    $('#textColor').val(uiSettings.textColor || state.defaultTextColor);
                    $('#rotationAngle').val(uiSettings.rotationAngle || 0);
                    $('#rotationAngleValue').text((uiSettings.rotationAngle || 0) + '°');
                    $('#modelProvider').val(uiSettings.modelProvider || 'siliconflow');
                    $('#modelName').val(uiSettings.modelName || '');
                    $('#promptContent').val(uiSettings.promptContent || state.defaultPromptContent);
                    $('#enableTextboxPrompt').prop('checked', uiSettings.enableTextboxPrompt || false);
                    $('#textboxPromptContent').val(uiSettings.textboxPromptContent || state.defaultTextboxPromptContent);
                    $('#useInpainting').val(uiSettings.useInpaintingMethod || 'false');
                    $('#fillColor').val(uiSettings.fillColor || state.defaultFillColor);
                    $('#inpaintingStrength').val(uiSettings.inpaintingStrength || 1.0);
                    $('#inpaintingStrengthValue').text($('#inpaintingStrength').val());
                    $('#blendEdges').prop('checked', uiSettings.blendEdges === undefined ? true : uiSettings.blendEdges);

                    // --- 新增：恢复rpm设置 ---
                    if (uiSettings.rpmLimitTranslation !== undefined) {
                        state.setrpmLimitTranslation(uiSettings.rpmLimitTranslation);
                    } else {
                        state.setrpmLimitTranslation(constants.DEFAULT_rpm_TRANSLATION); // 如果会话中没有，则用默认值
                    }
                    if (uiSettings.rpmLimitAiVisionOcr !== undefined) {
                        state.setrpmLimitAiVisionOcr(uiSettings.rpmLimitAiVisionOcr);
                    } else {
                        state.setrpmLimitAiVisionOcr(constants.DEFAULT_rpm_AI_VISION_OCR);
                    }
                    ui.updaterpmInputFields(); // 更新UI显示
                    // ------------------------

                    // === 新增：恢复描边设置 START ===
                    const enableStroke = uiSettings.enableTextStroke === undefined ? state.defaultEnableTextStroke : uiSettings.enableTextStroke;
                    const strokeColor = uiSettings.textStrokeColor || state.defaultTextStrokeColor;
                    const strokeWidth = uiSettings.textStrokeWidth === undefined ? state.defaultTextStrokeWidth : parseInt(uiSettings.textStrokeWidth);
                    
                    state.setEnableTextStroke(enableStroke);
                    state.setTextStrokeColor(strokeColor);
                    state.setTextStrokeWidth(strokeWidth);
                    
                    // 更新描边相关的UI控件
                    $('#enableTextStroke').prop('checked', enableStroke);
                    $('#textStrokeColor').val(strokeColor);
                    $('#textStrokeWidth').val(strokeWidth);
                    $("#textStrokeOptions").toggle(enableStroke); // 根据加载的值显示/隐藏
                    console.log("描边 UI 设置已恢复。");
                    // === 新增：恢复描边设置 END ===

                    // 再次触发 change 事件确保依赖 UI 更新
                    $('#modelProvider').trigger('change');
                    $('#useInpainting').trigger('change');
                    $('#enableTextboxPrompt').trigger('change');
                    $('#autoFontSize').trigger('change');
                    $('#enableTextStroke').trigger('change'); // 如果描边选项的显示依赖于此
                    // --- 结束 UI 恢复代码块 ---

                    console.log("UI 设置已最终恢复 (含描边)。");

                } catch (uiError) {
                    console.error("最终恢复 UI 设置时出错 (含描边):", uiError);
                    ui.showGeneralMessage("部分 UI 设置恢复失败，但会话数据已加载。", "warning");
                }

                // 6. 最后更新按钮状态
                ui.updateButtonStates(); // 更新所有按钮状态

                ui.hideLoading();
                ui.showGeneralMessage(`会话 "${sessionName}" 加载成功！`, "success");

            } else {
                throw new Error(response.error || `加载会话 "${sessionName}" 失败。`);
            }
        })
        .catch(error => {
            ui.hideLoading();
            console.error(`加载会话 "${sessionName}" 失败 (含描边恢复):`, error);
            ui.showGeneralMessage(`加载会话失败: ${error.message || '未知错误'}`, "error");
            // 加载失败后，可能需要重置界面到初始状态或保留当前状态
            // state.clearImages(); // 可以选择清空
            // ui.renderThumbnails();
            // ui.showResultSection(false);
            // ui.updateButtonStates();
            state.setCurrentSessionName(null); // 加载失败也清除当前会话名
        });
}

/**
 * 处理删除指定会话的逻辑。
 * @param {string} sessionName - 要删除的会话名称。
 */
export function handleDeleteSession(sessionName) {
    console.log(`请求删除会话: ${sessionName}`);

    // 1. 确认是否删除
    if (!confirm(`确定要删除会话 "${sessionName}" 吗？此操作不可撤销。`)) {
        console.log("用户取消了删除操作。");
        ui.showGeneralMessage("删除已取消。", "info", false, 2000);
        return;
    }

    // 2. 调用 API 删除
    console.log(`正在删除会话 "${sessionName}"...`);
    // 可以显示一个小的加载指示

    api.deleteSessionApi(sessionName)
        .then(response => {
            if (response.success) {
                ui.showGeneralMessage(response.message || `会话 "${sessionName}" 已删除。`, "success");
                console.log("删除成功。");
                // 3. 刷新列表
                showSessionManager(); // 重新加载并显示列表
            } else {
                throw new Error(response.error || `删除会话 "${sessionName}" 失败。`);
            }
        })
        .catch(error => {
            console.error(`删除会话失败:`, error);
            ui.showGeneralMessage(`删除失败: ${error.message || '未知错误'}`, "error");
        });
}

/**
 * 处理重命名指定会话的逻辑。
 * @param {string} oldName - 当前的会话名称。
 */
export function handleRenameSession(oldName) {
    console.log(`请求重命名会话: ${oldName}`);

    // 1. 弹出输入框获取新名称
    const newName = prompt(`请输入会话 "${oldName}" 的新名称:`, oldName);

    // 2. 验证输入
    if (newName === null) {
        console.log("用户取消了重命名操作。");
        ui.showGeneralMessage("重命名已取消。", "info", false, 2000);
        return;
    }
    if (!newName.trim()) {
        ui.showGeneralMessage("新会话名称不能为空！", "warning");
        return;
    }
    if (newName.trim() === oldName) {
        ui.showGeneralMessage("新旧名称相同，无需重命名。", "info", false, 2000);
        return;
    }

    const finalNewName = newName.trim();

    // 3. 调用 API 重命名
    console.log(`正在重命名会话 "${oldName}" 为 "${finalNewName}"...`);
    // 可以显示一个小的加载指示

    api.renameSessionApi(oldName, finalNewName)
        .then(response => {
            if (response.success) {
                ui.showGeneralMessage(response.message || `会话已成功重命名为 "${finalNewName}"。`, "success");
                console.log("重命名成功。");
                // 4. 刷新列表
                showSessionManager(); // 重新加载并显示列表
            } else {
                throw new Error(response.error || `重命名会话失败。`);
            }
        })
        .catch(error => {
            console.error(`重命名会话失败:`, error);
            ui.showGeneralMessage(`重命名失败: ${error.message || '未知错误'}`, "error");
        });
}

// --- 新增：自动存档核心函数 ---
/**
 * 实际执行自动存档操作。
 */
function performAutoSave() {
    console.log("执行自动存档...");

    // 1. 检查是否有内容可保存
    if (!state.images || state.images.length === 0) {
        console.log("自动存档跳过：没有图片数据。");
        return;
    }

    // 2. 收集当前状态
    const currentSessionData = collectCurrentSessionData(); // 复用之前的函数
    if (!currentSessionData) {
        console.error("自动存档失败：无法收集会话数据。");
        return; // 收集失败则不保存
    }

    // 3. 调用 API 保存到自动存档槽位
    // 注意：自动存档不显示"加载中"提示，静默进行
    api.saveSessionApi(constants.AUTO_SAVE_SLOT_NAME, currentSessionData)
        .then(response => {
            if (response.success) {
                console.log("自动存档成功。");
                // 短暂显示一个成功的提示消息
                ui.showGeneralMessage("进度已自动保存", "success", false, 1500);
                // 可选：如果会话管理器当前打开，可以尝试刷新列表（但可能引入复杂性）
                // if ($("#sessionManagerModal").is(":visible")) {
                //     showSessionManager(); // 重新加载列表
                // }
            } else {
                // 保存失败，只记录日志，不打扰用户
                console.error("自动存档失败:", response.error || "未知错误");
                // ui.showGeneralMessage("自动存档失败", "warning", false, 2000); // 可以选择性提示
            }
        })
        .catch(error => {
            // API 调用本身失败
            console.error("调用自动存档 API 时出错:", error);
            // ui.showGeneralMessage("自动存档出错", "error", false, 2000); // 可以选择性提示
        });
}

/**
 * 触发自动存档（带节流）。
 * 在重要操作后调用此函数。
 * @param {number} [delay=AUTO_SAVE_DELAY] - 延迟执行保存的时间（毫秒）。
 */
export function triggerAutoSave(delay = AUTO_SAVE_DELAY) {
    // 检查是否正在进行批量翻译，如果是则跳过自动保存
    if (state.isBatchTranslationInProgress) {
        console.log("批量翻译进行中，跳过自动存档");
        return;
    }
    
    // 检查自动存档是否启用，如果未启用则跳过
    if (!state.getAutoSaveEnabled()) {
        console.log("自动存档功能已禁用，跳过自动存档");
        return;
    }

    // 清除之前的延迟计时器（如果有）
    if (autoSaveTimeout) {
        clearTimeout(autoSaveTimeout);
        // console.log("自动存档计时器重置。");
    }

    // 设置新的延迟计时器
    // console.log(`计划在 ${delay / 1000} 秒后自动存档...`);
    autoSaveTimeout = setTimeout(() => {
        performAutoSave();
        autoSaveTimeout = null; // 执行后清除 ID
    }, delay);
}
// --------------------------

// --- TODO: 加载、删除、重命名等逻辑 ---
// export function handleDeleteSession(sessionName) { ... }
// export function handleRenameSession(oldName, newName) { ... } // 被 Session Manager 调用 