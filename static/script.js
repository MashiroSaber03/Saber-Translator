$(document).ready(function() {
    let images = [];
    let currentImageIndex = -1;
    const dropArea = $("#drop-area");
    const imageUploadInput = $("#imageUpload");
    const selectFileLink = $("#select-file-link");
    const translateButton = $("#translateButton");
    const translateAllButton = $("#translateAllButton");
    const prevImageButton = $("#prevImageButton");
    const nextImageButton = $("#nextImageButton");
    const loadingMessage = $("#loadingMessage");
    const loadingAnimation = $("#loadingAnimation");
    const errorMessage = $("#errorMessage");
    const resultSection = $("#result-section");
    const translatedImageDisplay = $("#translatedImageDisplay");
    const downloadButton = $("#downloadButton");
    const downloadAllImagesButton = $("#downloadAllImagesButton");
    const detectedTextInfo = $("#detectedTextInfo");
    const detectedTextList = $("#detectedTextList");
    const thumbnailList = $("#thumbnailList");
    const modelProviderSelect = $("#modelProvider");
    const modelNameInput = $("#modelName");
    const modelSuggestionsDiv = $("#model-suggestions");
    const translatingMessage = $("#translatingMessage");
    const downloadingMessage = $("#downloadingMessage");
    const fontSizeInput = $("#fontSize");
    const fontFamilySelect = $("#fontFamily");
    const clearAllImagesButton = $("#clearAllImagesButton");
    const deleteCurrentImageButton = $("#deleteCurrentImageButton");
    const applyFontSettingsToAllButton = $("#applyFontSettingsToAllButton");
    const layoutDirectionSelect = $("#layoutDirection");

    const defaultFontSize = fontSizeInput.val();
    const defaultFontFamily = fontFamilySelect.val();

    const promptContentTextarea = $("#promptContent");
    const savePromptButton = $("#savePromptButton");
    const rememberPromptCheckbox = $("#rememberPrompt");
    const promptNameInput = $("#promptName");
    const promptDropdownButton = $("#promptDropdownButton");
    const promptDropdown = $("#promptDropdown");

    let currentPromptContent = "";
    let defaultPromptContent = "";
    let savedPromptNames = [];

    function initializePromptSettings() {
        $.ajax({
            url: '/get_prompts',
            type: 'GET',
            dataType: 'json',
            success: function(response) {
                defaultPromptContent = response.default_prompt_content;
                currentPromptContent = defaultPromptContent;
                promptContentTextarea.val(currentPromptContent);
                savedPromptNames = response.prompt_names || [];
                populatePromptDropdown(savedPromptNames);
            },
            error: function(error) {
                console.error("获取提示词信息失败:", error);
                defaultPromptContent = "获取默认提示词失败，请检查后端";
                currentPromptContent = defaultPromptContent;
                promptContentTextarea.val(currentPromptContent);
            }
        });
    }

    initializePromptSettings();


    function populatePromptDropdown(promptNames) {
        promptDropdown.empty();
        if (promptNames && promptNames.length > 0) {
            const ul = $("<ul></ul>");
            promptNames.forEach(function(name) {
                const li = $("<li></li>").text(name).click(function() {
                    loadPromptContent(name);
                    promptDropdown.hide();
                });

                const deleteButton = $('<span class="delete-prompt-button">&times;</span>');
                deleteButton.click(function(e) {
                    e.stopPropagation();
                    deletePrompt(name);
                });
                li.append(deleteButton);
                ul.append(li);
            });
            promptDropdown.append(ul);
            promptDropdownButton.show();
        } else {
            promptDropdownButton.hide();
        }
    }

    function deletePrompt(promptName) {
        $.ajax({
            url: '/delete_prompt',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ prompt_name: promptName }),
            success: function(response) {
                alert("提示词 '" + promptName + "' 删除成功！");
                initializePromptSettings();
            },
            error: function(error) {
                console.error("删除提示词失败:", error);
                alert("删除提示词失败，请重试。");
            }
        });
    }


    function loadPromptContent(promptName) {
        if (promptName === "默认提示词") {
            currentPromptContent = defaultPromptContent;
            promptContentTextarea.val(currentPromptContent);
        } else {
            $.ajax({
                url: '/get_prompt_content',
                type: 'GET',
                data: { prompt_name: promptName },
                dataType: 'json',
                success: function(response) {
                    currentPromptContent = response.prompt_content;
                    promptContentTextarea.val(currentPromptContent);
                },
                error: function(error) {
                    console.error("加载提示词内容失败:", error);
                    alert("加载提示词内容失败");
                }
            });
        }
    }


    savePromptButton.click(function() {
        const promptName = promptNameInput.val();
        const promptContent = promptContentTextarea.val();

        if (rememberPromptCheckbox.is(':checked')) {
            if (!promptName) {
                alert("请为要保存的提示词输入名称。");
                return;
            }
            $.ajax({
                url: '/save_prompt',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ prompt_name: promptName, prompt_content: promptContent }),
                success: function(response) {
                    alert("提示词保存成功！");
                    initializePromptSettings();
                },
                error: function(error) {
                    console.error("保存提示词失败:", error);
                    alert("保存提示词失败，请重试。");
                }
            });
        } else {
            alert("提示词已应用，但未保存。下次启动程序将使用默认提示词。");
            currentPromptContent = promptContent;
        }
    });

    rememberPromptCheckbox.change(function() {
        promptNameInput.toggle(this.checked);
    });
    promptNameInput.hide();

    promptDropdownButton.click(function(e) {
        promptDropdown.toggle();
        e.stopPropagation();
    });

    $(document).click(function() {
        promptDropdown.hide();
    });


    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.on(eventName, function(e) {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.on(eventName, function() {
            dropArea.addClass('highlight');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.on(eventName, function() {
            dropArea.removeClass('highlight');
        });
    });

    dropArea.on('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        const dt = e.originalEvent.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    selectFileLink.click(function(e) {
        e.preventDefault();
        imageUploadInput.click();
    });


    imageUploadInput.change(function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            errorMessage.hide();
            resultSection.hide();
            downloadButton.hide();
            downloadAllImagesButton.hide();
            detectedTextInfo.hide();
            translateButton.prop('disabled', true);
            translateAllButton.prop('disabled', true);
            prevImageButton.prop('disabled', true);
            nextImageButton.prop('disabled', true);
            deleteCurrentImageButton.prop('disabled', true);

            currentImageIndex = -1;
            let filesArray = Array.from(files);
            const filePromises = [];

            for (let i = 0; i < filesArray.length; i++) {
                const file = filesArray[i];
                if (file.type.startsWith('image/')) {
                    const filePromise = new Promise((resolve, reject) => {
                        const reader = new FileReader();
                        reader.onload = function(e) {
                            const originalDataURL = e.target.result;
                            createThumbnail(originalDataURL, file.name, images.length + i)
                                .then(thumbnailDataURL => {
                                    images.push({
                                        originalDataURL: originalDataURL,
                                        translatedDataURL: null,
                                        bubbleTexts: [],
                                        bubbleCoords: [],
                                        thumbnailDataURL: thumbnailDataURL,
                                        fileName: file.name,
                                        fontSize: defaultFontSize,
                                        fontFamily: defaultFontFamily,
                                        layoutDirection: layoutDirectionSelect.val()
                                    });
                                    resolve();
                                })
                                .catch(error => { 
                                    console.error("创建缩略图失败:", file.name, error);
                                    errorMessage.text("创建缩略图失败，请检查图片文件: " + file.name).show();
                                    resolve(); 
                                });
                        };
                        reader.onerror = reject;
                        reader.readAsDataURL(file);
                    });
                    filePromises.push(filePromise);

                } else if (file.type === 'application/pdf') {
                    const filePromise = new Promise((resolve, reject) => {
                        const formData = new FormData();
                        formData.append('pdfFile', file);
                        $.ajax({
                            url: '/upload_pdf',
                            type: 'POST',
                            data: formData,
                            contentType: false,
                            processData: false,
                            dataType: 'json',
                            success: function(response) {
                                if (response.images && response.images.length > 0) {
                                    const pdfImagePromises = [];
                                    response.images.forEach((imageData, index) => {
                                        const pdfImagePromise = new Promise((resolveThumbnail, rejectThumbnail) => {
                                            const originalDataURL = "data:image/png;base64," + imageData;
                                            createThumbnail(originalDataURL, "PDF图片" + (images.length + index), images.length + index)
                                                .then(thumbnailDataURL => {
                                                    images.push({
                                                        originalDataURL: originalDataURL,
                                                        translatedDataURL: null,
                                                        bubbleTexts: [],
                                                        bubbleCoords: [],
                                                        thumbnailDataURL: thumbnailDataURL,
                                                        fileName: "PDF图片" + (images.length + index),
                                                        fontSize: defaultFontSize,
                                                        fontFamily: defaultFontFamily,
                                                        layoutDirection: layoutDirectionSelect.val()
                                                    });
                                                    resolveThumbnail();
                                                })
                                                .catch(error => { 
                                                    console.error("PDF图片创建缩略图失败:", "PDF图片" + (images.length + index), error);
                                                    resolveThumbnail(); 
                                                });
                                        });
                                        pdfImagePromises.push(pdfImagePromises);
                                    });
                                    Promise.all(pdfImagePromises).then(resolve).catch(reject);
                                } else {
                                    errorMessage.text("PDF中没有图片").show();
                                    resolve();
                                }
                            },
                            error: function(error) {
                                console.error("上传 PDF 文件失败:", error);
                                errorMessage.text("上传 PDF 文件失败，请检查文件格式或重试").show();
                                reject(error);
                            }
                        });
                    });
                    filePromises.push(filePromise);
                } else {
                    errorMessage.text("请选择图片或PDF文件").show();
                    filePromises.push(Promise.resolve());
                }
            }

            Promise.all(filePromises)
                .then(() => {
                    finishHandleFiles();
                })
                .catch(error => {
                    console.error("文件处理过程中发生错误:", error);
                    errorMessage.text("文件处理过程中发生错误，请查看控制台。").show(); 
                });

        }
    }

    function finishHandleFiles() {
        loadingMessage.hide();
        loadingAnimation.hide();
        translateButton.prop('disabled', false);
        translateAllButton.prop('disabled', false);
        deleteCurrentImageButton.prop('disabled', false);
        sortImagesByName();

        setTimeout(function() {
            renderThumbnails();
            if (images.length > 0 ) {
                if (currentImageIndex === -1) {
                    switchImage(0);
                } else {
                    switchImage(currentImageIndex);
                }
            }
        }, 50); 

    }

    function sortImagesByName() {
        images.sort((a, b) => {
            if (a.fileName < b.fileName) {
                return -1;
            }
            if (a.fileName > b.fileName) {
                return 1;
            }
            return 0;
        });
    }


    function createThumbnail(imageSrc, fileName, index) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = function() {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                const maxWidth = 100;
                const maxHeight = 100;
                let width = img.width;
                let height = img.height;

                if (width > maxWidth) {
                    height *= maxWidth / width;
                    width = maxWidth;
                }
                if (height > maxHeight) {
                    width *= maxHeight / height;
                    width = maxWidth;
                }
                canvas.width = width;
                canvas.height = height;
                ctx.drawImage(img, 0, 0, width, height);
                const thumbnailDataURL = canvas.toDataURL('image/jpeg');
                resolve(thumbnailDataURL);
            };
            img.onerror = reject;
            img.src = imageSrc;
        });
    }


    function renderThumbnails() {
        thumbnailList.empty();
        images.forEach((imageData, index) => {
            const listItem = $("<li></li>");
            const thumbnailImg = $("<img>").attr('src', imageData.thumbnailDataURL).attr('alt', imageData.fileName);
            listItem.append(thumbnailImg);
            listItem.click(function() {
                switchImage(index);
            });
            thumbnailList.append(listItem);
        });
        updateNavigationButtons();
    }


    function switchImage(index) {
        if (index >= 0 && index < images.length) {
            currentImageIndex = index;
            console.log("switchImage called, index:", index, "currentImageIndex updated to:", currentImageIndex);
            translatedImageDisplay.attr('src', images[index].translatedDataURL || images[index].originalDataURL).show();
            resultSection.toggle(images[index].translatedDataURL !== null);
            downloadButton.toggle(images[index].translatedDataURL !== null);
            downloadAllImagesButton.toggle(images.some(img => img.translatedDataURL !== null));
            detectedTextInfo.toggle(images[index].bubbleTexts.length > 0 && images[index].translatedDataURL !== null);
            detectedTextList.text(images[index].bubbleTexts.join("\n"));

            $("#thumbnailList li").removeClass('active');
            $("#thumbnailList li").eq(index).addClass('active');

            fontSizeInput.val(images[index].fontSize);
            fontFamilySelect.val(images[index].fontFamily);
            deleteCurrentImageButton.prop('disabled', false);
            layoutDirectionSelect.val(images[index].layoutDirection);


            updateNavigationButtons();
        } else {
            deleteCurrentImageButton.prop('disabled', true);
        }
    }

    function updateNavigationButtons() {
        console.log("updateNavigationButtons called, currentImageIndex:", currentImageIndex, "images.length:", images.length);
        if (images.length <= 1) {
            prevImageButton.prop('disabled', true);
            nextImageButton.prop('disabled', true);
        } else {
            prevImageButton.prop('disabled', currentImageIndex <= 0);
            nextImageButton.prop('disabled', currentImageIndex >= images.length - 1);
        }
    }

    function reRenderTranslatedImage(fontSize, fontFamily, imageIndex) {
        if (imageIndex === -1 || !images[imageIndex].translatedDataURL || !images[imageIndex].bubbleTexts || !images[imageIndex].bubbleCoords) {
            return;
        }

        const imageData = images[imageIndex].originalDataURL.split(',')[1];
        const translatedText = images[imageIndex].bubbleTexts;
        const bubbleCoords = images[imageIndex].bubbleCoords;


        $.ajax({
            type: 'POST',
            url: '/re_render_image',
            data: JSON.stringify({
                image: imageData,
                translated_text: translatedText,
                bubble_coords: bubbleCoords,
                fontSize: fontSize,
                fontFamily: fontFamily,
                textDirection: layoutDirectionSelect.val()
            }),
            contentType: 'application/json',
            success: function(response) {
                images[imageIndex].translatedDataURL = 'data:image/png;base64,' + response.rendered_image;
                images[imageIndex].fontSize = fontSize;
                images[imageIndex].fontFamily = fontFamily;
                images[imageIndex].layoutDirection = layoutDirectionSelect.val();
                if (currentImageIndex === imageIndex) {
                    switchImage(currentImageIndex);
                }
            },
            error: function(error) {
                console.error('Error re-rendering image:', error);
                errorMessage.text("重新渲染图片出错，请查看控制台。").show();
            }
        });
    }


    fontSizeInput.on('change', function() {
        const newFontSize = fontSizeInput.val();
        if (currentImageIndex !== -1 && images[currentImageIndex].translatedDataURL) {
            images[currentImageIndex].fontSize = newFontSize;
            reRenderTranslatedImage(newFontSize, fontFamilySelect.val(), currentImageIndex);
        }
    });

    fontFamilySelect.on('change', function() {
        const newFontFamily = fontFamilySelect.val();
        if (currentImageIndex !== -1 && images[currentImageIndex].translatedDataURL) {
            images[currentImageIndex].fontFamily = newFontFamily;
            reRenderTranslatedImage(fontSizeInput.val(), newFontFamily, currentImageIndex);
        }
    });

    layoutDirectionSelect.on('change', function() {
        if (currentImageIndex !== -1 && images[currentImageIndex].translatedDataURL) {
            images[currentImageIndex].layoutDirection = layoutDirectionSelect.val();
            reRenderTranslatedImage(fontSizeInput.val(), fontFamilySelect.val(), currentImageIndex);
        }
    });


    prevImageButton.click(function() {
        console.log("prevImageButton clicked, currentImageIndex before:", currentImageIndex);
        if (currentImageIndex > 0) {
            switchImage(currentImageIndex - 1);
        }
    });

    nextImageButton.click(function() {
        console.log("nextImageButton clicked, currentImageIndex before:", currentImageIndex);
        if (currentImageIndex < images.length - 1) {
            switchImage(currentImageIndex + 1);
        }
    });

    translateButton.click(function() {
        if (currentImageIndex === -1) {
            alert("请先选择要翻译的图片");
            return;
        }
        translateCurrentImage();
    });

    function translateCurrentImage() {
        translatingMessage.show();

        loadingMessage.hide();
        loadingAnimation.hide();
        errorMessage.hide();
        resultSection.show();
        downloadButton.hide();
        downloadAllImagesButton.hide();
        detectedTextInfo.hide();

        const targetLanguage = $("#targetLanguage").val();
        const apiKey = $("#apiKey").val();
        const modelName = $("#modelName").val();
        const fontSize = fontSizeInput.val();
        const fontFamily = fontFamilySelect.val();
        const modelProvider = $("#modelProvider").val();
        const customProviderBaseUrl = modelProvider === 'custom'? $("#customProviderBaseUrl").val() : undefined;
        const imageData = images[currentImageIndex].originalDataURL.split(',')[1];
        const promptContent = currentPromptContent;

        $.ajax({
            type: 'POST',
            url: '/translate_image',
            data: JSON.stringify({
                image: imageData,
                target_language: targetLanguage,
                textDirection: layoutDirectionSelect.val(),
                fontSize: fontSize,
                model_provider: modelProvider,
                custom_base_url: customProviderBaseUrl,
                api_key: apiKey,
                model_name: modelName,
                fontFamily: fontFamily,
                prompt_content: promptContent
            }),
            contentType: 'application/json',
            success: function(response) {
                loadingMessage.hide();
                loadingAnimation.hide();
                images[currentImageIndex].translatedDataURL = 'data:image/png;base64,' + response.translated_image;
                images[currentImageIndex].bubbleTexts = response.bubble_texts || [];
                images[currentImageIndex].bubbleCoords = response.bubble_coords || [];
                images[currentImageIndex].fontSize = fontSize;
                images[currentImageIndex].fontFamily = fontFamily;
                images[currentImageIndex].layoutDirection = layoutDirectionSelect.val();


                switchImage(currentImageIndex);

                const data = {
                    modelProvider: modelProvider,
                    modelName: modelName
                };
                $.ajax({
                    url: '/save_model_info',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify(data),
                    success: function() {
                        console.log("模型信息保存成功");
                    },
                    error: function(error) {
                        console.error("模型信息保存失败:", error);
                    }
                });

                translatingMessage.hide();

            },
            error: function(error) {
                loadingMessage.hide();
                loadingAnimation.hide();
                resultSection.hide();
                downloadButton.hide();
                downloadAllImagesButton.hide();
                detectedTextInfo.hide();

                console.error('Error:', error);
                if (error.responseJSON && error.responseJSON.error) {
                    errorMessage.text("错误: " + error.responseJSON.error).show();
                } else {
                    errorMessage.text("处理出错，请查看控制台。").show();
                }
                translatingMessage.hide();
            }
        });
    }


    translateAllButton.click(function() {
        if (images.length === 0) {
            alert("请先上传漫画图片");
            return;
        }
        translateAllImages();
    });

    function translateAllImages() {
        translatingMessage.show();

        loadingMessage.hide();
        loadingAnimation.hide();
        errorMessage.hide();
        resultSection.show();
        downloadButton.hide();
        downloadAllImagesButton.hide();
        detectedTextInfo.hide();
        translateButton.prop('disabled', true);
        translateAllButton.prop('disabled', true);
        deleteCurrentImageButton.prop('disabled', true);

        const targetLanguage = $("#targetLanguage").val();
        const apiKey = $("#apiKey").val();
        const modelName = $("#modelName").val();
        const fontSize = fontSizeInput.val();
        const fontFamily = fontFamilySelect.val();
        const modelProvider = $("#modelProvider").val();
        const customProviderBaseUrl = modelProvider === 'custom'? $("#customProviderBaseUrl").val() : undefined;
        const promptContent = currentPromptContent;
        const layoutDirection = layoutDirectionSelect.val();

        Promise.all(images.map((imageData, index) => {
            return new Promise((resolve, reject) => {
                const postData = {
                    image: imageData.originalDataURL.split(',')[1],
                    target_language: targetLanguage,
                    textDirection: layoutDirection,
                    fontSize: fontSize,
                    model_provider: modelProvider,
                    custom_base_url: customProviderBaseUrl,
                    api_key: apiKey,
                    model_name: modelName,
                    fontFamily: fontFamily,
                    prompt_content: promptContent
                };

                $.ajax({
                    type: 'POST',
                    url: '/translate_image',
                    data: JSON.stringify(postData),
                    contentType: 'application/json',
                    success: function(response) {
                        images[index].translatedDataURL = 'data:image/png;base64,' + response.translated_image;
                        images[index].bubbleTexts = response.bubble_texts || [];
                        images[index].bubbleCoords = response.bubble_coords || [];
                        images[index].fontSize = fontSize;
                        images[index].fontFamily = fontFamily;
                        images[index].layoutDirection = layoutDirection;
                        resolve();
                    },
                    error: function(error) {
                        console.error(`图片 ${imageData.fileName} 翻译失败:`, error);
                        resolve();
                    }
                });
            });
        }))
        .then(() => {
            loadingMessage.hide();
            loadingAnimation.hide();
            translateButton.prop('disabled', false);
            translateAllButton.prop('disabled', false);
            deleteCurrentImageButton.prop('disabled', false);

            if (currentImageIndex !== -1) {
                switchImage(currentImageIndex);
            } else if (images.length > 0) {
                switchImage(0);
            }

            alert("所有图片翻译完成 (部分图片可能翻译失败，请查看控制台)");

            translatingMessage.hide();
        });
    }

    downloadAllImagesButton.click(function() {
        downloadingMessage.show();

        const zip = new JSZip();
        images.forEach(imageData => {
            if (imageData.translatedDataURL) {
                const dataURL = imageData.translatedDataURL;
                const base64Image = dataURL.split(',')[1];
                zip.file(imageData.fileName.replace(/\..+$/, '') + '_translated.png', base64Image, {base64: true});
            }
        });

        zip.generateAsync({type:"blob"})
            .then(function(content) {
                const a = document.createElement('a');
                a.href = URL.createObjectURL(content);
                a.download = "translated_images.zip";
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                downloadingMessage.hide();
            });
    });


    downloadButton.click(function() {
        downloadingMessage.show();

        if (currentImageIndex !== -1 && images[currentImageIndex].translatedDataURL) {
            const dataURL = images[currentImageIndex].translatedDataURL;
            const a = document.createElement('a');
            a.href = dataURL;
            a.download = images[currentImageIndex].fileName.replace(/\..+$/, '') + '_translated.png';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
        downloadingMessage.hide();
    });

    clearAllImagesButton.click(function() {
        images = [];
        currentImageIndex = -1;
        thumbnailList.empty();
        translatedImageDisplay.attr('src', '#').hide();
        resultSection.hide();
        downloadButton.hide();
        downloadAllImagesButton.hide();
        detectedTextInfo.hide();
        translateButton.prop('disabled', true);
        translateAllButton.prop('disabled', true);
        prevImageButton.prop('disabled', true);
        nextImageButton.prop('disabled', true);
        deleteCurrentImageButton.prop('disabled', true);
    });

    deleteCurrentImageButton.click(function() {
        if (currentImageIndex !== -1) {
            images.splice(currentImageIndex, 1);
            if (images.length === 0) {
                currentImageIndex = -1;
                thumbnailList.empty();
                translatedImageDisplay.attr('src', '#').hide();
                resultSection.hide();
                downloadButton.hide();
                downloadAllImagesButton.hide();
                detectedTextInfo.hide();
                translateButton.prop('disabled', true);
                translateAllButton.prop('disabled', true);
                prevImageButton.prop('disabled', true);
                nextImageButton.prop('disabled', true);
                deleteCurrentImageButton.prop('disabled', true);
            } else {
                if (currentImageIndex >= images.length) {
                    currentImageIndex = images.length - 1;
                }
                renderThumbnails();
                switchImage(currentImageIndex);
            }
        }
    });


    modelProviderSelect.change(function() {
        updateModelSuggestions();
        const customSettings = document.getElementById('customProviderSettings');
        if ($(this).val() === "custom") {
            customSettings.style.display = "block";
        } else {
            customSettings.style.display = "none";
        }
    });

    $(document).ready(function() {
        const customSettings = document.getElementById('customProviderSettings');
        if (modelProviderSelect.val() === "custom") {
            customSettings.style.display = "block";
        } else {
            customSettings.style.display = "none";
        }
    });
    function updateModelSuggestions() {
        const provider = modelProviderSelect.val();
        modelSuggestionsDiv.empty();
        $.ajax({
            url: '/get_used_models',
            type: 'GET',
            data: { model_provider: provider },
            dataType: 'json',
            success: function(response) {
                if (response.models && response.models.length > 0) {
                    const ul = $("<ul></ul>");
                    response.models.forEach(modelName => {
                        const li = $("<li></li>").text(modelName).click(function() {
                            $("#modelName").val(modelName);
                            modelSuggestionsDiv.empty();
                        });
                        ul.append(li);
                    });
                    modelSuggestionsDiv.append(ul);
                }
            },
            error: function(error) {
                console.error("获取模型建议失败:", error);
            }
        });
    }

    $('#modelName').on('focus', updateModelSuggestions);
    $('#modelName').on('blur', function() {
        setTimeout(function() {
            modelSuggestionsDiv.empty();
        }, 200);
    });
    $('#modelName').on('input', function() {
        if ($(this).val().trim() === '') {
            updateModelSuggestions();
        } else {
            modelSuggestionsDiv.empty();
        }
    });


    applyFontSettingsToAllButton.click(function() {
        if (currentImageIndex === -1) {
            alert("请先选择一张图片。");
            return;
        }

        const currentFontSize = fontSizeInput.val();
        const currentFontFamily = fontFamilySelect.val();
        const currentLayoutDirection = layoutDirectionSelect.val();

        images.forEach((imageData, index) => {
            imageData.fontSize = currentFontSize;
            imageData.fontFamily = currentFontFamily;
            imageData.layoutDirection = currentLayoutDirection;
            if (imageData.translatedDataURL) {
                reRenderTranslatedImage(currentFontSize, currentFontFamily, index);
            }
        });
        alert("已将当前字号、字体和排版设置应用到所有图片。");
    });

    function reRenderTranslatedImage(fontSize, fontFamily, imageIndex) {
        if (imageIndex === -1 || !images[imageIndex].translatedDataURL || !images[imageIndex].bubbleTexts || !images[imageIndex].bubbleCoords) {
            return;
        }

        const imageData = images[imageIndex].originalDataURL.split(',')[1];
        const translatedText = images[imageIndex].bubbleTexts;
        const bubbleCoords = images[imageIndex].bubbleCoords;


        $.ajax({
            type: 'POST',
            url: '/re_render_image',
            data: JSON.stringify({
                image: imageData,
                translated_text: translatedText,
                bubble_coords: bubbleCoords,
                fontSize: fontSize,
                fontFamily: fontFamily,
                textDirection: layoutDirectionSelect.val()
            }),
            contentType: 'application/json',
            success: function(response) {
                images[imageIndex].translatedDataURL = 'data:image/png;base64,' + response.rendered_image;
                images[imageIndex].fontSize = fontSize;
                images[imageIndex].fontFamily = fontFamily;
                images[imageIndex].layoutDirection = layoutDirectionSelect.val();
                if (currentImageIndex === imageIndex) {
                    switchImage(currentImageIndex);
                }
            },
            error: function(error) {
                console.error('Error re-rendering image:', error);
                errorMessage.text("重新渲染图片出错，请查看控制台。").show();
            }
        });
    }

    $(document).keydown(function(event) {
        if (event.altKey) {
            switch (event.key) {
                case 'ArrowUp':
                    event.preventDefault();
                    let currentFontSizeUp = parseInt(fontSizeInput.val());
                    fontSizeInput.val(currentFontSizeUp + 1).trigger('change');
                    break;
                case 'ArrowDown':
                    event.preventDefault();
                    let currentFontSizeDown = parseInt(fontSizeInput.val());
                    fontSizeInput.val(Math.max(10, currentFontSizeDown - 1)).trigger('change');
                    break;
                case 'ArrowLeft':
                    event.preventDefault();
                    prevImageButton.click();
                    break;
                case 'ArrowRight':
                    event.preventDefault();
                    nextImageButton.click();
                    break;
            }
        }
    });

});
