<p align="center">
    <img src="https://github.com/MashiroSaber03/Saber-Translator/raw/main/pic/logo.png" alt="Saber-Translator Logo" width="200">
</p>

<h1 align="center">Saber-Translator</h1>

<p align="center">
    ✨ 你的专属 AI 漫画翻译神器，轻松扫除语言障碍，畅享原版漫画乐趣！✨
</p>

<p align="center">
    <a href="https://github.com/MashiroSaber03/Saber-Translator/stargazers">
        <img src="https://img.shields.io/github/stars/MashiroSaber03/Saber-Translator?style=social" alt="GitHub stars">
    </a>
    <a href="LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
    </a>
</p>

<div align="center">
    <a href="https://github.com/MashiroSaber03/Saber-Translator/issues">问题反馈 / 建议</a>
</div>

---

## ✨ 主要功能

1.  **漫画图片/PDF 翻译**:
    *   支持 **图片** (JPG, PNG 等) 和 **PDF** 漫画文件上传。
    *   一键批量上传，轻松处理多页漫画。

2.  **智能气泡检测与文本识别**:
    *   **YOLOv5** 模型精准定位漫画对话气泡。
    *   基于 **[Manga OCR](https://github.com/kha-white/manga-ocr)** 技术，高效识别气泡中的日文文本。

3.  **强大的 AI 翻译引擎**:
    *   内置 **SiliconFlow** 和 **DeepSeek** 翻译服务商支持 (更多服务商敬请期待)。
    *   支持 **API Key** 验证，保障翻译质量和稳定性。
    *   可自由选择和切换翻译模型。

4.  **高度可定制的翻译效果**:
    *   **自定义提示词 (Prompt)**: 根据漫画风格调整 Prompt，优化翻译结果，更贴合语境。
    *   **字体与字号选择**: 多种中文字体和字号可选，打造个性化阅读体验。
    *   **灵活的排版设置**: 支持 **横向** 和 **竖向** 排版，完美适配不同漫画的阅读习惯。

5.  **实时预览与便捷操作**:
    *   **Web 界面实时预览**: 上传图片即可立即查看翻译效果。
    *   **参数调整即时生效**: 字体、字号、排版等设置更改后实时更新预览。
    *   **快捷键支持**: Alt + 上/下方向键 快速调整字号 (更多快捷键敬请期待)。

6.  **高效的图片管理与下载**:
    *   **缩略图概览**: 侧边栏展示图片缩略图，方便快速切换和管理。
    *   **单张/批量下载**: 支持下载单张翻译后的图片，也支持一键打包下载所有翻译结果。

7.  **模型记录与推荐**:
    *   自动记录用户常用的 AI 模型，方便下次快速选择。
    *   提供模型选择建议 (未来功能)。

## ✨ Demo (效果示例)

<div align='center'>
    <img src="https://github.com/MashiroSaber03/Saber-Translator/raw/main/pic/interface.png" width="80%" alt="Saber-Translator 翻译界面">
    <p>_✨ Saber-Translator 翻译界面概览 ✨_</p>
</div>

<br/>

**翻译效果对比：**

<div align='center'>
    <table style="width: 80%; border-collapse: collapse;">
        <tr>
            <th style="width: 50%; border: 1px solid #ddd; padding: 8px; text-align: center;">翻译前</th>
            <th style="width: 50%; border: 1px solid #ddd; padding: 8px; text-align: center;">翻译后</th>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">
                <img src="https://github.com/MashiroSaber03/Saber-Translator/raw/main/pic/before1.png" width="90%" alt="翻译前图片 1">
            </td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">
                <img src="https://github.com/MashiroSaber03/Saber-Translator/raw/main/pic/after1.png" width="90%" alt="翻译后图片 1">
            </td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">
                <img src="https://github.com/MashiroSaber03/Saber-Translator/raw/main/pic/before2.png" width="90%" alt="翻译前图片 2">
            </td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">
                <img src="https://github.com/MashiroSaber03/Saber-Translator/raw/main/pic/after2.png" width="90%" alt="翻译后图片 2">
            </td>
        </tr>
    </table>
</div>

<br/>

## ✨ 使用方式

### 快速上手

1.  **下载最新版本**:  前往 [Releases](https://github.com/MashiroSaber03/Saber-Translator/releases) 页面，下载最新版本的压缩包。
2.  **解压**:  将下载的压缩包解压到任意目录。
3.  **运行程序**:  在解压后的目录中，找到 `Saber-Translator.exe` ，双击运行即可。
4.  **开始翻译**:  程序启动后，会自动在浏览器中打开 Saber-Translator 的 Web 界面 (默认地址为 `http://127.0.0.1:5000/`)。
5.  **后续步骤**:  参考 "快速上手" 部分的说明，上传漫画、配置翻译设置、开始翻译，并查看和下载翻译结果。

## 🚀 路线图 (未来计划)

*   **更多 AI 翻译服务商支持**: 例如 OpenAI, Google Translate, Azure Translate 等，提供更多模型选择。
*   **多语言支持**: 支持翻译成更多目标语言，例如 英文、日文、繁体中文等。
*   **更智能的排版优化**: 根据气泡形状和文本内容自动调整字体大小和排版，提升阅读体验。
*   **上下文理解**: 优化翻译引擎，提升对漫画语境的理解，使翻译更准确自然。
*   **插件系统**: 开放插件接口，允许用户扩展更多功能 (例如 词典查询、术语库、风格迁移等)。
*   **移动端 App**: 开发移动端 App，方便用户在手机或平板上使用。

## ⚡ 支持情况

*   **支持文件格式**:
    *   图片: JPG, PNG, JPEG 等常见图片格式
    *   PDF 漫画文件
*   **支持翻译服务商**:
    *   SiliconFlow (硅基流动)
    *   DeepSeek (深度求索)
    *   (未来计划支持更多)
*   **支持目标语言**:
    *   简体中文 (zh)
    *   (未来计划支持更多语言)
*   **支持排版方向**:
    *   竖向排版 (vertical)
    *   横向排版 (horizontal)
*   **支持字体 (static 文件夹下)**:
    *   华文行楷 (STXINGKA.TTF)
    *   华文新魏 (STXINWEI.TTF)
    *   华文中宋 (STZHONGS.TTF)
    *   楷体 (STKAITI.TTF)
    *   隶书 (STLITI.TTF)
    *   宋体 (STSONG.TTF)
    *   微软雅黑 (MSYH.TTC)
    *   微软雅黑粗体 (MSYHBD.TTC)
    *   幼圆 (SIMYOU.TTF)
    *   仿宋 (STFANGSO.TTF)

## 💖 赞助 (Sponsorship)

感谢你对 Saber-Translator 项目的关注和支持！ 如果你觉得本项目对你有所帮助，并且愿意支持我继续开发和维护，可以通过以下方式进行赞助。 你的支持是我前进的最大动力！

**赞助方式：**

你可以通过微信或支付宝扫描下方二维码进行赞赏。 你的每一份支持都将用于项目的服务器维护、功能开发和持续优化。

<div align="center">
    <table>
        <tr>
            <td style="text-align: center; padding: 0 20px;">
                <img src="pic/wechat_qrcode.png" width="200" alt="微信赞赏码">
                <p>微信赞赏码</p>
            </td>
            <td style="text-align: center; padding: 0 20px;">
                <img src="pic/alipay_qrcode.png" width="200" alt="支付宝赞赏码">
                <p>支付宝赞赏码</p>
            </td>
        </tr>
    </table>
</div>

<p align="center">
    再次感谢你的慷慨支持！
</p>

## ❤️ 贡献

欢迎任何形式的贡献，包括但不限于：

*   **代码贡献 (Pull Requests)**: 修复 Bug, 增加新功能, 优化代码性能等。
*   **问题反馈 (Issues)**: 报告 Bug, 提出功能建议, 分享使用遇到的问题等。
*   **文档完善**: 改进 README 文档, 编写更详细的使用教程等。
*   **翻译贡献**: 将 README 文档翻译成其他语言。
*   **推广宣传**: 向更多人介绍和推荐 Saber-Translator。

如果你希望为 Saber-Translator 做出贡献，请随时提交 Issues 或 Pull Requests！

## 🌟 支持

*   **Star 项目**: 如果你觉得 Saber-Translator 对你有帮助，请给项目点个 Star，这将是对我最大的鼓励和支持！
*   **分享推荐**: 将 Saber-Translator 分享给你的朋友或有需要的人。

## Disclaimer (免责声明)

1.  Saber-Translator 仅供学习交流使用，请勿用于商业用途。
2.  使用本项目请遵守当地法律法规，禁止用于非法用途。
3.  本项目不对翻译结果的准确性、完整性、及时性等做任何保证，因使用本项目产生的任何风险和责任由用户自行承担。
4.  本项目使用第三方 AI 翻译服务，请用户自行了解并遵守相关服务商的使用协议和条款。
5.  对于因不当使用本项目造成的任何损失，本项目作者概不负责。

---

<p align="center">
    感谢使用 Saber-Translator！希望它能帮助你更好地享受漫画阅读的乐趣！
</p>
