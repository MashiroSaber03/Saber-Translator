@echo off
:: 1. 激活专属房间（虚拟环境）
call venv\Scripts\activate

:: 2. 拨下你需要的“高级开关”
set SABER_PROFILE_DETECT=1
set SABER_DEFAULT_FP16=1

:: 3. 带着这些开关启动主程序
python app.py

:: 4. 暂停一下，防止报错时窗口一闪而过
pause