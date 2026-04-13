@echo off
setlocal

set "APP_PATH=%~dp0app.py"

:: 0. 关闭当前项目遗留的旧后端进程，避免页面请求打到旧实例，导致当前窗口看不到后续日志
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$appPath = [System.IO.Path]::GetFullPath('%APP_PATH%');" ^
  "$targets = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like ('*' + $appPath + '*') };" ^
  "foreach ($p in $targets) { try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; Write-Host ('已停止旧后端进程 PID=' + $p.ProcessId) } catch {} }"

timeout /t 1 >nul

:: 1. ����ר�����䣨���⻷����
call venv\Scripts\activate

:: 2. ��������Ҫ�ġ��߼����ء�
set SABER_PROFILE_DETECT=1
set SABER_DEFAULT_FP16=1

:: 3. ������Щ��������������
venv\Scripts\python.exe app.py

:: 4. ��ͣһ�£���ֹ����ʱ����һ������
pause
