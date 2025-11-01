@echo off
chcp 65001 >nul 2>&1
title 脚本名称

echo 正在启动脚本...
python "%~dp0script_name.py" %*

if errorlevel 1 (
    echo.
    echo 执行出错，请检查错误信息
    pause
    exit /b 1
)

echo.
echo 执行完成
pause
