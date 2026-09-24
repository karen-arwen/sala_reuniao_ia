@echo off
chcp 65001 >nul
title Sala de Reuniao Inteligente
cd /d "%~dp0"
echo.
echo   ==  Sala de Reuniao Inteligente  ==
echo.
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Criando ambiente virtual .venv ...
    py -3 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERRO: Python nao encontrado. Instale o Python 3.10+ em https://www.python.org/downloads/
    echo e marque a opcao "Add python.exe to PATH" na instalacao.
    pause
    exit /b 1
)
echo [2/3] Instalando/verificando bibliotecas (a primeira vez demora alguns minutos)...
".venv\Scripts\python.exe" -m pip install --upgrade pip -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt -q
if errorlevel 1 (
    echo ERRO ao instalar as bibliotecas. Veja a mensagem acima.
    pause
    exit /b 1
)
echo [3/3] Abrindo a interface no navegador... (para fechar: feche esta janela)
".venv\Scripts\python.exe" -m streamlit run src\app.py
pause
