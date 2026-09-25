@echo off
chcp 65001 >nul
title Sala de Reuniao Inteligente
cd /d "%~dp0"
echo.
echo   Sala de Reuniao Inteligente
echo.
if exist ".venv\Scripts\python.exe" goto instalar

echo [1/4] Procurando o Python - prefere 3.12 ou 3.11, as mais compativeis...
set PYEXE=
py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 set PYEXE=py -3.12
if defined PYEXE goto criar
py -3.11 -c "import sys" >nul 2>nul
if not errorlevel 1 set PYEXE=py -3.11
if defined PYEXE goto criar
if exist "C:\Python312\python.exe" set PYEXE="C:\Python312\python.exe"
if defined PYEXE goto criar
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PYEXE="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PYEXE goto criar
py -3 -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
if not errorlevel 1 set PYEXE=py -3
if defined PYEXE goto criar
python -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
if not errorlevel 1 set PYEXE=python
if defined PYEXE goto criar
if exist "%USERPROFILE%\anaconda3\python.exe" set PYEXE="%USERPROFILE%\anaconda3\python.exe"
if defined PYEXE goto criar
if exist "%USERPROFILE%\miniconda3\python.exe" set PYEXE="%USERPROFILE%\miniconda3\python.exe"
if defined PYEXE goto criar
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do if exist "%%D\python.exe" set PYEXE="%%D\python.exe"
if defined PYEXE goto criar
echo.
echo ERRO: nao encontrei o Python 3.10 ou mais novo.
echo Instale em https://www.python.org/downloads/ e marque "Add python.exe to PATH".
pause
exit /b 1

:criar
echo       Usando: %PYEXE%
echo [2/4] Criando o ambiente .venv ...
%PYEXE% -m venv .venv
if not exist ".venv\Scripts\python.exe" goto erro_venv

:instalar
echo [3/4] Instalando/verificando bibliotecas - a primeira vez demora alguns minutos...
".venv\Scripts\python.exe" -m pip install --upgrade pip -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt -q
if errorlevel 1 goto erro_pip
echo.
".venv\Scripts\python.exe" src\verificar_ambiente.py
echo.
echo [4/4] Abrindo a interface no navegador... para fechar, feche esta janela.
".venv\Scripts\python.exe" -m streamlit run src\app.py
pause
exit /b 0

:erro_venv
echo ERRO ao criar o ambiente .venv. Tente apagar a pasta .venv e rodar de novo.
pause
exit /b 1

:erro_pip
echo ERRO ao instalar as bibliotecas. Confira a internet e veja a mensagem acima.
pause
exit /b 1
