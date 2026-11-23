@echo off
cls
del /q %~dp0\build\*.*
del /q %~dp0\dist\*.*

cd %~dp0\
pip uninstall llmopenfinance -y 2>nul
pip uninstall llmopenfinance -y 2>nul

uv build

for %%A in ("%~dp0\dist\*.tar.gz") do pip install %%~fA[web]

rmdir /s /q %~dp0\src\llmopenfinance.egg-info 2>nul

REM Generate beautiful HTML from readme.md using pandoc
pandoc %~dp0\README.md -f markdown -t html5 --standalone --css=https://cdn.jsdelivr.net/npm/water.css@2/out/dark.css -o %~dp0\docs\index.html

@echo off
set PYTHONPATH=src;%PYTHONPATH%
python -m llmopenfinance --version