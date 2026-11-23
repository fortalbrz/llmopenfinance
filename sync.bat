@echo off
chcp 65001 >nul
set PYTHONPATH=src;%PYTHONPATH%
set PYTHONIOENCODING=utf-8
python --version
python -m llmopenfinance sync