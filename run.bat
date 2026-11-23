@echo off
chcp 65001 >nul
set PYTHONPATH=src;%PYTHONPATH%
set PYTHONIOENCODING=utf-8
python --version
python -m llmopenfinance %1 %2 %3 %4 %5 %6 %7 %8 %9
