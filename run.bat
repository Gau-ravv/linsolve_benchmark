@echo off
rem Build if needed, then run. Most of the wall time is the two slow rows at
rem 2048: blocked-lu/naive and inversion. Shorten the run by editing
rem SOLVE_SIZES at the top of linsolve.cpp.
cd /d "%~dp0"

if not exist linsolve.exe (
    call "%~dp0build.bat"
    if errorlevel 1 exit /b 1
)

"%~dp0linsolve.exe"
