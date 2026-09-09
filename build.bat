@echo off
rem Build the linear-solver benchmark with the MSYS2 mingw64 g++ installed on
rem this PC.
rem   -O2           optimize (never benchmark at -O0: it kills inlining and
rem                 auto-vectorization, so every kernel would look equally slow)
rem   -std=c++17    language level the code targets
rem   -march=native target this exact CPU (i7-9750H) so the AVX2/FMA intrinsics compile
rem   -fopenmp      enable the #pragma omp parallel for in the threaded kernels
rem   -static       bake libgomp/libstdc++/libwinpthread into the .exe, so it runs
rem                 from any folder without C:\msys64\mingw64\bin on the PATH
cd /d "%~dp0"
set "PATH=C:\msys64\mingw64\bin;%PATH%"

g++ -O2 -std=c++17 -Wall -Wextra -march=native -fopenmp -static -o linsolve.exe linsolve.cpp
if errorlevel 1 (
    echo.
    echo BUILD FAILED
    exit /b 1
)

echo Build OK -^> linsolve.exe
