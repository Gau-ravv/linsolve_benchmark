@echo off
rem Build the linear-solver benchmark with the MSYS2 mingw64 g++ already
rem installed on this PC. Same flags as the matmul benchmark, deliberately --
rem the six kernels are copied verbatim, so they must be compiled the same way
rem for the timings to mean anything next to that project's numbers.
rem   -O2           optimize (never benchmark at -O0: it kills inlining and
rem                 auto-vectorization, so every version would look equally slow)
rem   -march=native target this exact CPU (i7-9750H) so the AVX2/FMA intrinsics compile
rem   -fopenmp      enable the #pragma omp parallel for in the parallel kernels
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
