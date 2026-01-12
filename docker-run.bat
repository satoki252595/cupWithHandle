@echo off
REM Cup with Handle Pattern Detector - Docker Helper Script for Windows
REM ====================================================================
REM
REM Usage:
REM   docker-run.bat build          - Build Docker image
REM   docker-run.bat scan 7203      - Scan specific stock
REM   docker-run.bat sample         - Run sample test
REM   docker-run.bat ml 7203        - ML prediction mode
REM   docker-run.bat train          - Train ML models

setlocal enabledelayedexpansion

if "%1"=="" goto :help
if "%1"=="help" goto :help
if "%1"=="--help" goto :help
if "%1"=="-h" goto :help

if "%1"=="build" goto :build
if "%1"=="scan" goto :scan
if "%1"=="sample" goto :sample
if "%1"=="ml" goto :ml
if "%1"=="full" goto :full
if "%1"=="train" goto :train
if "%1"=="backtest" goto :backtest
if "%1"=="evaluate" goto :evaluate
if "%1"=="shell" goto :shell
if "%1"=="clean" goto :clean

echo Unknown command: %1
goto :help

:help
echo Cup with Handle Pattern Detector - Docker Helper
echo.
echo Usage: docker-run.bat ^<command^> [options]
echo.
echo Commands:
echo   build              Build Docker image
echo   scan ^<symbols^>     Scan specific stocks (e.g., scan 7203 6758)
echo   sample [type]      Run sample test (nikkei/semiconductor/growth)
echo   ml ^<symbols^>       ML-enhanced prediction
echo   full               Full screening (all Japanese stocks)
echo   train              Train ML models
echo   backtest           Run backtest
echo   evaluate           Evaluate ML models
echo   shell              Interactive bash shell
echo   clean              Remove containers and images
echo.
echo Examples:
echo   docker-run.bat build
echo   docker-run.bat scan 7203 6758 8035
echo   docker-run.bat sample nikkei
echo   docker-run.bat ml 7203
goto :eof

:build
echo Building Docker image...
docker-compose build
echo Build complete!
goto :eof

:scan
shift
if "%1"=="" (
    echo Error: Please specify stock symbols
    echo Usage: docker-run.bat scan 7203 6758 8035
    goto :eof
)
echo Scanning stocks: %*
docker-compose run --rm scanner %*
goto :eof

:sample
shift
set SAMPLE_TYPE=%1
if "%SAMPLE_TYPE%"=="" set SAMPLE_TYPE=nikkei
echo Running sample test: %SAMPLE_TYPE%
docker-compose run --rm scanner --sample %SAMPLE_TYPE%
goto :eof

:ml
shift
if "%1"=="" (
    echo Error: Please specify stock symbols
    echo Usage: docker-run.bat ml 7203 6758
    goto :eof
)
echo ML prediction for: %*
docker-compose run --rm scanner --ml-mode %*
goto :eof

:full
echo Starting full screening (this may take a while)...
docker-compose run --rm scanner
goto :eof

:train
shift
echo Training ML models...
docker-compose run --rm trainer %*
goto :eof

:backtest
echo Running backtest...
docker-compose run --rm backtest
goto :eof

:evaluate
echo Evaluating ML models...
docker-compose run --rm evaluator
goto :eof

:shell
echo Starting interactive shell...
docker-compose run --rm shell
goto :eof

:clean
echo Removing containers and images...
docker-compose down --rmi local -v
echo Cleanup complete!
goto :eof
