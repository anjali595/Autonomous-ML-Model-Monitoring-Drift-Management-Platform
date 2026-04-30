@echo off
setlocal
set NODE_DIR=%~dp0\node\node-v24.14.1-win-x64
if not exist "%NODE_DIR%\node.exe" (
  echo Portable Node.js not found in %NODE_DIR%.
  exit /b 1
)
set PATH=%NODE_DIR%;%PATH%
"%NODE_DIR%\npm.cmd" %*
