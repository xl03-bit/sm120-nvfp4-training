@echo off
setlocal
if not defined H3_PYTHON set "H3_PYTHON=python"
pushd "%~dp0"
"%H3_PYTHON%" -u %*
set "H3_EXIT=%ERRORLEVEL%"
popd
exit /b %H3_EXIT%
