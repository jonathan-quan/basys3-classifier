@echo off
rem Push this project to an EMPTY GitHub repository you created at https://github.com/new
cd /d "%~dp0"
echo.
set "URL="
set /p "URL=Paste your new GitHub repo URL (https://github.com/USER/REPO.git): "
if "%URL%"=="" (echo No URL entered.& pause & exit /b 1)
git remote get-url origin >NUL 2>&1
if errorlevel 1 (git remote add origin "%URL%") else (git remote set-url origin "%URL%")
git push -u origin main
set RC=%errorlevel%
(echo EXIT=%RC%)> jobs\push.status
echo.
if "%RC%"=="0" (echo PUSH OK - open %URL%) else (echo PUSH FAILED - exit code %RC%)
pause
