@echo off
REM Git Repository Corruption Fix Script for Windows

echo === Git Repository Corruption Fix ===
echo.

REM Step 1: Remove corrupted pack
echo Step 1: Removing corrupted pack files...
del /F /Q .git\objects\pack\pack-9635e49f9e6ba3e1fc7e39ed440b4d619da0aa43.* 2>nul
echo Done
echo.

REM Step 2: Run git fsck
echo Step 2: Running git fsck...
git fsck --full
echo.

REM Step 3: Repack
echo Step 3: Repacking repository...
git repack -a -d
echo Done
echo.

REM Step 4: Garbage collection
echo Step 4: Running garbage collection...
git gc --aggressive --prune=now
echo Done
echo.

echo === Fix Complete ===
echo Now try: git push -u origin main
pause
