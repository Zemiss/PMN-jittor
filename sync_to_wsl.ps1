# 同步文件到 WSL
# PowerShell 脚本

$sourceDir = "C:\Users\12445\Desktop\pybeta2.0"
$targetDir = "/home/x12345/py-wsl-beta"

Write-Host "开始同步文件到 WSL..." -ForegroundColor Green

# 同步关键文件
$files = @(
    "trainer_SID.py",
    "base_trainer.py",
    "losses.py",
    "utils.py",
    "archs\ELD_models.py",
    "archs\modules.py"
)

foreach ($file in $files) {
    $sourceFile = Join-Path $sourceDir $file
    $targetFile = "$targetDir/$($file -replace '\\', '/')"
    
    if (Test-Path $sourceFile) {
        Write-Host "同步: $file" -ForegroundColor Cyan
        wsl cp "$($sourceFile -replace '\\', '/' -replace 'C:', '/mnt/c')" "$targetFile"
    } else {
        Write-Host "警告: 文件不存在 $file" -ForegroundColor Yellow
    }
}

Write-Host "`n同步完成！" -ForegroundColor Green
Write-Host "现在可以在 WSL 中运行训练命令了。" -ForegroundColor Green
