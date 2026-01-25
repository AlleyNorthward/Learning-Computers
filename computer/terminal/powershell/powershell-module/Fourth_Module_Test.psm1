function Invoke-Keil {
    $UV4Path = "F:\0Embedded\Code\UV4\UV4.exe"

    $ProjectFile = Join-Path (Get-Location) "Template.uvprojx"
    if (-not (Test-Path $ProjectFile)) {
        Write-Host "未找到 Template.uvprojx，请确保在项目目录下运行"
        return
    }

    $LogDir = "F:\0Embedded\log"
    if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
    $LogFile = Join-Path $LogDir "build.log"

    if (Test-Path $LogFile) { Remove-Item $LogFile -Force }

    $process = Start-Process -FilePath $UV4Path -ArgumentList "-b `"$ProjectFile`" -j0 -t `"Target 1`" -o `"$LogFile`"" -PassThru -Wait -NoNewWindow

    if (Test-Path $LogFile) {
        Get-Content $LogFile
    } else {
        Write-Host "日志文件未生成，编译可能失败"
    }
}

Export-ModuleMember -Function Invoke-Keil 
