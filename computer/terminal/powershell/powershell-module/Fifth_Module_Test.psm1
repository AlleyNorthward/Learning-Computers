function Set-Opacity {
    param(
        [Parameter(Position=0)]
        [ValidateRange(0,100)]
        [int]$opacity = 25,

        [Parameter(Position=1)]
        [bool]$useAcrylic = $false
    )

    $Path = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"

    if (-not (Test-Path $Path)) {
        Write-Error "找不到 Windows Terminal 配置文件: $Path"
        return
    }

    try {
        $Json = Get-Content $Path -Raw | ConvertFrom-Json

        $Json.profiles.defaults.opacity    = $opacity
        $Json.profiles.defaults.useAcrylic = $useAcrylic

        $Json | ConvertTo-Json -Depth 20 |
            Set-Content $Path -Encoding UTF8

        Write-Host "已设置: 透明度=$Opacity, 亚克力=$UseAcrylic"
    }
    catch {
        Write-Error "修改失败: $_"
    }
}
Export-ModuleMember -Function Set-Opacity
