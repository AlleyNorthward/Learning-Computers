function Invoke-Main {
    param()

    # 设置源文件和输出文件
    $source = ".\main.c"
    $output = ".\main.exe"

    # 包含头文件路径
    $includePaths = @(
        "."
        "..\App\led"
        "..\App\clock"
        "..\App\beep"
        "..\App\key"
        "..\Public"
        "..\Libraries\CMSIS"
    )

    # 拼接 -I 参数
    $includeArgs = $includePaths | ForEach-Object { "-I `"$($_)`"" } | Out-String
    $includeArgs = $includeArgs -replace "\r?\n", " "  # 去掉换行

    # 编译命令
    $compileCmd = "gcc $source $includeArgs -o $output"

    Write-Host "Compiling..."
    Write-Host $compileCmd
    Invoke-Expression $compileCmd

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Compilation succeeded. Running $output ..."
        & $output
    } else {
        Write-Host "Compilation failed."
    }
}


Export-ModuleMember -Function Invoke-Main
