$PATH_FILE = "F:\0github\Learning-Computers\computer\terminal\powershell\powershell-module\package\path_marks.json"
$Global:PathMarks = @{}

if (Test-Path $PATH_FILE){
    $Global:PathMarks = Get-Content $PATH_FILE | ConvertFrom-Json
}

function Move-to {
    param([string]$Name)
    if ($Global:PathMarks.PSObject.Properties.Name -contains $Name){
        Set-Location $Global:PathMarks.$Name
        Write-Host "move to $($Global:PathMarks.$Name)"
    } else{
        Write-Host "move failed: path name not found"
    }
}

Export-ModuleMember -Function Move-to
