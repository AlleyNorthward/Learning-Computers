function Save-position {
    Set-Variable -Name DirMark -Value (Get-Location) -Scope Global
    Write-Host "Location marked: $DirMark"
}

function Restore-position {
    if ($DirMark) {
        Set-Location $DirMark
	Write-Host "Restore Location."
    } else {
        Write-Host "No location has been marked yet."
    }
}


Export-ModuleMember -Function @("Save-position", "Restore-position")
