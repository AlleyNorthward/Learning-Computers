function Invoke-LogoffThenShutdown-Once {
    [CmdletBinding()]
    param(
        [string]$TaskName = "TempShutdownAfterLogoff_Once",
        [int]$DelaySeconds = 10
    )

    if ($DelaySeconds -lt 5) {
        throw "DelaySeconds is too small. Please set it to at least 5 seconds to avoid a race condition."
    }

    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {

        throw "This function requires administrator privileges to create a SYSTEM-level scheduled task."
    }

    try {
        if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
            Write-Host "Detected an existing scheduled task with the same name, attempting to delete..." -ForegroundColor Yellow
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        }

        $action = New-ScheduledTaskAction -Execute "shutdown.exe" -Argument "/s /t 0"

        $runTime = (Get-Date).AddSeconds($DelaySeconds)
        $trigger = New-ScheduledTaskTrigger -Once -At $runTime

        $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -DeleteExpiredTaskAfter (New-TimeSpan -Minutes 10)

        Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Temporary one-time shutdown after logoff" -ErrorAction Stop

        if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
            Write-Host "Task registered: shutdown will execute at $runTime (approximately $DelaySeconds seconds later). Logging off the current user..." -ForegroundColor Yellow
        } else {
            throw "Scheduled task instance not found after registration. Registration may have failed. Logoff aborted to prevent accidental sign-out."
        }

        shutdown.exe /l
    }
    catch {
        Write-Error "An error occurred: $($_.Exception.Message)"
    }
}

Export-ModuleMember -Function Invoke-LogoffThenShutdown-Once
