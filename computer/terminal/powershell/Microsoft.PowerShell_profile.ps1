[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
oh-my-posh init pwsh --config $env:POSH_THEMES_PATH\M365Princess.omp.json | Invoke-Expression

Set-PSReadLineOption -EditMode Vi

function basic-command{
	nvim "F:\0github\Learning-Computers\computer\terminal\powershell\basic_operation.md"
}

function activate-manim{
	conda activate ManimCEV19_0
}

function _edit{
	nvim F:\0github\Learning-Computers\computer\terminal\powershell\Microsoft.PowerShell_profile.ps1
}

function _copy{
        pwd | Select-Object -ExpandProperty Path | Set-Clipboard
}

function _github{
	cd F:\0github
}

function _gpt{
	& "C:\Program Files\Google\Chrome\Application\chrome.exe" "https://chatgpt.com/"
}

function _ps{
	cd F:\Project-Simulation\simulation0
}

function _xxt{
	start "https://i.chaoxing.com/base?vflag=true&fid=&backUrl="
}

function _computer{
	cd F:\0github\Learning-Computers\computer
}

function _mobject{
	cd F:\0github\Manim-Mobject\Mobject
}
