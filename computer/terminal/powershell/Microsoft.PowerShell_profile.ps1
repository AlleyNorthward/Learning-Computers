function Open-command{
	nvim "F:\0github\Learning-Computers\computer\terminal\powershell\basic_operation.md"
}

function Start-manim{
	conda activate ManimCEV19_0
}

function Edit-profile{
	nvim F:\0github\Learning-Computers\computer\terminal\powershell\Microsoft.PowerShell_profile.ps1
}

function Copy-path{
        pwd | Select-Object -ExpandProperty Path | Set-Clipboard
}

function Open_github{
	cd F:\0github
}

function Open_gpt{
	& "C:\Program Files\Google\Chrome\Application\chrome.exe" "https://chatgpt.com/"
}

function Move_ps{
	cd F:\Project-Simulation\simulation0
}

function Open_xxt{
	start "https://i.chaoxing.com/base?vflag=true&fid=&backUrl="
}

function Move_computer{
	cd F:\0github\Learning-Computers\computer
}

function Move_mobject{
	cd F:\0github\Manim-Mobject\Mobject
}
