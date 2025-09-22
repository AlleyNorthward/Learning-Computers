function Open-command{
	nvim "F:\0github\Learning-Computers\computer\terminal\powershell\basic_operation.md"
}

function Start-manim{
	conda activate ManimCEV19_0
}

function Edit-profile{
	nvim "F:\0github\Learning-Computers\computer\terminal\powershell\powershell-module\First_Module_Test.psm1"
}

function Copy-path{
        pwd | Select-Object -ExpandProperty Path | Set-Clipboard
}


function Open-gpt{
	& "C:\Program Files\Google\Chrome\Application\chrome.exe" "https://chatgpt.com/"
}


function Open-xxt{
	start "https://i.chaoxing.com/base?vflag=true&fid=&backUrl="
}


function Open-path{
        nvim "F:\0github\Learning-Computers\computer\terminal\powershell\powershell-module\package\path_marks.json"
}

function Open-github{
	& "C:\Program Files\Google\Chrome\Application\chrome.exe" "https://github.com"
}
Export-ModuleMember -Function *
