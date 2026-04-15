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


function Open-Google {
    & "C:\Program Files\Google\Chrome\Application\chrome.exe"
}

function Open-V2rayN {
  & "F:\0Embedded\v2rayN.exe.lnk"
}

function Open-xxt{
    & "C:\Program Files\Google\Chrome\Application\chrome.exe"  "https://i.chaoxing.com"
}

function Show-path{
    nvim "F:\0github\Learning-Computers\computer\terminal\powershell\powershell-module\package\path_marks.json"
}

function Open-github{
	& "C:\Program Files\Google\Chrome\Application\chrome.exe" "https://github.com"
}

function Open-pu{
  ssh -i "C:\Users\HP\.ssh\id_rsa" root@115.28.208.100
}

Export-ModuleMember -Function *
