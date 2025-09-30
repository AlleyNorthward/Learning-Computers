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
	start "https://mooc2-ans.chaoxing.com/mooc2-ans/mycourse/stu?courseid=227595041&clazzid=126777158&cpi=340219436&enc=3d4e0f77a3654995081514b49e2b98bc&t=1759227129168&pageHeader=21&v=0&hideHead=0"
}


function Show-path{
        nvim "F:\0github\Learning-Computers\computer\terminal\powershell\powershell-module\package\path_marks.json"
}

function Open-github{
	& "C:\Program Files\Google\Chrome\Application\chrome.exe" "https://github.com"
}
Export-ModuleMember -Function *
