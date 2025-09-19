[TOC]

# 基本操作

## 1.进入某盘/文件

cd F

cd F/dir_name

# 2.返回上一目录

cd ..

cd ..\\..

# 3.新建文件夹

mkdir

# 4.删除文件夹

rmdir -Recurse

# 5.查看当前路径文件

ls/dir

# 6.终端分屏

alt shift D

# 7.删除当前页面

ctrl shift w

# 8.切换分屏

alt + 方向

# 9.快速删除外界窗口

ctrl + w

# 10.快速返回根目录

cd \

# 11.文件重命名
Rename-Item "pre_name" "new_name"

# 12.复制当前路径
Get-Location | Set-Clipboard

# 13.复制、剪切文件到指定路径
Move-Item file1.txt -Destination newfolder\file1_new.txt

# 14.切换vim语法与返回普通语法
Set-PSReadLineOption -EditMode Vi
Set-PSReadLineOption -EditMode Windows

# 15.配置自己命令行,快速打开文件

- vim打开文件
`nvim $PROFILE`

- 设置函数
```powershell
function myfile {
    nvim "F:\test.py"
}
```

