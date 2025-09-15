[TOC]

# git命令行使用

## 克隆仓库

- 克隆仓库

~~~go
git clone + 地址
~~~

- 进入仓库

~~~go
cd + 仓库名
~~~

- 查看文件

~~~ go
dir + 文件名
~~~

## 上传代码

身份验证这块就不说了

- 初始化git

~~~go
git init
~~~

- 添加文件

~~~ go
git add + 文件名
~~~

- 提交信息

~~~go
git commit -m "提交信息"
~~~

- 提交

~~~go
git push
~~~

git commit 必不可少

## 查看.git

~~~go
ls -Force
~~~

静默文件,本地看不到

## 重命名文件

~~~go
git mv 旧文件名 新文件名
git commit -m "重命名文件"
git push
~~~

## git add 拆解

~~~go
git add   某文件  无法操作删除
git add . 当前文件夹 无法操作删除
git add -A 所有	能操作删除
~~~

## 查看git add 内容

~~~go
git status
~~~

## 状态标识

| 标识  | 含义                                                |
| ----- | --------------------------------------------------- |
| **U** | **Untracked** → 文件未被 Git 追踪（还没 `git add`） |
| M     | Modified → 文件已修改（修改过但还没 commit）        |
| A     | Added → 文件已添加到暂存区（`git add`）             |
| D     | Deleted → 文件已删除（已 `git add` 暂存区）         |
| R     | Renamed → 文件重命名了                              |
| C     | Copied → 文件被复制了                               |

## 撤销git add内容

- 单文件

~~~go
git restore --staged 文件名
~~~

- 多文件

~~~go
git restore --staged .
or
git reset
~~~

## git 直接下载参考命令行

~~~go
git clone git@github.com:AlleyNorthward/VSCode-Usage-Summary.git
~~~

# github本地ssh密钥建立

## 查看旧公钥

~~~go
type $env:USERPROFILE\.ssh\id_rsa.pub
~~~

有就删除，没有就跳过删除这一步

## 删除旧公钥

~~~go
C:\Users\<你的用户名>\.ssh\id_rsa       # 私钥
C:\Users\<你的用户名>\.ssh\id_rsa.pub   # 公钥
~~~

## 生成新公钥

~~~go
ssh-keygen -t rsa -b 4096 -C "你的邮箱@example.com"
~~~

一直回车就行

## 添加新公钥到github中

路径：GitHub → **Settings → SSH and GPG keys → New SSH key**

输入命令行

~~~go
type $env:USERPROFILE\.ssh\id_rsa.pub
~~~

将所有信息复制到new ssh key 中

## 测试是否成功

~~~go
ssh -T git@github.com
~~~

成功后，就可以直接使用下面命令行下载了

~~~go
git clone git@github.com:AlleyNorthward/VSCode-Usage-Summary.git
~~~

感觉能避免网络限制



