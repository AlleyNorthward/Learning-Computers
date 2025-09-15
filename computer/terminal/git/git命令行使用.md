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

# 分支解析
明确,本地分支与远程分支不一致,需要建立联系即可
## 本地分支提交
- 如果本地分支提交,最后仍然push main,无任何发生.只能本地main提交到远程main,非常安全
## 块状管理
项目包含的每一部分,单独对其操作,就是建立对应分支,块状化管理、编写.所以需要要求我们会创建本地私有分支,同步创建远程分支,建立联系,整合到main

# 分支详解
## 本地分支操作
- 查看分支
~~~go
git branch
~~~
- 新建分支
~~~go
git checkout -b name
~~~
- 删除分支
~~~go
git branch -d name (安全删除,如有未整合内容,无法删除)
git branch -D name (强制删除)
~~~
- 分支转换
~~~go
git switch
~~~

## 远程分支操作
### 本地创建远程分支
- 查看远程分支
~~~go
git branch -r
~~~

- 远程分支建立
~~~go
git push -u origin name (-u 建立联系.本地建立后,第一次上传这么操作，之后恢复原装就好)
~~~
- 远程分支删除
~~~go
git push origin --delete name
~~~

### github建立远程带回本地
~~~go
git fetch origin
git checkout -b dev origin/dev
~~~

## 远程分支本地分支建立联系
~~~go
git push -u origin 分支名
git checkout -b 本地分支名 origin/远程分支名
git branch -u origin/远程分支名 本地分支名
~~~

## 查看分支联系
~~~go
git branch -vv
~~~


# 安全的git操作方式

- 查看分支
~~~go
git status
git branch
~~~
- 如果有多分支,切换到对应分支
~~~go
git checkout main 切换主分支
git checkout my-feature 切换到个人对应分支
~~~
- 同步仓库
~~~go
git pull origin main
~~~
- 创建新分支(不要直接在主分支上修改, 保护主分支)
~~~go
git checkout -b my-featuer (创建并切换到该分支)
~~~
- 再次检查仓库状态
~~~go
git status
~~~
确认无提交,无修改
- 提交
此时,有多种选择.

①提交到新建远程分支
~~~go
git add -A
git commit -m "提交修改"
git push -u origin my-feature
~~~
②提交到远程主分支
~~~go
git add -A
git commit -m "保存当前修改"
git checkout main (转换到主分支,之间转换不行)
git merge my-feature

或者
git stash
git checkout main
git stash pop (类似, 不如上面的安全)
~~~
- 合并分支
先切换到主分支
~~~go
git checkout origin main
~~~
- 合并
~~~go
git merge my-feature
~~~
- 如有冲突
解决冲突即可
- 推送
~~~go
git push origin main
~~~

# 分支解析
明确,本地分支与远程分支不一致,需要建立联系即可
## 本地分支提交
- 如果本地分支提交,最后仍然push main,无任何发生.只能本地main提交到远程main,非常安全
## 块状管理
项目包含的每一部分,单独对其操作,就是建立对应分支,块状化管理、编写.所以需要要求我们会创建本地私有分支,同步创建远程分支,建立联系,整合到main

# 分支详解
## 本地分支操作
- 查看分支
~~~go
git branch
~~~
- 新建分支
~~~go
git checkout -b name
~~~
- 删除分支
~~~go
git branch -d name (安全删除,如有未整合内容,无法删除)
git branch -D name (强制删除)
~~~

## 远程分支操作
### 本地创建远程分支
- 查看远程分支
~~~go
git branch -r
~~~

- 远程分支建立
~~~go
git push -u origin name (-u 建立联系.本地建立后,第一次上传这么操作，之后恢复原装就好)
~~~
- 远程分支删除
~~~go
git push origin --delete name
~~~

### github建立远程带回本地
~~~go
git fetch origin
git checkout -b dev origin/dev
~~~

## 远程分支本地分支建立联系
~~~go
git push -u origin 分支名
git checkout -b 本地分支名 origin/远程分支名
git branch -u origin/远程分支名 本地分支名
~~~

## 查看分支联系
~~~go
git branch -vv
~~~