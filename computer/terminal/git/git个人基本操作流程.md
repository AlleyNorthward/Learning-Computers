[TOC]
# git个人基本操作流程

## 简述
个人习惯使用.莫叫
远程无分支, 本地有分支.
本地整合后,统一提交到远程.
## 流程
### 查看分支
~~~go
git status
git branch
~~~
### 切换到目的分支
~~~go
git checkout main 切换主分支
git checkout my-feature 切换到个人对应分支
git checkout -b my-featuer (无目的分支,创建对应分支)
~~~
### 同步仓库
~~~go
git pull origin main
~~~
### 再次检查仓库状态
~~~go
git status
~~~
### 修改对应文件
pass
### 分支保存commit
~~~go
git add -A
git commit -m "对应操作"
~~~
### 转换到main
~~~go
git checkout main
~~~
### 整合
~~~go
git merge name
~~~
### 提交
~~~go
git push origin main
~~~
### 结束
以后差不多就这么个流程.
为什么不直接在main操作?安全性是其次吧,主要还是练习操作.好了如今也大差不差了,就以此作为最后一次测试(这个修改是在本地git文档整理分支写的.保存,退出后,进来看看main中有没有,然后再切换到该分支,看看这些内容有无保存,是不是做到真的安全)
测试结果:本地内容全在,git进入分支是最后退出的分支,无add,无commit,没有意外发生.当前也无法切换到主分支,只能add,commit后切换.总之,使用分支十分安全.