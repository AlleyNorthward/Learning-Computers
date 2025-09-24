# git分支使用规定

@auther 巷北
@time 2025.9.24 20:01
目前只用着本地的main,一直没有使用分支.但是调试,运行代码时有些麻烦.比如我想调试Mobject中某些对象,需要复制到本地,来回切换目录,copy,等等等等,相对而言,比较麻烦.
后面想了想,似乎分支测试是比较好的方式.之前可能认为,会在main和分支之间搞混乱了,所以一直没弄,因为代码提交只从main中提交就好.目前来看,测试代码不得不使用分支了,所以做出如下规定,避免发生问题.

# 1
分支永远不得提交到origin main 分支(不使用git add)

# 2
永远只从main更新到origin main(main不使用git pull)

# 3
不从网页提交任何代码

# 4
分支永远从云端获取代码(只使用merge origin main,不得使用merge main,如果发现main和origin main不统一,更新originmain)

# 5
分支不得合并到main(不得使用merge)

# 6
测试结束,使用git add . git commit 保存分支代码

# 总结
分支需要 merge origin main, git add
main只能add
似乎只用这两个动作

测试了origin main 与分支不一致情况下merge,目前没有任何问题.有问题时,再说.但一般情况下,分支是最新代码,不要更新到main,再通过main提交到origin main,所以之后可能不会merge origin main.只会在md文档中更新的时候选择merge.



