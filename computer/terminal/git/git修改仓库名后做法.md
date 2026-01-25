首先git修改仓库名,直接去网站修改就行(本地修改也不会)
修改好了之后,再自己将本地仓库名自己手动同步就行.
之后,查看一下当前远程地址.
~~~powershell
git remote -v
~~~

大概率是旧的仓库名.
此时同步一下就行了

~~~powershell
git remote set-url origin git@github.com:AlleyNorthward/new-name.git
~~~
然后再看一下同步成功没有

~~~powershell
git remote -v
~~~


远端修改似乎只需要一条命令

~~~powershell
git remote set-url origin git@github.com:yourname/manim-all-in-one.git
~~~

没试过哈,ai这么说的.
本地自己手动修改就行.
其实也可以自己写个powershell脚本,一个命令,自动修改云端名字+本地仓库名.
