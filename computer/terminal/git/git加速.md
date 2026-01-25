# git加速讲解

很无语,使用vim,发现github中的内容还clone不了,我自己平时都能克隆啊,怎么到vim这里就不行了.况且还开着科学加速器,实在无语之下,深入ai,解决了问题,其实也不困难.

# 配置端口信息
```powershell
git config --global http.proxy socks5://127.0.0.1:10808
git config --global https.proxy socks5://127.0.0.1:10808
```
后面是端口信息,怎么说呢,我也不知道是不是固定的...

# 查看是否能连接git
```
git ls-remote https://github.com/neoclide/coc.nvim.git

```

# 不行的话还有排查方式,我这里就不说了

这样做大概率就行了.据说是全局配置的哈
