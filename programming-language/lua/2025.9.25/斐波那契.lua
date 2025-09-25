function fact(n)
    if n == 0 then
        return 1

    else 
        return n * fact(n - 1)

    end

end

print("请输入一个数字:")

a = io.read("*n")

print(fact(a))

-- then的t是小写,没有冒号
-- print别忘添加字符串
-- end没有自动补全
-- *n代表读入数字
-- *line代表读一行
-- *all代表读取全部
-- 10代表读取10个字符
