function hannuo(n, A, B, C, moves)
    moves = moves or {}
    if n == 0 then
        return moves
    end

    hannuo(n-1, A, C, B, moves)
    table.insert(moves, {disk = n, from = A, to = C})
    hannuo(n-1,B, A, C,moves)
    return moves

end

local moves = hannuo(3, "A", "B", "C")
for i, m in ipairs(moves) do
    print(i, string.format("disk %d:%s -> %s", m.disk, m.from, m.to))

end

print("Total:", #moves)

-- ..A..代表字符串拼接
-- 注意到,调用函数时,少穿了一个参数,第二行
-- 总结一下lua传参技巧,了解就行,等需要时,再回过头来看即可
-- 注意到表的insert方法的使用,并不是传统的方法形式存在,而是table.insert()插入,类似append使用
-- #是长度运算符,获取当前对象长度
