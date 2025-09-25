function fact(n)
    if n == 0 then
        return 1
    else
        return n * fact(n - 1)
    end

end

print("请输入一个数字")

a = io.read('*n')
print(fact(a))
