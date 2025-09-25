-- 这里只是简单总结,不做运行
-- 根据个人学习哲学,这没用,仅做了解

-- 1.参数不固定,多传忽略,少传位nil
function f(a, b)
    print(a, b)

end

f(1, 2)  --> 1, 2
f(1)     --> 1, nil
f(1, 2, 4)--> 1, 2 忽略

-- 2.默认参数,可通过内部实现
function greet(name)
    name = name or "Guest" --默认参数
    print("Hello, "..name)
end

greet("Alice")
greet()

-- 3. 可变参数...
function sum(...)
    local total = 0
    for i, v in ipairs({...}) do
        total = total + v

    end
    return total
end
print(sum(1, 2, 3, 4)) --> 10 ipairs 类似enumerate

--  4. 关键字参数(用表模拟)
function connect(opts)
    local host = opts.host or "localhost"
    local port = opts.port or 80

    print("Connecting to "..host..":"..port)
end

connect{host = "127.0.0.1", port = 8080}
connect{} -- 此时忽略了()

-- 5.多返回值
function divide(a, b)
    return math.floor(a/b), a%b
end

q, r = divide(10, 3)
print(q, r) --> 3, 1

-- 6.传递函数
function apply(f, x)
    return f(x)
end
print(apply(math.sqrt, 16)) --> 4

-- 7. 传递可变参数到另一个函数
function wrapper(...)
    print("Before call")
    return math.max(...)

end

print(wrapper(1, 5, 3)) --> 5

