-- 思考一下如何解决.传入数据是一个表,数字是1,2,3,4
-- 函数参数是传入表table,过度表temp_table,结果表result_table

function get_permutation(tab, temp_tab, result_tab)
    temp_tab = temp_tab or {}
    result_tab = result_tab or {}

    if #tab <= 0 then
        tem = {table.unpack(temp_tab)}

        table.insert(result_tab, tem)
        return result_tab
    end

    for i = 1, #tab do
        temp = {table.unpack(tab)}

        table.remove(temp, i)
        table.insert(temp_tab, tab[i])
        get_permutation(temp, temp_tab, result_tab)
        table.remove(temp_tab)
    end
    return result_tab
end


p = {1, 2, 3, 4}

result = get_permutation(p)
--print(result)
for i, r in ipairs(result) do
    print(i, table.concat(r, ", "))
end

print("Total:", #result)


-- 这是按照之前的python算法写的,效率很低,但目前不考虑效率,只考虑语法
-- unpack方法,解包操作,模仿复制操作,且存放的不是原变量的引用
-- 变量循环方式,不一定非得是pair或ipair,传统调用方式如图
-- remove和insert都默认尾部操作,加上参数pos则是对应位置操作
-- remove有两个参数,obj和pos,insert有三个参数,obj,pos,value
-- 输出方式不让我直接输出,具体操作如图.
