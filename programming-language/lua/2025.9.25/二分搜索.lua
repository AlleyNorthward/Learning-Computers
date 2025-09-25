function get_value(arr, target, left, right)
    left = left or 1    
    right = right or #arr
    if left > right then 
        return nil

    end
    
    mid = math.floor((left + right) / 2)
    if arr[mid] == target then
        return mid
    elseif arr[mid] < target then
        return get_value(arr, target, mid + 1, right)
    else 
        return get_value(arr, target, left, mid - 1)
    end

end

arr = {1, 3, 5, 9, 2, 4, 7, 8, 10, 6}
table.sort(arr)

target = 7
index = get_value(arr, target)
print(index)

-- 以前以为用递归做的,回头一看,原来是用while做的...
-- 整个部分没啥好说的,math.floor()是取整的意思.
