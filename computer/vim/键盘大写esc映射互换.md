[TOC]

# 键盘大写esc映射互换

- 按 `Win + R` → 输入 `regedit` → 回车
- 定位到`HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Keyboard Layout`
- 右键 → **新建 → 二进制值** → 名称填：`Scancode Map`
- 双击 `Scancode Map`，输入以下二进制内容（互换 Caps Lock 与 Esc）：

~~~go
00 00 00 00 00 00 00 00
03 00 00 00 01 00 3A 00
3A 00 01 00 00 00 00 00
~~~

- 保存后重启