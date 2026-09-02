name = input()
a, b = map(int, input().split())
values = list(map(int, input().split()))

values.append(a + b)
print("hello", name)
print("pair", a, b, sep=" | ")
print("values", values, sep=": ")

s = 0
for x in values:
    s += x
print("sum", s, sep=": ")
