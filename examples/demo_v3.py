name = input()
x = int(input())
y = int(input())

lo, hi = min(x, y), max(x, y)
print("hello", name)
print("sum", x + y, sep=": ")
print("product", x * y, sep=": ")
print("ordered", lo, hi, sep=" | ")

if x < y < 100:
    print("x < y < 100")
else:
    print("comparison failed")
