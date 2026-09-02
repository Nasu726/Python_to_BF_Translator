# Ordinary line-oriented Python input patterns supported by the compiler.

n = int(input())
a, b = map(int, input().split())
A = list(map(int, input().split()))

s = 0
for x in A:
    s += x

print(n)
print(a + b)
print(A)
print(s)
