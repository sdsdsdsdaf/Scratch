import sys
input = sys.stdin.readline

n, m = map(int, input().split())
result = -1

for _ in range(n):
    data = list(map(int, input().split()))

    min_val = min(data)
    result = result if result >= min_val else min_val

print(result)

