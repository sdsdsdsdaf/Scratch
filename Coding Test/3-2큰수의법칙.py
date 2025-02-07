import sys
input = sys.stdin.readline

n, m, k = map(int, input().split())
data = list(map(int, input().split()))

data.sort()

first_big = data[-1]
sec_big = data[-2]

del data

sec_big_rep_time = m // (k+1)
first_big_rep_time = m - sec_big_rep_time

result = first_big*first_big_rep_time + sec_big*sec_big_rep_time

print(result)