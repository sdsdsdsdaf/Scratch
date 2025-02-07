import sys
input = sys.stdin.readline

N,M = map(int, input().split())
array = []
for _ in range(N):
    array.append(int(input()))
INF = 100001
dp = [INF]*(M+1)
dp[0] = 0

for i in range(M+1):
    for coin in array:

        if i - coin >= 0 :
            dp[i] = min(dp[i], dp[i-coin] + 1)

print(dp[M])