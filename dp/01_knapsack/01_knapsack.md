# 0-1 배낭 문제

0-1 배낭 문제는 유명한 최적화 문제 중 하나입니다. 이 문제에서는 물건의 가치와 무게가 주어졌을 때, 주어진 가방의 용량을 초과하지 않는 선에서 최대 가치를 얻을 수 있는 물건들의 조합을 찾는 것이 목표입니다.

여기서 "0-1"이란 각 물건을 선택하거나 선택하지 않거나 두 가지만 존재함을 나타내는 것입니다.

백준 `평범한 배낭` 문제를 풀며 이해해 봅시다.

## 평범한 배낭

준서는 여행을 가려합니다. 여행에 필요한 N개의 물건이 있습니다. 각 물건은 무게 W와 가치 V를 갖습니다. 준서의 배낭이 최대 K만큼의 무게를 버틸 수 있을 때, 가치의 합의 최댓값을 출력해 주세요.

간단하게 말하면 배낭에 담을 수 있는 무게의 최댓값이 주어졌을 때, 가치의 합이 최대가 되도록 짐을 고르면 됩니다.

### 입력

첫 줄에 물품의 수 N과 준서가 버틸 수 있는 무게 K가 주어집니다.

두 번째 줄부터 N개의 줄에 거쳐 각 물건의 무게 W와 해당 물건의 가치 V가 주어집니다.

### 제한사항

- 1 <= N <= 100
- 1 <= K <= 100,000
- 1 <= W <= 100,000
- 0 <= V <= 1,000

### 예제

입력

```
4 7
6 13
4 8
3 6
5 12
```

출력

```
14
```

### 문제 풀이

조합과 DP, 두 가지 방법으로 풀어보겠습니다.

#### 조합

물건을 선택하는 순서에 상관없이 배낭의 최대 무게를 넘지 않는 선에서 물건들의 가치의 최댓값을 찾으면 되므로 조합으로 풀 수 있습니다.

조합 코드의 순서는 아래와 같습니다.

1. 물건과 stuff, N, K를을 받는 combinations 함수를 정의합니다.
2. start와 weight, value를 받는 backtrack 함수를 combinations 내부에 정의합니다.
    1. start와 N을 비교합니다.
        1. 현재 value가 max_value보다 크면 max_value에 value를 넣습니다.
        2. 함수를 종료합니다.
    2. start부터 N까지 반복합니다.
       (이는 지금까지 선택했던 원소를 제외한 모든 원소를 고려하기 위함입니다.)
        1. 가방에 현재 물건이 들어갈 수 없는지 확인합니다.
            1. 현재 value가 max_value보다 크면 max_value에 value를 넣습니다.
            2. 다음 요소로 넘어갑니다.
        2. 이번 요소를 포함하고 다음 요소부터 다시 backtrack()을 실행합니다.

```python
def combinations(stuff, N, K):
    def backtrack(start, weight, value):
        nonlocal result

        if start == N:
            if value > result:
                result = value
            return

        for i in range(start, N):
            if weight + stuff[i][0] > K:
                if value > result:
                    result = value
                continue
            backtrack(i + 1, weight + stuff[i][0], value + stuff[i][1])

    result = 0
    backtrack(0, 0, 0)
    return result


N, K = map(int, input().split())
stuff = []

for _ in range(N):
    W, V = map(int, input().split())
    stuff.append((W, V))

print(combinations(stuff, N, K))
```

위와 같이 작성하면 분명 0-1 배낭 문제의 답이 나옵니다. 하지만 `O(N!)`의 시간 복잡도를 갖기 때문에 시간 제한을 통과하지 못합니다.

조합은 중복 계산이 매우 많습니다. 예를 들어, [A, B, C, D, E] 라는 물건이 주어졌을 때,
[A, B, C, D]와 [B, C, D, E]는 서로 동일한 [B, C, D]를 갖고있음에도 반복하여 계산합니다. 이러한 중복 계산을 반복하여 계산하기 때문에 조합이 매우 느린 것입니다.

DP를 사용하면 이러한 반복 되는 계산을 기억해 두었다가 재활용할 수 있습니다.

#### DP

DP는 문제를 빨리 풀기 위해 답을 재활용하는 기술이므로 이 문제에 적용한다면, 시간 복잡도를 상당히 향상시킬 수 있을 것입니다.

<details>
<summary>0-1 배낭 문제에 DP를 적용할 수 있는 이유 </summary>
우선 DP를 적용하기 위해서는 해당 문제가 `최적 부분 구조` 와 `부분 반복 문제` 두 가지 성질을
갖고 있어야 합니다.

0-1 배낭 문제에서 배낭이 총 1 만큼의 무게를 견딜 수 있을 때의 최적의 해가 배낭이 총 2 만큼의 무게를 견딜 수 있을 때의 최적의 해를 구하는 데 사용될 수 있으므로 최적 부분 구조입니다.

또한, 배낭이 총 n 만큼의 무게를 견딜 수 있을 때의 최적의 해를 구할 때 n-1 만큼의 무게를 견딜 수 있을 때 계산했던 것들을 중복으로 요하므로 부분 반복 문제입니다.

따라서 0-1 배낭 문제는 DP를 적용할 수 있습니다.
</details>

순서는 아래와 같습니다.

1. x축에는 가방의 무게, y축에는 n의 수 만큼의 2차원 배열을 준비합니다.
2. 물건과 배낭의 크기를 차례대로 방문합니다.
3. 현재 물건을 넣을 수 없다면 이전의 가치를 넣고 현재 물건을 넣을 수 있다면 max(현재 물건의 가치 + 이전 배낭에서의 현재 물건 무게를 제외했을 때의 가치, 이전의 가치)를 실행합니다.
4. 모든 물건과 모든 배낭 크기에 대해 이를 반복합니다.

```python
def knapsack_bottom_up(limit: int, input_list, n):
    dp = [[0 for _ in range(limit + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, limit + 1):
            if input_list[i - 1][0] > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], input_list[i - 1][1] + dp[i - 1][j - input_list[i - 1][0]])

    return dp[n][limit]


N, K = map(int, input().split())
items = [list(map(int, input().split())) for _ in range(N)]

print(knapsack_bottom_up(K, items, N))
```

## 출처

- [평범한 배낭- 백준](https://www.acmicpc.net/problem/12865)
