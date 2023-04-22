# 동적 계획법

동적 계획법(Dynamic Programming, DP)이란 문제를 여러 하위 문제로 나누어 푼 뒤, 큰 문제에서 그 답을 재활용하여 문제 풀이 시간을 단축하는 최적화 기법입니다. 단순하게 말하자면 문제를 빨리 풀기
위해 답을 재활용하는 것입니다.

동적 계획법을 적용하기 위해서는 문제가 `최적 부분 구조(optimal substructure)` 과 `부분 반복 문제`
라는 두 가지 속성을 갖고 있어야 합니다.

## 최적 부분 구조

`최적 부분 구조(optimal substructure)` 란 하위 문제의 최적의 답으로 큰 문제의 최적의 답을 구할 수 있는 구조를 뜻합니다.

피보나치의 `f(n)`를 구하는 경우 `f(n) = f(n-1) + f(n-2)` 이므로
`f(n-1)` 와 `f(n-2)` 을 구하면 `f(n)` 를 구할 수 있습니다. 부분 문제인 f(n-1)와 f(n-2)의 최적의 답을 통해 전체 문제인 f(n)의 최적의 답을 구하는 문제이므로 피보나치
문제는 `최적 부분 구조` 를 갖습니다.

## 부분 반복 문제

`부분 반복 문제(overlapping subproblems)` 란 하위 문제의 해결에 사용되는 값이 다른 하위 문제에도 중복으로 사용되는 것을 뜻합니다. 마찬가지로 피보나치 수열을 생각해 봅시다.

![b](https://user-images.githubusercontent.com/50406129/233780742-438a72fb-9414-495f-81eb-b71e41406261.png)

위 사진은 피보나치 수열의 5번째 수를 구하는 재귀 함수의 호출을 그래프로 나타낸 것입니다. f(0), f(1), f(2), f(3)이 단 한 번 호출되는 것이 아니라 두 번 이상 호출되는 것을 확인할 수 있습니다.
이처럼 작은 문제들의 해결에 사용되는 부분 문제들이 서로 겹치는 것을
`부분 반복 문제` 라 합니다.

## 구현

동적 계획법은 Top-down( tabulation ) 혹은 Bottom-up( memoization ) 방식으로 구현됩니다. Top-down은 말 그대로 위에서부터 아래로 내려오며 동적 계획법을 적용하는 것이고
Bottom-up은 바닥에서부터 위로 올라가며 동적 계획법을 적용하는 것입니다.

피보나치 문제는 `최적 부분 구조`와 `부분 반복 문제` 의 성질을 모두 가지므로 DP를 적용할 수 있습니다. 피보나치 문제를 기준으로 DP를 적용하지 않았을 때(재귀 함수)와 DP를 적용했을 때(Bottom-up,
Top-down)를 구현해 보겠습니다.

### 재귀 함수

재귀 함수를 통해 피보나치를 구현하면 다음과 같습니다.

```python
def fib_recursion(n):
    if n < 2:
        return n
    return fib_recursion(n - 1) + fib_recursion(n - 2)
```

위 코드에서 `fib(5)` 를 호출하면 동일한 값에 대해 여러 번 함수를 호출하는 트리가 생성됩니다.

![b](https://user-images.githubusercontent.com/50406129/233780742-438a72fb-9414-495f-81eb-b71e41406261.png)

1. `fib(5)`
2. `fib(4) + fib(3)`
3. `(fib(3) + fib(2)) + (fib(2) + fib(1))`
4. `((fib(2) + fib(1)) + (fib(1) + fib(0))) + ((fib(1) + fib(0)) + fib(1))`
5. `(((fib(1) + fib(0)) + fib(1)) + (fib(1) + fib(0))) + ((fib(1) + fib(0)) + fib(1))`

fib(2)의 값을 한 번 계산한 뒤 그 값을 `재활용`하면 이후에 다시 fib(2)의 값이 필요할 때 fib(0) + fib(1)을 호출할 필요가 없을 거라는 생각이 듭니다.

앞서 언급했듯 DP는 답을 `재활용` 하여 문제 풀이 시간을 단축시키는 최적화 기법이므로, 우리의 요구 사항을 들어줄 수 있을 것 같습니다.

자 이제 DP를 적용해 봅시다.

### Bottom-up

```python
def fib_bottom_up(n):
    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]


def fib_bottom_up2(n):
    a, b = 0, 1
    for i in range(2, n):
        a, b = b, a + b
    return a + b
```

위 코드에서 `fib(5)` 를 호출하면 아래와 같은 트리가 생성됩니다.

![bb](https://user-images.githubusercontent.com/50406129/233783713-6aeb49b4-082d-4116-8166-eb844cd2b3d4.jpg)

DP를 적용하지 않았을 때와 비교하면 확연한 차이를 느낄 수 있습니다. 아주 작은 단위인 fib(5)의 차이가 이 정도니까 더 큰 수라면 훨씬 큰 차이가 날것입니다. 실제로 재귀를 통한 구현은 O(1.6^n)인데
반해 DP를 사용한 구현은 O(2n)입니다. 참고로, fib_bottom_up()의 공간 복잡도는 O(n)이고, fib_bottom_up2()의 공간 복잡도는 O(1)입니다.

### Top-down

마지막으로 Top-down 식으로 구현해 보겠습니다. Bottom-up과 유일한 차이점은 그저 어느 방향에서 시작해서 어느 방향으로 향하느냐의 차이입니다. 시간 복잡도의 차이는 없습니다.

```python
def fib_top_down(n):
    dp = [0] * (n + 1)

    def fib(m):
        if m < 2:
            return m
        if dp[n] != 0:
            return dp[m]
        return fib(m - 1) + fib(m - 2)

    return fib(n)
```

## 동적 계획법 vs 그리디 알고리즘 vs 분할 정복

`동적 계획법` 은 문제를 여러 하위 문제로 나누고 모든 경우를 계산합니다. 단, 계산 도중 중복되는 계산이 요구될 수 있으므로 `값을 기억` 해뒀다가 필요할 때 재사용합니다.

`그리디 알고리즘` 은 `미래를 고려하지 않고, 현재 가장 좋은 값을 선택` 합니다. 그 순간의 최적을 선택하므로, `최종적인 결괏값이 최적이 아닐 수 있습니다.`
이러한 이유로 그리디 알고리즘을 근사 알고리즘이라고 합니다. 모든 경우를 계산하는 동적 계획법을 사용하기에는 시간이 부족할 때 그리디 알고리즘을 고려할 수 있습니다.

`분할 정복` 은 문제를 여러 하위 문제로 나누고, 각각을 계산한 뒤 합칩니다. 동적 계획법은 최적화를 위해 계산값을 기억해 두는데, 분할 정복은 그런 `최적화에 포커스를 맞춘 게 아니라,`
그저 `나눈다 > 푼다 > 합친다` 입니다. 그리고 분할 정복은 보통 하위 문제에서 중복되는 계산이 없습니다.

## 정리

동적 계획법은 큰 문제를 작은 부분 문제로 나누어 푸는 기법입니다. 같은 문제를 두 번 풀면 시간이 아까우니 한 번 푼 문제는 기억해 뒀다가 동일한 문제가 나올 때 그 값을 사용하는 방식으로 동작합니다.

## DP 연관 문제

- [0-1 배낭 문제](https://github.com/haeseong123/algorithm/blob/main/dp/01_knapsack/01_knapsack.md)
- [연쇄 행렬 곱셈](https://github.com/haeseong123/algorithm/blob/main/dp/chained_matrix_multiplication/chained_matrix_multiplication.md)
- [최장 공통 부분 수열: Longest Common Subsequence, LCS](https://github.com/haeseong123/algorithm/blob/main/dp/lcs/lcs.md)
- [최장 증가 부분 수열: Longest Increasing Subsequence, LIS](https://github.com/haeseong123/algorithm/blob/main/dp/lis/lis.md)

## 출처

- [동적 계획법](https://ko.wikipedia.org/wiki/%EB%8F%99%EC%A0%81_%EA%B3%84%ED%9A%8D%EB%B2%95)
- [Dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming)
- [Overlapping subproblems](https://en.wikipedia.org/wiki/Overlapping_subproblems)
- [Optimal substructure](https://en.wikipedia.org/wiki/Optimal_substructure)
- [최적 부분 구조](https://namu.wiki/w/%EA%B7%B8%EB%A6%AC%EB%94%94%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98#s-2.1)
- [알고리즘 - Dynamic Programming(동적 계획법)](https://hongjw1938.tistory.com/47)
