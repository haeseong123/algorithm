# 최장 증가 부분 수열 (Longest Increasing Subsequence, LIS)

최장 증가 부분 수열 문제는 주어진 수열은 가장 긴 증가하는 부분 수열을 구하는 문제입니다. 예를 들어 수열 `[4, 2, 1, 3, 5, 8, 6, 7]` 이 주어졌을 때의 LIS는
`[2, 3, 5, 6, 7]` 입니다. 아래 사진을 참고해 주세요.

![a](https://user-images.githubusercontent.com/50406129/233972567-9d831944-c1df-4582-91d1-46d1afe2400a.png)
![b](https://user-images.githubusercontent.com/50406129/233972585-85739170-56f6-4355-9b9e-8616a58e00df.png)

실제 LIS 문제인 백준 `가장 긴 증가하는 부분 수열` 문제를 풀어봅시다.

## 가장 긴 증가하는 부분 수열

수열 A가 주어졌을 때, 가장 긴 증가하는 부분 수열을 구하는 프로그램을 작성하시오.

예를 들어, 수열 `A = {10, 20, 10, 30, 20, 50}` 인 경우에 가장 긴 증가하는 부분 수열은 `{10, 20, 30, 50}` 이고 길이는 4입니다.

### 입력

첫째 줄에 수열 A의 크기 N(1 <= N <= 1,000)이 주어집니다.

둘째 줄에는 수열 A를 이루고 있는 원소가 차례대로 주어집니다. (1 <= 원소 <= 1,000)

### 출력

첫째 줄에 수열 A의 가장 긴 증가하는 부분 수열의 길이를 출력해 주세요.

### 예제

입력

```
6
10 20 10 30 20 50
```

출력

```
4
```

### 문제 풀이

문제를 보면 딱 떠오르는 생각은 이중 for문을 돌려서 문제를 해결하는 방법입니다.

이는 다음과 같습니다.

1. dp 배열을 생성합니다.
2. i=0부터 dp[i]를 계산합니다. 각각의 계산을 할 때마다 앞에 위치한 수들을 모두 확인하여 자신보다 작은 숫자들의 D값 중 최댓값을 찾고 1을 더해 D[i]를 계산합니다.
3. max(dp)로 최장 증가 부분 수열의 길이를 찾습니다.

코드로 나타내면 아래와 같습니다.

```python
n = int(input())
arr = list(map(int, input().split()))
dp = [1] * n

for i in range(n):
    for j in range(i):
        if arr[j] < arr[i]:
            dp[i] = max(dp[i], dp[j] + 1)

print(max(dp))
```

이 방법은 내 앞에 있는 모든 값들을 조회하므로 `O(N * N) = O(N^2)` 의 시간 복잡도를 갖습니다. 만약 모든 값을 조회하지 않고 이분 탐색을 사용한다면 `O(N * logN)` 의 시간 복잡도를 갖게
될 것입니다.

이진 탐색을 이용한 LIS 알고리즘의 동작 방식은 다음과 같습니다.

1. dp 배열을 만듭니다.
2. 수열을 순회하면서 dp에 삽입될 수 있는 위치를 이진 탐색으로 찾습니다.
    1. dp의 마지막 원소보다 큰 경우, dp의 마지막에 삽입합니다.
    2. 그렇지 않으면, dp에서 이진 탐색을 통해 삽입될 위치를 찾아 해당 위치에 삽입합니다.
3. dp의 길이를 출력합니다.

이를 코드로 나타내면 아래와 같습니다.

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left < right:
        mid = (left + right) // 2

        if arr[mid] < target:
            left = mid + 1
        elif arr[mid] > target:
            right = mid
        else:
            return mid

    return right


n = int(input())
a = list(map(int, input().split()))
dp = [a[0]]

for item in a[1:]:
    if dp[-1] < item:
        dp.append(item)
    else:
        dp[binary_search(dp, item)] = item

print(len(dp))
```

## 출처

- [LIS (Longest Increasing Subsequence) - 최장 증가 부분 수열](https://rebro.kr/33)
- [[Algorithm] 최장 증가 부분 수열(LIS: Least Increasing Subsequence) 문제를 푸는 두 가지 알고리즘](https://cider.tistory.com/12)
