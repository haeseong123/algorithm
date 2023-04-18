# 순열과 조합

순열(Permutation)과 조합(Combination)은 수학과 컴퓨터 과학에서 사용되는 용어로, 집합에서 원소들을 조합하여 새로운 집합을 만들거나, 원소들을 순서에 따라 배열하는 것을 말합니다.

순열(Permutation)은 원소들의 **순서에 의미**를 두어, 순서에 따라 배열하는 것을 말합니다. 예를 들어, 3개의 원소 A, B, C를 가지고 만들 수 있는 순열은 AB, BA, AC, CA, BC, CA
등 6개의 경우가 있습니다.

조합(Combination)은 원소들의 **순서가 중요하지 않고**, 순서와 무관하게 조합을 만들어 내는 것을 말합니다. 예를 들어, 3개의 원소 A, B, C를 가지고 만들 수 있는 조합은 AB, AC, BC 총
3개의 경우 가 있습니다. 순서가 달라져도(예를 들어 BA, CA, CB) 동일한 것으로 간주합니다.

순열과 조합은 수학, 통계, 알고리즘, 데이터 처리 등 다양한 분야에서 활용되며, 원하는 결과에 따라 적절하게 사용될 수 있습니다.

순열과 조합에 대해 더 자세히 알아봅시다.

## 순열

순열(Permutation)은 서로 다른 n 개 중 r 개를 골라 순서를 고려해 나열한 경우의 수 입니다. 이는 일반적으로 다음과 같이 표현됩니다.

`nPr` 또는 `P(n, r)`

예를 들어 조별 활동을 위해 같은 조가 된 a, b, c, d, e 다섯 명의 학생 중 조장과 부조장을 뽑으려 할 때 조장과 부조장을 뽑는 데 모두 몇 가지 경우의 수가 있을지 생각해 봅시다.

먼저 다섯 명 중 한 명을 조장으로 뽑으므로, 조장이 될 수 있는 경우의 수는 5입니다. 그러고 나서 조장이 된 한 명을 제외한 나머지 네 명 중 한 명을 부조장으로 뽑으면 이때의 경우의 수는 4입니다. 그러므로
조장을 뽑고 부조장을 뽑는 경우의 수는 20(= 5*4)가지입니다.

위 문제는 5개 중 2개를 골라 나열해야 하는 상황이므로 P(5, 2)로 표현합니다.

### 순열의 식

n개의 서로 다른 원소를 가진 집합에서 r개의 서로 다른 대상들을 선택하여 배열하는 가능한 모든 경우를 생각해 봅시다.

먼저, 첫 번째 원소는 n가지 방법으로 선택할 수 있습니다. 두 번째 원소는 첫 번째 선택된 원소와 달라야 하므로
(n-1) 가지 방법으로 선택할 수 있습니다. 나머지 원소들을 선택하는 방법의 수도 같은 방식으로 결정되며, 마지막 r번째 원소는 (n-(r-1))가지 방법으로 선택할 수 있습니다. 즉 아래와 같이 쓸 수 있습니다.

![p](https://user-images.githubusercontent.com/50406129/232701396-9ddc1dfb-f593-4941-a53d-db8ffd5c49ca.PNG)

### 순열의 구현

파이썬을 사용하여 순열을 구현해 봅시다.
[재귀 함수](https://github.com/haeseong123/algorithm/blob/main/brute_force/recursive/recursive.md)
를 사용하여 구현할 수 있고 반복문만으로 구현할 수도 있습니다. 우선 재귀 함수를 사용한 구현 방법에 대해 배워봅시다.

#### 재귀 함수로 구현

[A, B, C]에서 두 개의 서로 다른 대상들을 선택한다고 생각해 봅시다.

1. 첫 번째 선택에 `A` 를 선택했으면 우리는 `[B, C]`만 가지고 나머지 선택을 이어가야 하고
2. 두 번째 선택에 `B` 를 선택했으면 우리는 `[C]`만 가지고 나머지 선택을 이어가야 합니다.
3. 세 번째 선택은 하지 않아도 됩니다. 왜냐하면 현재 문제가 P(3, 2)이기 때문입니다.

이렇듯 순열은 선택에 올 수 있는 원소들을 제한해야 합니다.

실제 순열을 만드는 코드를 작성해 봅시다. 순열을 만드는 재귀 함수는 아래와 같은 순서로 동작합니다.

1. arr과 r을 받는 permutations 함수를 정의합니다.
2. start를 받는 backtrack 함수를 permutations 내부에 정의합니다.
    1. start와 r을 비교합니다. 만약 같다면 result.append(arr[:r]) 하고 return합니다.
    2. start부터 arr의 길이까지 반복합니다.
       (이는 지금까지 선택했던 원소를 제외한 모든 원소를 고려하기 위함입니다.)
        1. arr[start], arr[i] = arr[i], arr[start]
        2. 선택된 원소를 올바른 자리에 위치 시켜 다음 선택 시 고려 대상에서 제거하고 arr[:r]을 올바르게 동작하도록 만드는 코드입니다.
        3. backtrack(start + 1)
            1. 다음 원소를 선택하기 위해 start + 1을 매개변수로 넘겨 backtrack을 호출합니다.
        4. arr[start], arr[i] = arr[i], arr[start]
            1. 재귀 호출이 끝나면 원소의 자리를 원상태로 복구하기 위해 start 위치의 원소와 i 위치의 원소를 다시 교환합니다.
4. result를 만듭니다.
5. backtrack(0)을 호출합니다.
    1. 첫 번째 선택을 해야 하므로 0을 매개변수로 넣어 backtrack을 호출합니다.
6. result를 반환합니다.

이를 코드로 나타내면 아래와 같습니다.

```python
def permutations(arr, r):
    def backtrack(start):
        if start == r:
            result.append(arr[:r])
            return
        for i in range(start, len(arr)):
            arr[start], arr[i] = arr[i], arr[start]
            backtrack(start + 1)
            arr[start], arr[i] = arr[i], arr[start]

    result = []
    backtrack(0)
    return result
```

start는 지금 선택이 몇 번째 선택인지를 나타내기 위한 변수입니다. 예를 들어 start가 0이면 첫 번째 선택을 해야 할 차례이고 1이면 두 번째 선택을 해야 할 차례입니다. 위에서도 언급했지만, start를
사용하는 이유는 이번에 선택된 원소를 다음 선택 고려 대상에서 제외하기 위함입니다. 그 외의 것들은 사실상 자리 바꾸는 게 전부입니다.

#### 반복문으로 구현

아무래도 재귀 함수를 사용하면 코드가 간결해지는 장점이 있으니, 재귀 함수 없이 반복문만으로 구현하게 되면 코드가 더 복잡해지는 단점이 있습니다. 그렇지만 반복문은 일반적으로 스택에 재귀 호출을 쌓아두는 재귀 함수와
달리 변수의 값을 갱신하는 방식으로 동작하기 때문에 재귀 함수보다 메모리 사용량이 더 낮은 장점이 있습니다.

코드 먼저 보고 해당 코드를 해석하는 방식으로 배워봅시다. yield 문법을 모르시면 해당 [포스트](https://www.daleseo.com/python-yield/) 를 보고 오시면 좋습니다.

```python
def permutations(iterable, r=None):
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210

    pool = tuple(iterable)  # 1
    n = len(pool)  # 2
    r = n if r is None else r  # 3

    if r > n:  # 4
        return

    indices = list(range(n))  # 5
    cycles = list(range(n, n - r, -1))  # 6

    yield tuple(pool[i] for i in indices[:r])  # 7

    while n:  # 8
        for i in reversed(range(r)):  # 9
            cycles[i] -= 1  # 10

            if cycles[i] == 0:  # 11
                indices[i:] = indices[i + 1:] + indices[i:i + 1]  # 12
                cycles[i] = n - i  # 13
            else:  # 14
                j = cycles[i]  # 15
                indices[i], indices[-j] = indices[-j], indices[i]  # 16
                yield tuple(pool[i] for i in indices[:r])  # 17
                break  # 18

        else:  # 19
            return  # 20
```

1번부터 20번까지 하나하나 살펴봅시다.

1. `iterable`을 `tuple` 형태로 `pool` 변수에 저장합니다. `iterable`은 순열을 생성할 원본 요소들이 들어있는 iterable입니다.
2. `n`에 `pool`의 길이를 저장합니다. `n`은 원본 요소들의 개수를 나타냅니다.
3. `r`이 주어지지 않았을 경우 `r`을 `n`으로 설정하고 그렇지 않은 경우 전달받은 `r`을 그대로 사용합니다.
4. `r`이 `n`보다 큰 경우, 즉 순열의 길이가 원본 요소들의 개수보다 큰 경우, 빈 값을 반환하여 함수를 종료합니다.
5. `0`부터 `n-1`까지의 숫자를 리스트 형태로 `indices` 변수에 저장합니다. 이는 순열의 인덱스를 나타냅니다.
6. `n`부터 `n-r+1` 까지의 숫자를 역순으로 리스트 형태로 `cycles` 변수에 저장합니다. 이는 순열의 주기를 나타냅니다.
7. `pool`의 요소 중에서 `indices` 리스트의 처음부터 `r` 미만까지의 인덱스에 해당하는 요소들로 이루어진 튜플을 `yield`하여 반환합니다. 이는 순열의 초깃값을 생성하는 부분입니다.
8. 이후의 동작을 `n`이 0이 될 때까지 반복합니다.
9. `r`부터 `0`까지의 역순으로 반복합니다. `i`는 현재 순열의 주기를 나타냅니다.
10. `cycles` 리스트의 `i` 번째 요소를 1 감소시킵니다. 이는 순열의 주기를 나타내는 변수 `cycles`의 값을 갱신하는 부분입니다.
11. `cycles` 리스트의 `i` 번째 요소가 0이라면, 즉 현재 순열의 주기가 완료되었다면 12~13 코드를 실행합니다. 그렇지 않다면 14 코드를 실행합니다.
12. `indices` 리스트의 `i` 번째 요소부터 끝까지를 `indices` 리스트의 `i+1` 번째 요소부터 끝까지와
    `indices` 리스트의 `i` 번째 요소로 교체합니다. 이는 순열의 주기가 완료된 경우 해당 인덱스의 요소를 맨 뒤로 이동하는 역할을 합니다.
13. `cycles` 리스트의 `i` 번째 요소를 `n - i`로 갱신합니다. 이는 새로운 주기를 나타내기 위한 값으로, 순열의 길이에서 현재 인덱스 `i`를 뺀 값입니다. ??
14. 만약 현재 순열의 주기가 완료되지 않은 경우, 위의 `if` 조건에 해당하지 않는 경우 아래의 코드를 실행합니다.
15. `cycles` 리스트의 `i` 번째 요소를 `j` 변수에 저장합니다. 이는 현재 주기의 길이를 나타냅니다.
16. `indices[-j]` 리스트의 `i` 번째 요소와 `indices` 리스트의 뒤에서 `j` 번째 요소를 서로 교환합니다. 이는 현재 주기에 해당하는 두 개의 인덱스를 서로 바꾸는 역할을 합니다.
17. `pool`의 요소 중에서 `indices` 리스트의 처음부터 `r-1` 까지의 인덱스에 해당하는 요소들로 이루어진 튜플을 `yield`하여 반환합니다. 이는 새로운 순열의 값을 생성하는 부분입니다.
18. `for` 반복문을 중단합니다. 새로운 순열의 값을 생성하였으므로, 다음 순열을 생성하기 위해
    `for` 반복문의 다음 순회를 시작합니다.
19. `for` 반복문이 정상적으로 종료된 경우 해당 `else` 아래의 문장이 실행됩니다. 이는 더 이상 생성할 순열이 없는 경우 함수를 종료하기 위한 부분입니다.
20. 함수를 종료합니다.

> 반복문으로 구현 공부 필요

#### 모듈 사용

python에서는 itertools에서 제공하는 permutations 함수를 사용하여 간단하게 순열을 얻을 수도 있습니다.

내부 구현은 위에 있는 **반복문으로 구현**과 비슷하다고 합니다.

```python
from itertools import permutations

permutations(['A', 'B', 'C'], 2)
```

마지막으로 순열을 사용하여 실제 문제 두 개를 풀어봅시다.

### 소수 찾기

한 자리 숫자가 적힌 종잇조각이 흩어져 있습니다. 흩어진 종잇조각을 붙여 소수를 몇 개 만들 수 있는지 알아내려 합니다.

각 종잇조각에 적힌 숫자가 적힌 문자열 numbers가 주어졌을 때, 종잇조각으로 만들 수 있는 소수가 몇 개인지 return 하도록 solution 함수를 완성해 주세요.

#### 제한 사항

- numbers는 길이 1 이상 7 이하인 문자열입니다.
- numbers는 0~9까지 숫자만으로 이루어져 있습니다.
- "013"은 0, 1, 3 숫자가 적힌 종잇조각이 흩어져 있다는 의미입니다.

#### 입출력 예

|numbers|return|
|------|--------|
|"17"|3|
|"011"|2|

#### 입출력 예 설명

예제 #1: `[1, 7]`으로는 소수 `[7, 17, 71]`를 만들 수 있습니다. 예제 #1: `[0, 1, 1]`으로는 소수 `[11, 101]`를 만들 수 있습니다.

#### 문제 풀이

주어진 numbers를 가지고 P(numbers, 1), P(numbers, 2), ..., P(numbers, len(numbers))
와 같이 모든 경우의 순열을 생성하며 각 결과의 원소를 꺼내서 해당 숫자가 소수인지 판별하면 될 것 같습니다.

이를 순서로 나타내면 아래와 같습니다.

1. `answer` 변수를 생성하고 0을 넣습니다.
2. 주어진 `numbers`를 리스트로 변환합니다.
3. nPr에서 모든 r에 대한 순열을 생성하도록 1부터 len(numbers)까지 반복합니다.
4. i를 넣어 `permutations(numbers, i)`를 호출합니다.
5. 생성된 순열의 요소가 소수인지 확인합니다. 만약 소수라면 `answer += 1`을 합니다.
6. `answer`를 반환합니다.

아래는 이를 실제로 구현한 코드입니다.

```python
# 소수를 판별하는 함수입니다.
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def permutations_recursion(arr, r):
    # 중복을 피하기 위해 set을 사용합니다.
    # ex) "011"의 경우 P("011", 2) 일때
    # 첫 번째와 두 번째를 선택해도 01 이고
    # 첫 번째와 세 번째를 선택해도 01 입니다.
    # 이러한 경우를 방지하기 위해 result를 set으로 선언합니다.
    result = set()
    n = len(arr)

    def backtrack(start):
        if start == r:
            # 맨 앞자리가 0이면 숫자로 바꿀 때 자릿수가 
            # 줄어드므로 result에 추가하지 않습니다.
            if arr[0] != '0':
                result.add(int(''.join(arr[:r])))
            return
        for i in range(start, n):
            arr[start], arr[i] = arr[i], arr[start]  # 자리 바꾸기
            backtrack(start + 1)
            arr[start], arr[i] = arr[i], arr[start]  # 자리 원위치

    backtrack(0)
    return result


def solution(numbers):
    answer = 0
    numbers = list(numbers)

    # nPr에서 모든 r에대한 순열을 생성하도록 1부터 len(numbers)까지 반복합니다.
    for i in range(1, len(numbers) + 1):
        # 순열을 생성합니다. (P(numbers, 1), P(numbers, 2), ..., P(numbers, n))
        for p in permutations_recursion(numbers, i):
            # 생성된 순열이 소수인지 판별합니다.
            if is_prime(p):
                answer += 1

    return answer
```

### 모음 사전

사전에 알파벳 모음 'A', 'E', 'I', 'O', 'U'만을 사용하여 만들 수 있는, 길이 5 이하의 모든 단어가 수록되어 있습니다. 사전에서 첫 번째 단어는 "A"이고, 그다음은 "AA"이며, 마지막
단어는 "UUUUU"입니다.

단어 하나 word가 매개변수로 주어질 때, 이 단어가 사전에서 몇 번째 단어인지 return 하도록 solution 함수를 완성해 주세요.

#### 제한 사항

- word의 길이는 1 이상 5 이하입니다.
- word는 알파벳 대문자 'A', 'E', 'I', 'O', 'U'로만 이루어져 있습니다.

#### 입출력 예

|word|result|
|------|--------|
|"AAAAE"|6|
|"AAAE"|10|
|"I"|1563|
|"EIO"|1189|

#### 입출력 예 설명

입출력 예 #1: 사전에서 첫 번째 단어는 "A"이고, 그다음은 "AA",
"AAA", "AAAA", "AAAAA", "AAAAE", ... 와 같습니다. "AAAAE"는 사전에서 6번째 단어입니다.

입출력 예 #2: "AAAE"는 "A", "AA", "AAA", "AAAA", "AAAAA", "AAAAE",
"AAAAI", "AAAAO", "AAAAU"의 다음인 10번째 단어입니다.

입출력 예 #3: "I"는 1563번째 단어입니다.

입출력 예 #4: "EIO"는 1189번째 단어입니다.

#### 문제 풀이

['A', 'E', 'I', 'O', 'U']만으로 최대 길이가 5인 모든 단어가 들어있는 단어집을 만들고 해당 단어집에서 word가 몇 번째 인덱스에 존재하는지 확인하면 될 것 같습니다.

이를 순서로 나타내면 아래와 같습니다.

1. `alphabet` 변수를 생성하고 `['A', 'E', 'I', 'O', 'U']` 를 저장합니다.
2. `alphabet`를 n으로 사용하여 P(n, 1)부터 P(n, 5)까지의 **중복을 허용**하는 순열을 만들고 `dictionary` 에 저장합니다.
3. `dictionary` 를 사전순으로 정렬합니다.
4. `dictionary` 에서 `word` 의 인덱스를 찾고 +1을 하여 반환합니다.

아래는 이를 실제로 구현한 코드입니다.

```python
def permutations_allowed_duplicates(arr, r, dictionary):
    n = len(arr)

    def backtrack(word):
        if len(word) == r:
            # dictionary에 추가합니다.
            dictionary.append(''.join(word))
            return
        for i in range(n):
            word.append(arr[i])
            backtrack(word)
            # dictionary에 추가했으니 제외하는 코드입니다.
            word.pop()

    backtrack([])


def solution(word):
    alphabet = ['A', 'E', 'I', 'O', 'U']
    n = len(alphabet)
    dictionary = []

    # nPr에서 모든 r에 대한 순열을 생성하도록 1부터 n까지 반복합니다.
    for i in range(1, n + 1):
        # 순열을 생성합니다.
        permutations_allowed_duplicates(alphabet, i, dictionary)

    # dictionary를 사전순으로 정렬합니다.
    dictionary.sort()

    return dictionary.index(word) + 1
```

## 조합

서로 다른 물건 중 몇 가지 대상을 뽑는 것을 조합(Combination)이라고 합니다. 이때 서로 다른 n 개 중 **순서를 고려하지 않고** r 개를 고르는 방법의 수를 아래와 같이 표현합니다.

`nCr` 또는 `C(n, r)`

예를 들어 [1, 2, 3]이라는 원소들이 주어졌을 때 이들로부터 2개를 선택하여 조합을 구한다면
(C(3, 2)) 가능한 조합은 다음과 같이 3가지가 있습니다.

- [1, 2]
- [1, 3]
- [2, 3]

조합은 원소의 순서가 중요하지 않기 때문에 [1, 2]와 [2, 1]은 동일한 조합으로 간주합니다.

### 조합의 식

[A, B, C, D, E]에서 두 개의 서로 다른 대상들을 선택하여 배열하는 가능한 모든 경우를 생각해 봅시다.

조합의 식은 순열로부터 생각하면 쉬우므로 순열을 먼저 구해봅시다. 순열의 경우 순서를 고려하므로 `P(5, 2) = 5!/(5-2)!` 입니다. 조합의 경우 순서를 고려하지 않으므로 위 순열의 결과에서 동일한 원소를
결과로 하는 순열들을 지우면 그것이 조합입니다. 예를 들어 순열에 [A, B]와 [B, A]가 있다면 조합에서는 이 둘이 같은 것이므로 [A, B]를 지우거나 [B, A]를 지우면 조합이 되는 것입니다.

따라서 `C(5, 2) = P(5, 2)/2!` 이고, 이를 일반화하면 아래와 같습니다.

![c](https://user-images.githubusercontent.com/50406129/232777756-ccfda2b1-e3e8-45ea-9a18-44f15f2fc30d.PNG)

조금 다르게 말하면 P(n, r)는 먼저 C(n, r)가지 방법으로 r개의 원소를 가진 부분집합을 고른 후, 선택된 r개의 원소를 r! 가지 방법으로 배열하여 구할 수 있습니다. 따라서
`P(n, r) = C(n, r)r!`
이므로 `C(n, r) = P(n,r)/r! = n!/r!(n-r)!` 입니다.

### 조합의 구현

파이썬을 사용하여 조합을 구현해 봅시다. 순열과 마찬가지로 재귀 함수를 사용하여 구현할 수 있고 반복문만으로 구현할 수도 있습니다. 우선 재귀 함수를 사용한 구현 방법에 대해 배워봅시다.

#### 재귀 함수로 구현

조합은 순열과 달리 순서를 구분하지 않으므로 순열보다 더욱 간단하게 구현할 수 있습니다.

조합을 만드는 재귀 함수는 아래와 같은 순서로 동작합니다.

1. arr과 r을 받는 combinations 함수를 정의합니다.
2. start와 comb를 받는 backtrack 함수를 combinations 내부에 정의합니다.
    1. len(comb)와 r을 비교합니다. 만약 같다면 result.append(comb) 하고 return합니다.
    2. start부터 arr의 길이까지 반복합니다.
       (이는 지금까지 선택했던 원소를 제외한 모든 원소를 고려하기 위함입니다.)
        1. comb.append(arr[i])로 현재 요소를 삽입합니다.
        2. backtrack(i+1, comb)를 호출합니다.
        3. comb.pop()으로 현재 요소를 삭제합니다.
3. result를 만듭니다.
4. backtrack(0) 을 호출합니다.
    1. 첫 번째 선택을 해야 하므로 0을 매개변수로 넣어 backtrack을 호출합니다.
5. result를 반환합니다.

실제 코드로 구현해 봅시다.

```python
def combinations(arr, r):
    def backtrack(start, comb):
        if len(comb) == r:
            result.append(comb[:])
            return
        for i in range(start, n):
            comb.append(arr[i])
            backtrack(i + 1, comb)
            comb.pop()

    n = len(arr)
    result = []
    backtrack(0, [])
    return result
```

자리를 바꾸지 않아도 되므로 순열에 비해 간단합니다. 이제 반복문을 사용한 구현을 배워봅시다.

#### 반복문으로 구현

순열에서 했던 것과 마찬가지로 코드 먼저 보고 해당 코드를 해석하는 방식으로 배워봅시다.

```python
def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123

    pool = tuple(iterable)  # 1
    n = len(pool)  # 2

    if r > n:  # 3
        return

    indices = list(range(r))  # 4

    yield tuple(pool[i] for i in indices)  # 5

    while True:  # 6
        for i in reversed(range(r)):  # 7
            if indices[i] != i + n - r:
                break
        else:
            return

        indices[i] += 1  # 8

        for j in range(i + 1, r):  # 9
            indices[j] = indices[j - 1] + 1

        yield tuple(pool[i] for i in indices)  # 10
```

1번부터 10번까지 하나하나 살펴봅시다.

1. iterable을 tuple 형태로 변환하여 `pool` 변수에 저장합니다. iterable은 조합을 생성할 원본 요소들이 있는 iterable입니다.
2. `pool`의 길이를 `n` 변수에 저장합니다. 이는 원본 요소들의 개수를 나타냅니다.
3. `r`이 `n`보다 크다면 조합을 생성할 수 없으므로 함수를 종료합니다.
4. `r` 개의 원소로 이루어진 리스트를 생성하여 indices 변수에 저장합니다. 이 리스트는 조합의 인덱스를 나타냅니다. `0, 1, 2, ..., r-1`의 값을 가집니다.
5. 현재의 `indices` 리스트를 이용하여 `pool`의 원소들을 선택하여 조합을 생성하고, 이를 yield로 반환합니다.
6. 무한 루프를 시작합니다. 조합을 계속해서 생성하기 위해 사용합니다.
7. `indices` 리스트를 뒤에서부터 검사하면서 조합을 생성하는 데 필요한 인덱스를 찾습니다.
   `indices[i]`가 `i + n - r`과 같지 않다면 해당 인덱스를 찾은 것입니다.
8. 찾은 인덱스를 증가시킵니다.
9. 찾은 인덱스 이후의 인덱스들을 조정합니다. 인덱스 `i` 이후의 인덱스들은 `i` 번째 인덱스에 1을 더한 값으로 갱신됩니다.
10. 갱신된 `indices` 를 이용하여 `pool`의 원소들을 선택하여 조합을 생성하고, 이를 `yield`를 통해 반환합니다.

> 반복문으로 구현 공부 필요

#### 모듈 사용

python에서는 itertools에서 제공하는 combinations 함수를 사용하여 간단하게 순열을 얻을 수도 있습니다.

내부 구현은 위에 있는 **반복문으로 구현**과 비슷하다고 합니다.

```python
from itertools import combinations

combinations(['A', 'B', 'C'], 2)
```

마지막으로 조합을 사용하여 실제 문제를 풀어봅시다.

### 두 개 뽑아서 더하기

정수 배열 numbers가 주어집니다. numbers에서 서로 다른 인덱스에 있는 두 개의 수를 뽑아 더해서 만들 수 있는 모든 수를 배열에 오름차순으로 담아 return 하도록 solution 함수를 완성해
주세요.

#### 제한 사항

- numbers의 길이는 2 이상 100 이하입니다.
- numbers의 모든 수는 0 이상 100 이하입니다.

#### 입출력 예

|numbers|result|
|------|--------|
|[2, 1, 3, 4, 1]|[2, 3, 4, 5, 6, 7]|
|[5, 0, 2, 7]|[2, 5, 7, 9, 12]|

#### 입출력 예 설명

입출력 예 #1: `2 = 1+1` , `3 = 2+1` , `4 = 1+3` , `5 = 1+4 = 2+3`,
`6 = 2+4`, `7 = 3+4` 이므로 `[2, 3, 4, 5, 6, 7]` 을 반환해야 합니다.

입출력 예 #2: `2 = 0+2` , `5 = 5+0` , `7 = 0+7 = 5+2` ,
`9 = 2+7` , `12 = 5+7` 이므로 `[2, 5, 7, 9, 12]` 를 반환해야 합니다.

#### 문제 풀이

numbers 배열에서 서로 다른 두 개의 인자를 뽑아 만들 수 있는 합의 배열을 오름차순으로 정렬한 뒤 반환하면 됩니다.

이를 순서로 나타내면 아래와 같습니다.

1. `numbers` 로 C(numbers, 2)를 생성합니다. 생성된 값은 `answer` 에 저장합니다.
2. `answer` 를 오름차순으로 정렬합니다.
3. `answer` 를 반환합니다.

아래는 이를 실제로 구현한 코드입니다.

```python
def combination_without_duplicates(arr, r):
    # 합이 중복되면 안되므로 result를 set으로 선언합니다.
    result = set()
    n = len(arr)

    def backtrack(start, comb, size):
        if size == r:
            result.add(sum(comb))
            return
        for i in range(start, n):
            comb.append(arr[i])
            # start + 1이 아니라 i + 1인 이유는
            # 순열과 달리 중복은 요소의 자리를 바꾸지 않기 때문입니다.
            backtrack(i + 1, comb, size + 1)
            comb.pop()

    backtrack(0, [], 0)
    return list(result)


def solution(numbers):
    # 조합을 사용하여 중복 없는 합을 리스트로 받아온 뒤
    # 정렬하고
    # 반환합니다.
    return sorted(combination_without_duplicates(numbers, 2))
```

## 출처

- 순열
    - [두산백과](https://terms.naver.com/entry.naver?docId=1223701&cid=40942&categoryId=32213)
    - [수학백과](https://terms.naver.com/entry.naver?docId=3405187&cid=47324&categoryId=47324)
    - [파이썬 공식 문서](https://docs.python.org/3/library/itertools.html#itertools.permutations)
    - [소수 찾기: 프로그래머스](https://school.programmers.co.kr/learn/courses/30/lessons/42839)
    - [모음 사전: 프로그래머스](https://school.programmers.co.kr/learn/courses/30/lessons/84512)
- 조합
    - [수학백과](https://terms.naver.com/entry.naver?docId=3405317&cid=47324&categoryId=47324)
    - [파이썬 공식 문서](https://docs.python.org/3/library/itertools.html#itertools.combinations)
    - [두 개 뽑아서 더하기: 프로그래머스](https://school.programmers.co.kr/learn/courses/30/lessons/68644)
