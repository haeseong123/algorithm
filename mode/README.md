# 최빈값

최빈값은 주어진 데이터 집합에서 가장 빈번하게 나타나는 값을 의미합니다.
데이터 집합에서 각 값의 빈도(출현 빈도)를 계산하여 가장 높은 빈도를 가진 값을 선택하는 것이 최빈값을 구하는 방법입니다.
최빈값은 데이터의 분포와 중심 경향성을 파악하는 데에 유용하게 사용될 수 있습니다.
예를 들어, 시험 점수 데이터에서 가장 많이 나타나는 점수를 구하거나, 소비자 조사 데이터에서 가장 많이 언급된 제품을 파악하는 등 다양한 분야에서 활용될 수 있습니다.

## 최빈값 구하는 방법

### 해시

동일한 값을 넣으면 항상 동일한 다이제스트를 반환하는 ***해시 함수*** 를 사용하여
주어진 배열에서 최빈값을 구하는 방법이 있습니다.

아이디어는 해시의 key를 `최빈값`으로 사용하여 value를 `빈도수`로 사용하는 것입니다.

다음과 같은 절차를 따릅니다.

1. 입력 배열(array)를 순회하며 각 원소의 빈도를 해시 테이블(또는 딕셔너리)에 저장한다.
    1. 해시 테이블의 키(Key)는 배열의 원소이고, 값(Value)은 해당 원소의 빈도를 나타낸다.
    2. 입력 배열의 원소를 하나씩 읽으면서 해당 원소를 해시 테이블에 키로 추가하고, 해당 키의 값이 이미 존재하면 빈도를 1 증가시킨다.
2. 해시 테이블에서 가장 빈도가 높은 원소를 찾는다.
    1. 해시 테이블을 순회하면서 빈도가 가장 높은 원소(키-값 쌍)를 찾는다.
    2. 만약 최빈값이 여러개라면 None으로 표현한다.
3. 최빈값을 반환한다.

이 방법은 해시 테이블을 이용하여 각 원소의 빈도를 효율적으로 저장하고 관리하여 최빈값을 빠르게 찾아내는 방법입니다.

해시 테이블을 사용하므로 입력 배열의 크기에 관계없이 일정한 성능을 유지할 수 있습니다.

```python
def get_mode(array):
    hash_table = {}

    # 1.
    for num in array:
        if num in hash_table:
            hash_table[num] += 1  # 이미 존재하는 키의 값(빈도)을 1 증가
        else:
            hash_table[num] = 1  # 새로운 키를 추가하고 값(빈도)을 1로 설정

    max_count = 0  # 최빈값의 빈도를 저장할 변수
    mode = None  # 최빈값을 저장할 변수

    # 2.
    for num, count in hash_table.items():
        if count > max_count:  # 현재 원소의 빈도가 최빈값의 빈도보다 크면
            max_count = count  # 최빈값의 빈도를 갱신
            mode = num  # 최빈값을 현재 원소로 업데이트
        elif count == max_count:  # 현재 원소의 빈도가 최빈값의 빈도와 같으면
            mode = None  # 최빈값이 여러 개이므로 None으로 표시

    # 3.
    return mode  # 최빈값 반환
```

다양한 프로그래밍 언어에서는 이미 구현된 해시 테이블 라이브러리가 제공되어 있으므로, 해당 라이브러리를 활용하여 간편하게 최빈값을 구할 수 있습니다.
예를 들어 파이썬에서는 내장 라이브러리인 collections 모듈의 Counter 클래스를 사용하면 간단하게 최빈값을 구할 수 있습니다.

```python
from collections import Counter


def get_mode(array):
    counter = Counter(array)  # 입력 배열(array)의 원소를 카운팅하여 Counter 객체 생성
    mode = counter.most_common(1)  # 가장 빈도가 높은 원소(키-값 쌍)을 가져옴

    if len(mode) > 0:
        return mode[0][0]  # 최빈값(키) 반환
    else:
        return None  # 최빈값이 없으면 None 반환
```

해시를 사용하여 최빈값을 구하는 알고리즘의 시간복잡도는 입력 배열을 순회하며 O(N),
해시 테이블을 순회하며 최악의 경우 O(N) 이므로 O(N) + O(N) = O(N) 입니다.

## 실전 문제

### 문제 설명

최빈값은 주어진 값 중 가장 자주 나오는 값을 의미합니다. 정수 배열 `array`가 매개변수로 주어질 때,
최빈값을 retrun하도록 solution 함수를 완성해보세요. 최빈값이 여러 개면 -1을 return 합니다.

### 제한사항

- 0 < `array`의 길이 < 100
- 0 <= `array`의 원소 < 1000

### 입출력 예

| array              | result |
|--------------------|--------|
| [1, 2, 3, 3, 3, 4] | 3      |
| [1, 1, 2, 2]       | -1     |
| [1] | 1      |

### 입출력 예 설명

입출력 예 #1

- [1, 2, 3, 3, 3, 4]에서 1은 1개 2는 1개 3은 3개 4는 1개로 최빈값은 3입니다.

입출력 예 #2

- [1, 1, 2, 2]에서 1은 2개 2는 2개로 최빈값이 1, 2입니다. 최빈값이 여러 개이므로 -1을 return 합니다.

입출력 예 #3

- [1]에는 1만 있으므로 최빈값은 1입니다.

### 문제 풀이 접근 방식 1

주어진 array의 최빈값을 return하는 것이므로 딕셔너리를 사용하여 문제를 풀 수 있을것 같습니다.

### 문제 풀이 1

```python
from collections import defaultdict


def get_mode_hash(array):
    hash_table = defaultdict(int)  # value의 기본값이 0인 hash_table을 생성합니다.
    max_count = 0  # 빈도수를 나타냅니다.
    mode = -1  # 최빈값을 저장합니다.

    # 1. 입력 배열(array)를 순회하며 각 원소의 빈도를 해시 테이블(또는 딕셔너리)에 저장합니다.
    for key in array:
        hash_table[key] += 1

    # 2. 해시 테이블에서 가장 빈도가 높은 원소를 찾습니다.
    for key, value in hash_table.items():

        # 현재 최빈값의 빈도수보다 더 큰 빈도수를 갖는 원소를 찾으면..
        if max_count < value:

            # 최빈값과 빈도수를 갱신합니다.
            mode, max_count = key, value

        # 현재 최빈값의 빈도수와 같은 빈도수를 갖는 원소를 찾으면..
        elif max_count == value:

            # 최빈값을 -1로 바꿉니다. (문제 설명에 -1로 하라고 나와있습니다.)
            mode = -1

    # 3. 최빈값을 반환합니다.
    return mode

def solution(array):
    return get_mode_hash(array)
```

### 문제 풀이 접근 방식 2

`array`의 원소의 크기가 정해져 있으므로 (0이상 1000미만) 배열을 사용하여 풀 수도 있을 것 같습니다.
아이디어는 `array`의 원소를 index로 사용하여 카운트하는 것입니다.

1. new_array를 만듭니다.
    1. `array`의 원소의 크기가 0이상 1000미만이므로 총 길이 1000의 new_array를 만듭니다.
2. `array`를 순회하며 각 원소의 빈도를 카운팅합니다.
3. new_array를 순회하며 가장 빈도가 높은 원소를 찾습니다.
4. 최빈값을 반환합니다.

이 방법은 `array`가 많은 값을 담고있든 중복되는 값을 많이 담고있든 길이가 1000인 new_array를 생성하므로
해시에 비해 시간적으로나 공간적으로나 비효율적으로 보입니다. 그래도 작성해봅시다.

### 문제 풀이 2

```python
def get_mode_array(array):
    # 1. 새 어레이(new_array)를 만듭니다.
    # 문제의 제한사항에서 원소의 길이가 0이상 1000미만이므로 총 길이 1000의 배열을 생성합니다.
    new_array = [0 for _ in range(1000)]
    max_count = 0  # 빈도수를 나타냅니다.
    mode = -1  # 최빈값을 저장합니다.

    # 2. 입력 배열(array)을 순회하며 각 원소의 빈도를 카운팅합니다.
    for index in array:
        new_array[index] += 1

    # 3. 새 어레이(new_array)를 순회하며 가장 빈도가 높은 원소를 찾습니다.
    for (index, value) in enumerate(new_array):

        # 현재 최빈값의 빈도수보다 더 큰 빈도수를 갖는 원소를 찾으면..
        if max_count < value:

            # 최빈값과 빈도수를 갱신합니다.
            mode, max_count = index, value

        # 현재 최빈값의 빈도수와 같은 빈도수를 갖는 원소를 찾으면..
        elif max_count == value:

            # 최빈값을 -1로 바꿉니다. (문제 설명에 -1로 하라고 나와있습니다.)
            mode = -1

    # 4. 최빈값을 반환합니다.
    return mode

def solution(array):
    return get_mode_array(array)
```

## 문제 출처

[최빈값 구하기](https://school.programmers.co.kr/learn/courses/30/lessons/120812)
