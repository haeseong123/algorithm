# 비트 마스크

비트 마스크(bitmask)는 컴퓨터 프로그래밍에서 **비트(bit) 단위로 데이터를 처리하는 기술** 중 하나로, 이진수(binary)의 **각 비트에 특정한 의미를 부여하여 사용하는 것**을 말합니다.

비트 마스크는 그 자체가 알고리즘이라기보다 OR, AND, XOR, NOT, SHIFT 등의 비트 연산자를 활용하여 비트 연산을 하여 문제를 해결하는 하나의 기술입니다. 비트 마스크는 간단한 비트 연산으로 작동하므로
메모리 사용량을 줄이고 연산 속도를 높이는 장점이 있어, 효율적이고 간결한 코드를 작성할 수 있습니다. 이러한 비트 마스크는 알고리즘, 데이터 구조, 시스템 프로그래밍, 네트워크 프로그래밍 등 다양한 컴퓨터
프로그래밍 분야에서 활용됩니다.

비트 마스크가 사용되는 구체적인 예시와 함께 비트 마스크에 대해 더 알아봅시다.

## 사용자 권한 관리

보안 시스템에서 사용자 권한을 관리하는 경우 비트 마스크가 유용하게 사용될 수 있습니다. 예를 들어, 사용자에게 읽기(read), 쓰기(write), 실행(execute)의 세 가지 권한이 있는 경우 각각의 권한에
대해 비트를 할당할 수 있습니다. 예를 들어, 3개의 비트를 사용하여 다음과 같이 표현할 수 있습니다.

- 읽기 - `001`
- 쓰기 - `010`
- 실행 - `100`

읽기 권한이 있으면 1번 비트에 1이 들어가고 읽기 권한이 없으면 1번 비트에 0이 들어가는 형식입니다. 예를 들어 아래와 같이 나타낼 수 있습니다.

- 읽기, 쓰기 권한이 있는 경우 - `011`
- 쓰기, 실행 권한이 있는 경우 - `110`
- 아무 권한이 없는 경우 - `000`

이렇게 비트에 특정 의미를 부여하는 것이 비트 마스크입니다. 또한 비트 마스크는 비트 연산을 사용하여 값(이 경우 권한)을 변경할 수도 있습니다.

우선 특정 사용자에게 권한을 부여하는 작업을 살펴봅시다. 아래는 사용자에게 읽기와 실행 권한을 부여하는 과정입니다.

1. 읽기 권한과 실행 권한을 나타내는 비트를 생성합니다. - `101`
2. 어떤 사용자의 권한 비트를 가져옵니다. (사용자는 현재 쓰기 권한만 존재 합니다) - `010`
3. 1과 2에 대해 OR 연산을 진행합니다. - `101 | 010 == 111`

101과 010(사용자의 권한 비트)에 대해 OR 연산을 진행하여 사용자의 권한 비트를 111로 변경했습니다. 이는 기존 쓰기 권한만 있던 사용자에게 읽기와 실행에 대한 권한을 추가로 부여한 것입니다.

그럼, 이번에는 반대로 권한을 제거하는 연산을 해봅시다.

1. 읽기 권한과 실행 권한을 0으로 나타내는 비트를 생성합니다. - `010`
2. 어떤 사용자의 권한 비트를 가져옵니다. (사용자는 현재 읽기, 쓰기, 실행 권한을 모두 갖고 있습니다) - `111`
3. 1과 2에 대해 AND 연산을 진행 - `010 & 111 == 010`

010과 111(사용자의 권한 비트)에 대해 AND 연산을 진행하여 사용자의 권한 비트를 010으로 변경했습니다. 이는 기존 읽기, 쓰기, 실행 권한을 모두 갖고 있던 사용자에게 읽기와 실행에 대한 권한을 제거한
것입니다.

비트 마스크에 대한 감을 익혔을 것입니다. 이제 비트 마스크를 사용하여 완전 탐색(Exhaustive Search) 문제를 풀어 봅시다.

## N개의 원소를 갖는 집합의 모든 부분집합

N개의 원소를 갖는 배열이 주어졌을 때, 배열의 원소로 만들 수 있는 부분집합을 출력하는 문제입니다. 단, 배열의 원소는 중복이 없습니다.

예를 들어 [1, 2] 배열이 주어지면 [], [1], [2], [1, 2]가 출력되어야 합니다. 이를 표로 나타내면 아래와 같습니다.

|arr|결과|
|------|--------|
|1|[], [1]|
|1, 2|[], [1], [2], [1, 2]|
|1, 2, 3|[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]|

이 문제는 완전 탐색 알고리즘으로 풀 수 있으며 비트 마스크를 사용할 수 있는 문제입니다.

1. 이 문제는 완전 탐색 알고리즘으로 풀 수 있습니다.
    1. 모든 경우의 수를 찾을 수 있고, 정답인지 아닌지 알 수 있습니다. (이 경우 모든 경우가 전부 정답)
2. 이 문제는 비트 마스크를 사용할 수 있습니다.
    1. 주어진 배열의 부분집합은 배열의 원소를 뽑거나/뽑지 않거나 하여 만든 집합입니다. 이는 선택/미선택의 의미로서 bit로 나타낼 수 있다는 것이므로, 이 문제는 비트 마스크로 풀 수 있습니다.
    2. 예를 들어 [1, 2]가 주어졌을 때의 4가지 모든 경우를 아래와 같이 비트 마스크로 나타낼 수 있습니다.
        1. `00` 는 아무것도 선택하지 않은 것입니다. 따라서 `[]` 를 나타냅니다.
        2. `01` 는 첫 번째 인덱스 원소를 선택한 것입니다. 따라서 `[1]` 를 나타냅니다.
        3. `10` 는 두 번째 인덱스 원소를 선택한 것입니다. 따라서 `[2]` 를 나타냅니다.
        4. `11` 는 첫 번째와 두 번째 인덱스 원소를 선택한 것입니다. 따라서 `[1, 2]` 를 나타냅니다.

비트 마스크를 사용하여 주어진 배열의 모든 부분집합을 출력하는 과정은 아래와 같습니다.

1. 배열의 길이를 구하여 변수 n에 저장합니다.
2. total_subset 변수를 생성한 뒤 2^n을 저장합니다.
3. for 루프를 통해 total_subset-1 까지의 모든 값을 순회합니다.
    1. 비트 마스크를 사용하여 모든 부분집합을 표현하기 위한 인덱스 값을 생성하는 과정입니다.
4. 각 인덱스 값에 대해 빈 리스트 subset을 생성합니다.
    1. 해당 인덱스값에 대응하는 부분집합을 생성하기 위한 리스트입니다.
5. for 루프를 통해 0부터 n-1까지의 모든 값을 순회합니다.
    1. 배열의 각 원소에 대한 비트 상태를 검사하기 위한 인덱스 값을 생성하는 과정입니다.
6. 각 비트에 대해 해당 원소를 선택하면 (i & (1 << j)가 참이면) subset에 해당 원소를 추가합니다.
7. subset 리스트를 출력합니다.
8. 3~8 단계를 통해 모든 부분집합을 생성하고 출력합니다.

이를 코드로 나타내면 다음과 같습니다.

```python
def powerset_using_bitmask(arr):
    n = len(arr)
    total_subset = 1 << n  # 전체 부분집합의 개수를 구함 (2^n)

    for i in range(total_subset):
        subset = []  # 부분집합을 담을 배열
        for j in range(n):
            # 각 경우의 수에 대해 해당 비트가 포함되는지 확인
            if i & (1 << j):
                subset.append(arr[j])
        print(subset)  # 부분집합 출력
```

이 문제는 재귀 함수를 사용해서도 풀 수 있습니다.

아래에 답을 적어놓을 테니 한번 재귀 함수를 사용하여 문제를 풀어보세요!

<details>
<summary>재귀 함수로 푸는 코드</summary>

```python
def powerset_using_recursion(arr):
    n = len(arr)

    def generate_subset(idx, subset):  # 재귀 함수
        if idx >= n:  # 탈출 조건
            print(subset)
            return

        generate_subset(idx + 1, subset + [arr[idx]])  # 해당 인덱스 원소를 포함하기
        generate_subset(idx + 1, subset)  # 해당 인덱스 원소를 포함하지 않기

    generate_subset(0, [])  # generate_subset 호출
```

</details>
