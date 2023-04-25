# 나눌 수 있는 배낭 문제

무게와 가치를 가진 n개의 물건과 한 개의 배낭이 있습니다. 이 배낭은 최대 w의 무게를 담을 수 있습니다. 이때, 이 배낭에 담을 수 있는 물건들의 가치의 합이 최대가 되도록 물건들을 배낭에 담는 방법을 찾는
문제입니다.

[0-1 배낭 문제](https://github.com/haeseong123/algorithm/blob/main/dp/01_knapsack/01_knapsack.md)
와 다른 점은 물건을 쪼갤 수 있다는 것입니다. 다시 말해, 0-1 배낭 문제에서는 물건을 넣거나 아예 넣지 않거나 둘 중 하나인 반면, 나눌 수 있는 배낭 문제에서는 물건을 1/3만 담거나 1/7만 담거나 하는
식으로 물건을 나누어서 넣을 수 있는 차이점이 있습니다.

이 문제는 무게당 가치가 제일 높은 물건부터 차례대로 배낭에 담으면 최적해를 찾을 수 있으므로, 그리디 알고리즘으로 항상 최적해를 구할 수 있는 문제입니다.

문제를 하나 풀어보겠습니다.

## Maximum Units on a Truck

트럭 한 대에 일정량의 상자를 실어야 합니다.
`boxTypes[i] = [numberOfBoxesi, numberOfUnitsPerBoxi]` 인 2D 배열 `boxTypes`가 제공되며, 트럭에 실을 수 있는 최대 상자 수인 `truckSize`도
주어집니다. 이때, 트럭에 실을 수 있는 최대 units을 반환해주세요.

### 제약

- 1 <= boxTypes.length <= 1000
- 1 <= numberOfBoxesi, numberOfUnitsPerBoxi <= 1000
- 1 <= truckSize <= 106

### 예제

예제 1

```
Input: boxTypes = [[1,3],[2,2],[3,1]], truckSize = 4
Output: 8

Explanation: There are:
- 3 유닛이 들어가 있는 첫 번째 유형의 상자 1개 
- 각각 2 유닛이 들어가 있는 두 번째 유형의 상자 2개
- 각각 1 유닛이 들어가 있는 세 번째 유형의 상자 3개 

첫 번째와 두 번째 유형의 모든 상자와 세 번째 유형의 상자 하나를 가져갈 때
truckSize 이하의 상자를 담으며 최대의 유닛을 가져갈 수 있습니다.

총 Units는 (1 * 3) + (2 * 2) * (1 * 1) = 8입니다.
```

예제 2

```
Input: boxTypes = [[5,10],[2,5],[4,7],[3,9]], truckSize = 10
Output: 91
```

### 문제 풀이

이 문제는 표현하는 방법만 다를 뿐 사실상 fractional knapsack 문제와 동일한 문제입니다. 따라서 상자 당 단위가 가장 높은 것을 차례대로 담으면 됩니다.

문제 풀이 순서는 다음과 같습니다.

1. 상자 당 단위 내림차순으로 정렬합니다.
2. 상자를 담을 수 있는 만큼 담습니다.
3. 모두 담았다면 다음 상자로 넘어갑니다.
4. 담겨진 상자가 truckSize와 같아지거나 더이상 담을 상자가 없을 때까지 반복합니다.

이를 구현한 코드는 아래와 같습니다.

```python
class Solution:
    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        boxTypes.sort(key=lambda x: -x[1])
        answer = 0

        for boxes, units in boxTypes:
            if boxes == truckSize:
                answer += (boxes * units)
                break
            elif boxes > truckSize:
                answer += (truckSize * units)
                break
            else:
                truckSize -= boxes
                answer += (boxes * units)

        return answer
```

## 출처

- [1710. Maximum Units on a Truck](https://leetcode.com/problems/maximum-units-on-a-truck/description/)
