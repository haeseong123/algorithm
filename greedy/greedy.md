# 그리디 알고리즘

그리디 알고리즘(Greedy algorithm)은 매 선택에서 `지금 이 순간 당장 최적인 답` 을 선택을 하는 알고리즘을 뜻합니다. 대다수의 문제에서 그리디
알고리즘은 `최적의 솔루션을 생성하지 않지만 합리적인 시간 내에 합리적인 솔루션을 생성`해 냅니다. 그리디 알고리즘을 사용해도 최적해가 나오는 일부 문제가 존재하지만, 이를 증명하는 것은 매우 어렵습니다.

그리디 알고리즘은 단순하고 빠른 속도로 문제를 해결할 수 있지만, 대부분의 문제에서 최적해를 보장하진 않기때문에, 최적해를 보장해야 하는 경우 다른 알고리즘을 사용해야 합니다. 그러나 최적해를 찾는 것이 불필요하거나
시간이 너무 오래 걸린다면 그리디 알고리즘 사용을 고려할 수 있습니다.

그리디 알고리즘은 틀리긴 하지만 빠른 결과를 도출한다는 점에서 아래 사진과 비슷한 점이 있습니다.

![aa](https://user-images.githubusercontent.com/50406129/233999343-ccd4e488-1bad-4e0b-a0a7-0a7ec5ac4f76.jpg)

> 위 사진은 영 엉터리 대답을 내놓지만 실제 사용되는
> 그리디 알고리즘은 저정도 까진 아니고,
> 정확하진 않지만 쓸만한 답을 내놓습니다.

그리디 알고리즘은 `최적 부분 구조(optimal substructure)` 와
`탐욕적 선택 속성(greedy choice property)` 이라는 두 가지 속성을 만족하는 문제에서 잘 동작합니다.

하위 문제의 최적 솔루션으로 상위 문제의 최적의 솔루션을 구성할 수 있다면 최적 부분 구조를 갖는다고 말합니다. 일반적으로 최적 부분 구조가 각 단계에서 최적임을 귀납적으로 입증할 수 있는 경우 최적 부분 구조를
가진 문제를 해결하기 위해 그리디 알고리즘이 사용됩니다. 그렇지 않고 중복 하위 문제가 있는 경우 분할 정복 또는 DP를 사용할 수 있습니다.

탐욕적 선택 속성은 현재 상황에서 최선의 선택을 계속하면서, 그 선택이 이후의 문제에 영향을 끼치지 않는다면, 이 선택이 최적해를 구하는데 도움이 될 수 있다는 것입니다.

탐욕 알고리즘은 이러한 탐욕적 선택 속성을 이용하여 문제를 해결하는 알고리즘입니다. 탐욕 알고리즘은 `각각의 단계에서 이전 선택의 결과에만 의존하며, 이전의 선택을 다시 고려하지 않습니다.` 이 지점이 DP와의
차이점입니다. DP는 모든 가능한 경우의 수를 고려하여 최적의 해를 찾습니다. 이러한 차이로 인해 탐욕 알고리즘은 항상 최적의 해를 보장하는 것은 아니지만, 계산 속도가 빠른 것입니다.

## 그리디 알고리즘 연관 문제

- [거스름돈](https://github.com/haeseong123/algorithm/blob/main/greedy/change_making/change_making.md)
- [나눌 수 있는 배낭 문제](https://github.com/haeseong123/algorithm/blob/main/greedy/fractional_knapsack/fractional_knapsack.md)
- [최소 신장 트리: Minimum Spanning Tree, MST](https://github.com/haeseong123/algorithm/blob/main/greedy/minimum_spanning_tree/minimum_spanning_tree.md)
- [활동 선택 문제: Activity selection problem](https://github.com/haeseong123/algorithm/blob/main/greedy/activity_selection/activity_selection.md)

## 출처

- [Greedy algorithm](https://en.wikipedia.org/wiki/Greedy_algorithm)
- [그리디 알고리즘](https://namu.wiki/w/%EA%B7%B8%EB%A6%AC%EB%94%94%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98)
