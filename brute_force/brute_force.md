# 완전 탐색

완전 탐색은 ***가능한 모든 선택지를 시도*** 하여 문제의 해답을 찾는 알고리즘입니다.

다양한 경우의 수를 모두 검토하여 최적 또는 가능한 해답을 찾는 장점이 있지만, 경우의 수가 너무 많을 경우에는 실행 시간이 길어질 수 있는 단점이 있습니다. 완전 탐색은 `brute-force search`
또는 `exhaustive search`라고도 합니다.

주요 특징은 다음과 같습니다.

- 가능한 모든 선택지를 시도합니다.
- 비효율적일 수 있습니다.
    - 경우의 수가 많을 경우 실행 시간이 길어질 수 있습니다.
- 정확성을 보장합니다.
    - 가능한 모든 경우를 탐색하기 때문에 최적 또는 가능한 모든 해답을 찾을 수 있습니다.

> 완전 탐색은 경우의 수가 많을 경우 시간이 매우 길어질 수 있으므로
> 문제를 풀 때 제한 시간 내에 완전 탐색으로 해결이 가능한 문제인지
> 사전에 생각해보는 것이 중요합니다.

경우의 수가 많아지면 실행 시간이 길어질 수 있고, 가능한 모든 선택지를 탐색하는 것이 완전 탐색이라는 것을 배웠습니다.

완전 탐색은 `재귀 함수`, `비트 마스크`, `순열-조합`, `BFS-DFS` 등 다양한 방법으로 구현할 수 있습니다. 이들은 그들 스스로만을 사용하여 완전 탐색을 구현하기도 하고, 다른 구현의 도구로써 사용되기도
합니다. 예를 들어 재귀 함수는 재귀 함수만으로 완전 탐색 알고리즘을 구현할 수 있지만 BFS를 구현하기 위한 하나의 도구로 사용되기도 합니다.

자 이제 `재귀 함수`, `비트 마스크`, `순열-조합`, `BFS-DFS` 에 대해 배워봅시다.

- [재귀 함수](https://github.com/haeseong123/algorithm/blob/main/brute_force/recursive/recursive.md)
- [비트 마스크](https://github.com/haeseong123/algorithm/blob/main/brute_force/bitmask/bitmask.md)
- [순열, 조합](https://github.com/haeseong123/algorithm/blob/main/brute_force/perm_comb/perm_comb.md)
- [DFS, BFS](https://github.com/haeseong123/algorithm/blob/main/brute_force/dfs_bfs/dfs_bfs.md)
