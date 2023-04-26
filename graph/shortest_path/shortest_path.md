# 최단 경로 찾기

그래프 최단 경로 문제란, 그래프 상에서 두 정점을 연결하는 경로 중에서 가장 짧은 경로를 찾는 문제입니다. 즉, 시작점과 끝점이 주어졌을 때, 그 둘 사이를 가장 적은 비용으로 이동할 수 있는 경로를 찾는
것입니다.

최단 경로 문제는 몇 가지 유형으로 나누어 집니다.

- `Single Source` : 하나의 노드로부터 출발해서 다른 모든 노드의 최단 경로를 찾는 문제
- `Single Destination` : 모든 노드로부터 하나의 목적지 까지의 최단 경로를 찾는 문제
- `Single Pair` : 주어진 하나의 노드로부터 하나의 목적지까지의 최단 경로를 찾는 문제
- `All Pair` : 모든 노드 쌍에 대한 최단 경로를 찾는 문제

Single Destination은 방향 그래프의 간선 방향을 모두 반대로 뒤집으면 Single Source 문제가 됩니다.

Single Pair 역시 Single Source의 일반화된 문제이며, All Pair도 모든 정점에서 Single Source를 사용하면 풀립니다.

대표적인 최단 경로 알고리즘으로는 `벨만-포드 알고리즘`, `다익스트라 알고리즘` ,
`플로이드-워셜 알고리즘` 이 있습니다.

`벨만-포드, 다익스트라는 Single Source` 알고리즘이고
`플로이드-워셜은 All Pair` 알고리즘입니다. 물론, 위에 언급한대로 Single Source 알고리즘으로 모든 유형의 경로 찾기 문제를 해결할 수 있지만, 벨만 포드 혹은 다익스트라로 All pair 문제를
풀면 시간이 매우 오래 걸리기 때문에 각 문제 특성에 맞는 알고리즘이 따로 존재합니다. 그러므로 문제의 유형을 잘 파악하여 적절한 알고리즘을 사용하는 것이 필요합니다.

## optimal substructure

최단 경로 찾기 문제는 최적 부분 구조(optimal substructure) 속성을 갖습니다. 최단 경로 찾기 알고리즘은 이러한 최적 부분 구조 속성을 이용하여 문제를 해결합니다.

최단 경로 찾기 문제에서 optimal substructure는
`최단 경로의 부분 경로 역시 최단 경로이다.` 입니다.

귀류법으로 증명해 보겠습니다. 어떤 경로가 최단 경로일 때 해당 경로에 속하는 subpath가 optimal substructure가 아니라면, 해당 subpath 보다 더 짧은 subpath가 있다는 것을
의미합니다. 현재 subpath 보다 짧은 subpath가 있다면, global optimal solution을 위해서는 해당 subpath가 반드시 최종 solution에 포함되어 있어야 하고, 이는 곧 초기에
설정했던 경로가 더 이상 최단 경로가 될 수 없음을 의미합니다. 따라서 어떤 최단 경로가 있다면, 그 안에 속한 다른 모든 경로들도 최단 경로입니다.

따라서 어떤 경로가 최단 경로리면, 이 경로의 길이는 그래프 내에 다른 정점들을 거쳐서 오는 모든 경로보다 같거나 작습니다.

## single-source 알고리즘 전략

하나의 노드로부터 출발해서 다른 모든 노드의 최단 경로(single-source shortest path, SSP)를 찾는 문제를 해결하는 알고리즘에 통용되는 몇가지 전략이 있습니다.

### Distance

어떤 두 정점 사이의 길이를 저장하기 위해 배열을 사용합니다. `d[v]` 는 시작점으로부터
`정점 v` 까지의 최단 길이를 저장합니다. 따라서 여러 경로가 발견될 때 가장 길이가 짧은 경로의 길이 합을 이 배열에 저장해야합니다.

### Predecessor

어떤 정점에서 최단 경로로 연결된 다른 정점을 표현하기 위해서 배열을 사용합니다.
`p[v]` 는 최단 경로를 구성하는 정점들 중 `v` 의 부모 노드가 되는 정점을 저장합니다.

### Relaxation

최단 경로를 찾는 문제에서 가장 핵심이 되는 것이 이 relaxation 기법입니다.
`Relaxation` 은 어떤 한 정점으로부터 연결된 모든 정점을 탐색하고 그 중에서 가장 길이가 짧은 정점을 찾아 Shortest Path를 만드는 것입니다. 다시 말해, SSP는 이렇게 `d[v]` 값을 계속
조정하는 것이 핵심입니다.

아래 그림에서 Original을 보시면 d[v2]가 9입니다. 이는 현재의 최단 경로 추정 값이지 최단 경로 확정 값이 아닙니다.

경로를 구하는 방법은 이전 정점(v1)까지의 경로 비용과 연결된 가중치의 비용을 더하는 것입니다. 그래서 d[v1]이 5이고, 연결된 간선의 가중치가 2이므로 d[v2]는 7입니다.

이렇게 새로운 값 7은 기존 추정 값 9보다 더 작은 비용이므로 d[v2]를 7로 수정합니다. 이렇게 비교 연산을 수행하는 것을 Relax라 합니다.

만약 새로운 값이 기존 추정 값보다 크다면 값을 바꾸지 않아야 합니다.

![aa](https://user-images.githubusercontent.com/50406129/234487617-14324c72-8caa-428c-a57a-1ad1c1f7322c.PNG)

정리하자면 Relax란 `목적지 정점(v2)의 추정 값과 직전 정점(v1)의 추정값 + 간선의 가중치를 비교하여 목적지 정점의 추정 값을 조정하는 것` 을 의미합니다.

Relax의 수도 코드는 아래와 같습니다.

```
Relax(u, v, w){
    if(d[v] > d[u] + w)
        d[v] = d[u] + w;
        p[v] = u;
}
```

## all-pair 알고리즘 전략

모든 쌍의 최단 경로(All Pairs Shortest Path, ASP)를 찾는 문제는 말 그대로 모든 정점이 출발점이 되어 모든 정점에 대해 최단 경로를 구하는 것입니다.

예를 들어, 아래 그림에서 v1 -> v2 -> v4 -> v5 라는 v1 -> v5의 최단 경로가 존재할 수 있고 v3 -> v2 -> v4 라는 최단 경로가 존재할 수 있습니다.

ASP를 구현하는 알고리즘은 대부분 인접 행렬을 사용합니다.

### 인접 행렬

인접 행렬을 W 라고 할 때, 인접 행렬의 각 원소는 W_ij입니다. (i는 행, j는 열을 의미합니다.)
인접 행렬의 각 행과 열의 개수는 각각 정점의 개수만큼 존재합니다.

아래와 같이 정점 간의 최단 경로를 인접 행렬로 표현할 수 있습니다.

![aaa](https://user-images.githubusercontent.com/50406129/234491217-afe129f2-b407-43d4-ab6c-501f5a2ecbd4.PNG)

## 최단 경로 찾기 알고리즘

- [벨만-포드 알고리즘](https://github.com/haeseong123/algorithm/blob/main/graph/shortest_path/bellman_ford/bellman_ford.md)
- [다익스트라 알고리즘](https://github.com/haeseong123/algorithm/blob/main/graph/shortest_path/dijkstra/dijkstra.md)
- [플로이드-워셜 알고리즘](https://github.com/haeseong123/algorithm/blob/main/graph/shortest_path/floyd_warshall/floyd_warshall.md)

## 출처

- [위키- Shortest path problem](https://en.wikipedia.org/wiki/Shortest_path_problem)
- [위키- Optimal substructure](https://en.wikipedia.org/wiki/Optimal_substructure)
- [최단 경로 문제](https://ratsgo.github.io/data%20structure&algorithm/2017/11/25/shortestpath/#)
- [[알고리즘 정리] 최단경로(Shortest Path)](https://jeonyeohun.tistory.com/96)
- [🙈[알고리즘] SSP(1) - 단일 출발지 최단 경로 (Single Source Shortest Path)🐵](https://victorydntmd.tistory.com/103)
- [🙈[알고리즘] ASP(1) - 모든 쌍의 최단 경로 ( All Pairs Shortest Path )🐵](https://victorydntmd.tistory.com/106)
