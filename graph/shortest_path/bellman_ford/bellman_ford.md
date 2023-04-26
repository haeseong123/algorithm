# 벨만-포드 알고리즘

벨만-포드 알고리즘(Bellman-Ford algorithm)은 가중치가 있는 방향 그래프에서 최단 경로 문제를 푸는 알고리즘입니다. 이때 변의 가중치는 `음수` 일 수도 있습니다.

[다익스트라 알고리즘](https://github.com/haeseong123/algorithm/blob/main/graph/shortest_path/dijkstra/dijkstra.md)
은 벨먼-포드 알고리즘과 동일한 작업을 수행하며 실행 속도는 더 빠릅니다. 하지만, 가중치가 음수인 경우를 처리할 수 없으므로, 가중치에 음수가 있을 수 있는 경우 벨만-포드 알고리즘을 사용하는 것이 좋습니다.
반대로, 가중치에 음수가 없는 경우 굳이 벨만-포드 알고리즘을 사용할 필요는 없습니다.

## 음수 사이클이 존재하는 경우

음수 사이클이 존재하는 경우에 대해 생각해 봅시다.

- 음수 사이클이 존재하고 양수값이 더 클 경우
    - 사이클을 순환하여도 오히려 가중치의 합이 증가하므로, 해당 사이클을 순환하지 않을테니 별 문제가 되지 않습니다.
- 음수 사이클이 존재하고 음수값이 더 클 경우
    - 사이클을 순환할수록 가중치 합이 감소하므로 무한히 해당 사이클을 돕니다. 따라서 음수 사이클이 존재하고 음수값이 더 클 경우 해답을 찾을 수 없습니다.

## 아이디어

최단 경로 문제는 optimal substructure를 갖습니다. 따라서 시작 노드 s에서 목표 노드 v에 이르는 최단 경로는 s에서 u까지의 최단 경로에 u에서 v 사이의 가중치를 더한 값이라고 볼 수 있습니다.

![aaaa](https://user-images.githubusercontent.com/50406129/234499461-55e616aa-19d6-4fce-b82f-4a0f43a8a16e.PNG)

벨만 포드 알고리즘은 s, v 사이의 최단 경로를 구할 때 그래프 내 모든 엣지에
대해 [Relaxation](https://github.com/haeseong123/algorithm/blob/main/graph/shortest_path/shortest_path.md#relaxation) 을
수행합니다. Relaxation을 한 번 수행할 때마다 최소 하나의 최단 정점이 구해지므로 모든 엣지에 대해 V-1 번 Relaxation을 진행하면 최단 거리를 구할 수 있습니다.(맨 처음 시작 노드를 0으로
설정하기 때문에 V - 1번 하면 됩니다.)

## 구현

벨만 포드 알고리즘은 아래 순서를 따릅니다.

1. 출발 노드를 0으로, 다른 모든 노드는 무한으로 초기화합니다.
2. `모든 엣지에 대해 Relaxation을 수행` 합니다. 이를 총 V-1 번 반복합니다.
3. 마지막으로 Relaxation을 한번 더 수행합니다. 이때 값이 바뀌는 정점이 있다면 음수값이 큰 음수 사이클이 존재한다는 것이므로 None을 반환합니다.

시간 복잡도는 1번에서 O(V), 2번에서 O(VE), 3번에서 O(E) 이므로  
벨만 포드 알고리즘의 `시간 복잡도는 O(VE)` 입니다. V는 정점의 수, E는 간선의 수를 나타냅니다.

코드로 나타내면 다음과 같습니다.

```python
def bellman_ford(graph, source):
    # 초기화
    INF = float('inf')
    distance, predecessor = dict(), dict()
    for node in graph:
        distance[node] = INF
        predecessor[node] = None
    distance[source] = 0

    # 모든 간선에 대해서 V - 1번 확인
    for _ in range(len(graph) - 1):
        for u in graph:
            for v, w in graph[u].items():  # v는 정점, w는 가중치
                if distance[v] > distance[u] + w:
                    distance[v] = distance[u] + w
                    predecessor[v] = u

    # 음수 사이클 확인
    for u in graph:
        for v, w in graph[u].items():
            if distance[v] > distance[u] + w:
                return None

    return [distance, predecessor]


graph = {
    'A': {'B': 2, 'C': -1},
    'B': {'C': 1},
    'C': {}
}
print(bellman_ford(graph, 'A'))
# 결과: [{'A': 0, 'B': 2, 'C': -1}, {'A': None, 'B': 'A', 'C': 'A'}]
```

이제 실제 문제를 풀어봅시다.

## 타임머신

N개의 도시가 있습니다. 그리고 한 도시에서 출발하여 다른 도시에 도착하는 버스가 M개 있습니다. 각 버스는 A, B, C로 나타낼 수 있는데, A는 시작도시, B는 도착도시, C는 버스를 타고 이동하는데 걸리는
시간입니다. 시간 C가 양수가 아닌 경우가 있습니다. C = 0인 경우는 순간 이동을 하는 경우, C < 0인 경우는 타임머신으로 시간을 되돌아가는 경우입니다.

1번 도시에서 출발해서 나머지 도시로 가는 가장 빠른 시간을 구하는 프로그램을 작성해주세요.

### 입력

첫째 줄에 도시의 개수 N(1 <= N <= 500), 버스 노선의 개수 M(1 <= M <= 6,000)이 주어집니다.

둘째 줄부터 M개의 줄에는 버스 노선의 정보 A, B, C(1 <= A, B <= N, -10,000 <= C <= 10,000)이 주어집니다.

### 출력

만약 1번 도시에서 출발해 어떤 도시로 가는 과정에서 시간을 무한히 오래 전으로 되돌릴 수 있다면 첫째 줄에 -1을 출력합니다.

그렇지 않다면 N-1 개 줄에 걸쳐 각 줄에 1번 도시에서 출발해 2번 도시, 3번 도시, ..., N번 도시로 가는 가장 빠른 시간을 순서대로 출력합니다. 만약 해당 도시로 가는 경로가 없다면 대신 -1을
출력합니다.

### 예제

예제 입력 1

```
3 4
1 2 4
1 3 3
2 3 -1
3 1 -2
```

예제 출력 1

```
4
3
```

###

예제 입력 2

```
3 4
1 2 4
1 3 3
2 3 -4
3 1 -2
```

예제 출력 2

```
-1
```

###

예제 입력 3

```
3 2
1 2 4
1 2 3
```

예제 출력 3

```
3
-1
```

### 문제 풀이

정확히 벨만 포드 알고리즘을 사용하여 푸는 문제입니다. 만약 음수 사이클이 존재하며 해당 사이클의 합이 음수일 경우 -1을 출력하면 됩니다. 그리고 갈 수 없는 경우 해당 도시를 출력할 차례에 -1을 출력하라고
했으므로 제일 마지막에서 distance 원소에 무한이 있는 경우 -1로 바꿔 넣으면 됩니다.

코드는 아래와 같습니다.

```python
def bellman_ford(graph, source, n, m):
    # 초기화
    INF = float('inf')
    distance = [INF for _ in range(n + 1)]
    distance[source] = 0

    # V-1 만큼 ...
    for _ in range(n - 1):
        # 모든 간선에 대해 
        # relaxation 합니다.
        for u in range(1, n + 1):
            for v, w in graph[u]:
                if distance[v] > distance[u] + w:
                    distance[v] = distance[u] + w

    # 음수 사이클 확인
    for u in range(1, n + 1):
        for v, w in graph[u]:
            if distance[v] > distance[u] + w:
                print(-1)
                return

    # 출력
    for d in distance[2:]:
        print(-1 if d is INF else d)


# N은 정점의 개수
# M은 간선의 개수
N, M = map(int, input().split())
graph = [[] for _ in range(N + 1)]
for _ in range(M):
    u, v, w = map(int, input().split())
    graph[u].append((v, w))

bellman_ford(graph, 1, N, M)
```

## 출처

- [벨만-포드 알고리즘(Bellman-Ford Algorithm)](https://8iggy.tistory.com/153)
- [[알고리즘 정리] 벨만-포드 알고리즘(Bellman-Ford Algorithm)](https://jeonyeohun.tistory.com/97)
- [벨만-포드 알고리즘](https://ratsgo.github.io/data%20structure&algorithm/2017/11/27/bellmanford/)
- [타임머신](https://www.acmicpc.net/problem/11657)
