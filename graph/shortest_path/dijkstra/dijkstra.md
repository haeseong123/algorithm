# 다익스트라 알고리즘

다익스트라(Dijkstra) 알고리즘은 하나의 출발점에서 다른 모든 정점까지의 최단 경로를 찾는 알고리즘입니다. 이 알고리즘은 음의 가중치가 없는 그래프에서 동작합니다. 만약, 음의 가중치가 있다면 제대로 동작하지
않기 때문에 이
경우 [벨만 포드 알고리즘](https://github.com/haeseong123/algorithm/blob/main/graph/shortest_path/bellman_ford/bellman_ford.md) 을
사용하는 것이 좋습니다.

다만, 실제 세계에서는 음의 경로가 존재하는 경우가 많이 없고 벨만 포드 알고리즘보다 빠르기 때문에 실세계에서 범용적으로 사용될 수 있습니다.

## 구현

다익스트라 알고리즘의 동작은 다음과 같습니다.

1. 시작점은 0으로 초기화하고 나머지 모든 정점은 양의 무한대로 초기화합니다.
2. 시작 정점을 Q에 넣습니다.
3. Q에서 최소 정점을 뽑습니다.
    1. 항상 최소 정점을 봅아야 하므로 보통 우선 순위 큐(최소힙)를 사용합니다.
4. 꺼낸 정점과 이어진 모든 정점에 대해 Relaxation을 수행합니다.
    1. 연관된 정점의 값이 업데이트 된다면 Q에 업데이트 된 정점을 넣습니다.
5. Q가 빌 때까지 3~4를 반복합니다.

시작 지점에서부터 시작하여, 인접해 있는 모든 정점들에 대해 최단 거리를 계산하고, 갱신합니다. 그 뒤 최단 거리를 갖는 정점을 뽑고 또 반복합니다. 이렇게 하면 매 순간 최단 거리가 되는 노드만을 선택하여 거리를
설정하기 때문에 그리디 알고리즘의 성격을 띕니다. 이러한 매 최단 거리를 선택하고 해당 정점과 연결된 간선을 계산하는 일련의 작업이 가능한 이유는 음의 가중치를 갖는 간선이 없기 때문입니다.

Q가 최소힙이라는 가정하에, 방문하는 노드 수는 V 이고 방문 시 항상 최소 거리를 뽑으므로 LogV입니다. 또한 모든 변을 한 번씩은 지나가므로 E이며, 연관된 정점 값 업데이트 시 Q에 새로 넣으므로
LogV입니다. 따라서 시간복잡도는 `O(VlogV + ElogV) = O((V + E)logV)` 입니다.

다익스트라 알고리즘을 코드로 나타내면 아래와 같습니다.

```python
import heapq


def dijkstra(graph, start):
    # 초기화
    INF = float('inf')
    distances = {vertex: INF for vertex in graph}
    distances[start] = 0
    queue = [(distances[start], start)]

    # 최단 거리 계산 ...
    while queue:
        curr_distance, u = heapq.heappop(queue)

        # 한 노드가
        # 큐에 중복으로 들어갈 수 있습니다.
        # 다만 확실한 것은
        # 중복으로 들어갔더래도 최단 거리가 먼저
        # heappop() 되므로
        # distances는 이미 업데이트 되었을 것이고
        # 따라서 더 큰 값을 기준으로 계산을
        # 한번 더 할 필요는 없습니다.
        if curr_distance > distances[u]:
            continue

        for v, w in graph[u].items():
            new_distance = distances[u] + w  # u를 거쳐가는 경로
            if distances[v] > new_distance:  # u를 거쳐가는 거리가 기존 거리보다 짧다면
                distances[v] = new_distance  # 기존 거리를 변경
                heapq.heappush(queue, (new_distance, v))  # 다음 계산을 위해 큐에 삽입

    return distances
```
