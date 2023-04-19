# DFS와 BFS

DFS(depth-first search, 깊이 우선 탐색)와 BFS(breadth-first search)는 ***그래프 탐색
알고리즘***의 두 가지 주요 방법입니다. 배열의 경우 인덱스가 있으니, 반복문을 통해
0번 인덱스부터 n-1 번 인덱스까지 순회하면 배열 전체를 탐색할 수 있지만, 그래프의 경우
인덱스가 없습니다. ***인덱스라는 개념이 없는 그래프에서 탐색하는 방법***이 DFS와 BFS입니다.

DFS든 BFS든 어떤 것이든, 데이터를 조회하려면 데이터가 현재 메모리에 올라와 있어야 합니다.
이러한 방법, 다시 말해 그래프를 데이터로 표현하는 방법에 대해 배운 뒤에 
DFS와 BFS를 배우겠습니다.

그래프를 데이터로 표현하는 방법에는 여러 방법이 있지만 그중 가장 일반적인 
두 가지 방법인 인접 행렬과 인접 리스트에 대해 알아봅시다.

## 인접 행렬

인접 행렬(Adjacency Matrix)은 그래프의 연결 상태를 ***2차원 배열로 표현*** 한 것입니다.
즉, 그래프의 노드들을 배열의 인덱스로 표현하고, 각 노드 사이의 연결 여부를 배열의 값으로
표현합니다. 만약, 노드 u와 노드 v가 연결되어 있다면, 인접 행렬의(u, v)와 (v, u) 원소 값이
1이 됩니다. 연결되어 있지 않은 경우에는 0으로 표시됩니다. 

인접 행렬을 사용하면, ***노드 간 연결 확인을 
상수 시간 O(1)에 확인*** 할 수 있어서, 그래프의 크기가 작은 경우나 
간선의 수가 최대 간선의 수에 가까운 그래프 밀집 그래프(dense graph)의 경우에 효율적입니다. 
하지만, ***공간 복잡도가 O(N^2)*** 이기 때문에 
그래프의 크기가 큰 경우에는 메모리를 많이 차지하게 됩니다.

### 무방향 그래프 그리기

아래와 같은 무방향 그래프가 있을 때의 인접 행렬은 다음과 같습니다.

![graph](https://user-images.githubusercontent.com/50406129/232980124-413f0e88-e5b9-424c-9ca9-a86ad91dba22.png)

|_|0|1|2|3|
|---|---|---|---|---|
|0|0|1|1|0|
|1|1|0|1|0|
|2|1|1|0|1|
|3|0|0|1|0|

무방향 그래프에서는 간선이 서로의 연결, 즉 2와 3이 연결되어 있다면 2에서 3으로 갈 수도 있고
3에서 2로 갈 수도 있음을 표현하므로 무방향 그래프를 인접 행렬로 표현할 경우 
대각선을 기준으로 대칭인 모양이 됩니다.

### 방향 그래프 그리기

아래와 같은 방향 그래프가 있을 때의 인접 행렬은 다음과 같습니다.

![graph_directed](https://user-images.githubusercontent.com/50406129/232983394-14325359-7eff-4844-8064-d355ad155e41.png)

|_|0|1|2|3|
|---|---|---|---|---|
|0|0|1|1|0|
|1|0|0|0|0|
|2|0|1|0|1|
|3|0|0|0|0|

인접 행렬의 가로를 보면 내가 가리키고 있는 대상을 확인할 수 있고 
세로를 보면 나를 가리키는 대상을 확인할 수 있습니다.

예를 들어, 위 방향 그래프에서 `0`의 경우 `1`과 `2`를 가리키고 있기 때문에 첫 번째 `행`의 
1번 인덱스와 2번 인덱스에 1이 들어가 있습니다. 
반대로, 아무도 `0`을 가리키고 있지 않기 때문에 첫 번째 `열`의 모든 값은 0이 들어가 있습니다.

### 장단점

마지막으로 인접 행렬의 장단점에 대해 알아보겠습니다.

- 장점
  - 쉽게 구현할 수 있습니다.
  - 연결을 제거하는 시간 복잡도가 O(1)입니다.
    - 행렬의 해당 인덱스 요소의 값을 1에서 0으로 바꾸기만 하면 됩니다.
  - 연결을 확인하는 시간 복잡도가 O(1)입니다.
    - 행렬의 해당 인덱스 요소의 값이 1인지 0인지 확인하면 됩니다.
- 단점
  - O(N^2)의 공간 복잡도를 갖습니다.
  - 연결이 많든 적든 언제나 O(N^2)의 공간 복잡도를 갖습니다.
    - 예를 들어 노드가 1,000개이고 연결은 단 두 개일 때에도 1,000^2의 공간을 차지하므로
    공간 낭비가 심합니다.
  - 인접 행렬을 만드는 시간이 O(N^2)입니다.
  - 한 노드와 연결된 모든 인접 노드를 찾는 시간이 O(N)입니다.

## 인접 리스트

인접 리스트(Adjacency List)는 노드마다 연결된 노드들의 리스트를 저장하는 방법입니다.
즉, 그래프의 노드를 배열에 저장하고, ***배열의 인덱스에 해당하는 노드와 연결된 노드들을 
연결 리스트(Linked List)로 저장합니다.*** 인접 리스트를 사용하면, 연결 리스트를 순회하여 
노드 간의 연결 관계를 확인할 수 있습니다. 

인접 리스트는 ***그래프의 크기에 비례하는 공간을 차지***하므로, 
간선이 별로 없는 희소 그래프(sparse graph)의 경우에 효율적입니다. 
하지만, 노드 간의 연결을 확인하기 위해 순회해야 하는 시간이 더 오래 걸립니다.

### 무방향 그래프 그리기

아래와 같은 무방향 그래프가 있을 때의 인접 리스트는 다음과 같습니다.

![graph](https://user-images.githubusercontent.com/50406129/232980124-413f0e88-e5b9-424c-9ca9-a86ad91dba22.png)

![adjacency_list_non_directed](https://user-images.githubusercontent.com/50406129/232991605-dc924d56-494c-4956-85bd-01e8b91a2802.PNG)

리스트를 만들고 x번째 인덱스에 x와 연결된 노드들을 저장합니다.
`0`은 `1`과 `2`와 연결되어 있기 때문에 `0->1->2` 형식으로 저장된 것입니다.
여기서 `0->1->2->`는 `0`이 `1`과 `2`와 연결되어 있다는 뜻이지
`1`이 `2`와 연결되어 있다는 의미는 아닙니다. 

`1`이 누구와 연결되어 있는지 찾아보려면 1번 인덱스를 찾아가서 찾아봐야 합니다.

### 방향 그래프 그리기

아래와 같은 방향 그래프가 있을 때의 인접 리스트는 다음과 같습니다.

![graph_directed](https://user-images.githubusercontent.com/50406129/232983394-14325359-7eff-4844-8064-d355ad155e41.png)

![adjacency_list_directed](https://user-images.githubusercontent.com/50406129/232994012-6e9baf86-cecb-4510-9d04-d4e205986178.PNG)

인접 행렬로 방향 그래프를 나타냈을 때보다 확실히 간단해진 것을 확인할 수 있습니다.
이처럼 인접 리스트는 간선이 많지 않을 때 사용하면 인접 행렬보다 공간을 확실히 
덜 사용함을 확인할 수 있습니다.

### 장단점

인접 리스트의 장단점은 인접 행렬의 장단점의 반대라고 생각하면 됩니다.
아래는 인접 리스트의 장단점을 정리해 놓은 것입니다.

- 장점
  - 공간을 적게 사용합니다.
    - O(N + E)
    - 최악의 상황에는 O(N^2)
  - 한 노드와 연결된 모든 노드를 찾을 때 보통 인접 행렬보다 빠릅니다.
  - 삽입/삭제가 빠릅니다.
    - 연결 리스트의 경우 삽입/삭제가 빠르므로 연결 리스트를 사용하는 인접 리스트는 
    삽입/삭제가 빠릅니다. 단, 인접 리스트 내부에서 
    다른 데이터 구조를 사용한다면 바뀔 수 있습니다.
- 단점
  - 한 노드에서 다른 노드로 가는 간선이 존재하는지 확인하려면 해당 노드와 연결된 
  모든 노드를 순회해야 합니다.
    - 인접 행렬의 경우 인덱스를 통해 O(1) 시간에 확인할 수 있었는데, 인접 리스트의 경우 그보다 느립니다.

* * *

인접 행렬 혹은 인접 리스트를 사용하여 그래프를 메모리에 저장할 수 있게 되었습니다.
이제 해당 데이터를 통해 그래프를 순회해 봅시다.

* * *

## DFS

DFS(depth-first search)는 그래프에서 깊이를 우선으로 탐색하는 알고리즘입니다.
DFS는 스택(stack)이나 재귀 함수(recursive function)를 이용하여 구현할 수 있습니다.
아주 쉽게 말하자면 DFS는 한 우물 알고리즘이라고 할 수 있습니다.

DFS는 아래 그림과 같이 한 우물을 쭉 파고 다 팠으면 부모 노드로 올라와서 방문하지 않은
다른 자식 노드를 방문하는 식으로 그래프를 순회합니다.

![DFS](https://user-images.githubusercontent.com/50406129/232975970-49a8127d-6391-4543-a4b5-f997f3013679.gif)

DFS 알고리즘은 다음과 같은 과정으로 진행됩니다.

1. 시작 노드를 스택에 삽입하고, 시작 노드를 방문했다고 표시합니다.
2. 스택의 최상단 노드를 꺼내서 그 노드와 인접한 노드 중에서, 방문하지 않은
노드를 스택에 삽입하고, 방문했다고 표시합니다.
3. 스택이 빌 때까지 2번의 과정을 반복합니다.

### 그래프가 인접 행렬로 저장된 경우

그래프가 인접 행렬로 저장되어 있을 경우의 DFS에 대해 알아보겠습니다.

visited는 방문한 노드를 재방문하지 않기 위해 있는 변수입니다.
초깃값은 전부 False가 들어가 있고 방문하면 True로 값을 바꿉니다.

stack에 넣을 때 방문했음을 표시하기 위해 visited 값을 True로 바꿉니다.

이를 코드로 나타내면 아래와 같습니다.

```python
def dfs_adj_matrix(graph):
  n = len(graph)
  visited = [False for _ in range(n)]

  stack = [graph[0]]
  visited[0] = True

  while stack:
    node = stack.pop()
    print(node)

    for i, connected in enumerate(node):
      if connected and not visited[i]:
        stack.append(graph[i])
        visited[i] = True
```

모든 정점을 방문하며 각 정점이 갖고 있는 다른 정점에 대한 연결을 확인하기 위해 정점만큼의
길이를 또 반복하기 때문에 `O(V^2)` 시간 복잡도를 갖습니다.

### 그래프가 인접 리스트로 저장된 경우

그래프가 인접 리스트로 저장되어 있을 경우의 DFS에 대해 알아보겠습니다.

코드는 아래와 같습니다.

```python
def dfs_adj_list(graph):
  visited = set()

  stack = [graph[0]]
  visited.add(0)

  while stack:
    node = stack.pop()
    print(node)

    for neighbor_index in node:
      if neighbor_index not in visited:
        stack.append(graph[neighbor_index])
        visited.add(neighbor_index)
```

dfs_adj_matrix()와는 달리 재방문 확인을 위한 visited를 boolean 배열로 선언하지 않고
set을 사용했다는 점을 제외하면 거의 비슷한 코드입니다.

모든 정점을 방문하되 각 정점과 실제 연결된 정점만 방문하므로 `O(V+E)` 
시간복잡도를 갖습니다.

### 끊어져 있는 그래프

그래프가 아래와 같이 끊어져 있는 경우 위에 있는 함수를 사용해도 그래프 전체를 순회할 수 없습니다.
왜냐하면 위에 있는 함수들은 시작점이 하나인데 그래프가 끊어져 있으니 두 그룹 중 한 그룹은
순회할 수 없기 때문입니다.

아래 사진으로 예를 들자면, `0` 번에서 순회를 시작하면 `0`과 `2`는 방문할 수 있지만 `1`은 
방문하지 못합니다. 반대로 `1`에서 순회를 시작하면 `1`만 방문이 되고 `0`과 `2`는 방문하지
못한 채 함수가 종료됩니다.

따라서 이런 경우 함수의 시작점을 모든 노드로 설정해야 합니다. `0`에서도 순회하고 `1`에서도 
순회하고 `2`에서도 순회하는 것입니다. 

![two](https://user-images.githubusercontent.com/50406129/233084905-0e9ca2e8-0fd4-4eb2-b003-0b3536177b02.png)

끊어진 그래프에서 모든 노드를 순회하는 DFS 코드는 아래와 같습니다.

그래프는 인접 리스트로 저장되어 있다고 가정합니다.

```python
def dfs(graph):
  n = len(graph)
  visited = set()

  def dfs_helper(start):
    stack = [graph[start]]
    visited.add(start)

    while stack:
      node = stack.pop()
      print(node)

      for neighbor_index in node:
        if neighbor_index not in visited:
          stack.append(graph[neighbor_index])
          visited.add(neighbor_index)

  for i in range(n):
    if i in visited:
      continue
    dfs_helper(i)
```

기존 인접 리스트를 사용한 DFS와 비교하면, 
반복문을 통해 모든 노드를 시작 노드로 설정하여 DFS를 호출하는 것이 거의 유일한 차이점입니다.

시간 복잡도는 `O(V + E)`입니다.

DFS를 구현하는 방법을 배웠으니, DFS를 사용하여 실제 문제를 풀어봅시다.

### 타겟 넘버

n개의 음이 아닌 정수들이 있습니다. 
이 정수들을 순서를 바꾸지 않고 적절히 더하거나 빼서 타겟 넘버를 만들려고 합니다. 
예를 들어 [1, 1, 1, 1, 1]로 숫자 3을 만들려면 다음 다섯 방법을 쓸 수 있습니다.

```
-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3
```

사용할 수 있는 숫자가 담긴 배열 numbers, 
타겟 넘버 target이 매개변수로 주어질 때 숫자를 적절히 더하고 빼서 
타겟 넘버를 만드는 방법의 수를 return 하도록 solution 함수를 작성해 주세요.

#### 제한사항

- 주어지는 숫자의 개수는 2개 이상 20개 이하입니다.
- 각 숫자는 1 이상 50 이하인 자연수입니다.
- 타겟 넘버는 1 이상 1000 이하인 자연수입니다.

#### 입출력 예

|numbers|result|return|
|------------|--------|------|
|[1, 1, 1, 1, 1]|3|5|
|[4, 1, 2, 1]|4|2|

#### 입출력 예 설명

입출력 예 #1: 문제 예시와 같습니다.

입출력 예 #2:

```
+4+1-2+1 = 4
+4-1+2-1 = 4
```

#### 문제 풀이

DFS를 사용하여 주어진 숫자 배열의 각 원소에 대해 더하거나 빼는 경우의 수를 모두 확인하고,
타겟 넘버와 일치하는 경우의 수를 세는 것입니다.

아래는 이를 구현한 코드입니다.

```python
def solution(numbers, target):
  last_index = len(numbers) - 1
  count = 0
  stack = [(0, -numbers[0]), (0, numbers[0])]

  while stack:
    idx, acc = stack.pop()

    if idx == last_index:
      if acc == target:
        count += 1
      continue

    stack.append((idx + 1, acc - numbers[idx + 1]))
    stack.append((idx + 1, acc + numbers[idx + 1]))

  return count
```

### 네트워크

네트워크란 컴퓨터 상호 간에 정보를 교환할 수 있도록 연결된 형태를 의미합니다. 
예를 들어, 컴퓨터 A와 컴퓨터 B가 직접적으로 연결되어 있고, 
컴퓨터 B와 컴퓨터 C가 직접적으로 연결되어 있을 때 컴퓨터 
A와 컴퓨터 C도 간접적으로 연결되어 정보를 교환할 수 있습니다. 
따라서 컴퓨터 A, B, C는 모두 같은 네트워크상에 있다고 할 수 있습니다.

컴퓨터의 개수 n, 연결에 대한 정보가 담긴 2차원 배열 
computers가 매개변수로 주어질 때, 네트워크의 개수를 
return 하도록 solution 함수를 작성해 주세요.

#### 제한사항

- 컴퓨터의 개수 n은 1 이상 200 이하인 자연수입니다.
- 각 컴퓨터는 0부터 n-1인 정수로 표현합니다.
- i번 컴퓨터와 j번 컴퓨터가 연결되어 있으면 computers[i][j]를 1로 표현합니다.
- computer[i][i]는 항상 1입니다.

#### 입출력 예

|n|computers|return|
|------------|--------|------|
|3|[[1, 1, 0], [1, 1, 0], [0, 0, 1]]|2|
|3|[[1, 1, 0], [1, 1, 1], [0, 1, 1]]|1|

#### 문제 풀이

끊어진 그래프를 주므로, 해당 그래프의 총그룹의 개수를 세면 됩니다.

아래는 이를 구현한 코드입니다.

```python
def count_groups_in_graph(n, graph):
    visited = set()
    count = 0

    def dfs_helper(start):
        nonlocal count
        stack = [graph[start]]
        count += 1
        visited.add(start)

        while stack:
            node = stack.pop()

            for i, connected in enumerate(node):
                if connected and i not in visited:
                    stack.append(graph[i])
                    visited.add(i)

    for i in range(n):
        if i in visited:
            continue
        dfs_helper(i)

    return count


def solution(n, computers):
    return count_groups_in_graph(n, computers)
```

## BFS

BFS(breadth-first search)는 그래프에서 레벨을 우선으로 탐색하는 알고리즘입니다.
BFS는 보통 큐(Queue) 자료구조를 이용하여 구현합니다.
먼저 시작 노드에 큐를 삽입하고, 다음에 방문할 노드들을 차례로 큐에 삽입하면서 방문 여부를
체크합니다. 이때, 먼저 큐에 들어간 노드부터 방문하므로 레벨에 따라 순차적으로 탐색하는
효과가 있습니다.

BFS는 레벨을 우선으로 탐색하므로 그래프의 최단 거리를 구하는 문제에서 많이 사용됩니다.

DFS가 한 우물을 파는 알고리즘이었다면 BFS는 넓고 얇게 순회하는 알고리즘입니다.

![bfs](https://user-images.githubusercontent.com/50406129/233113021-13a34c44-e7e8-4974-93e6-092e17bb30e4.gif)

BFS 알고리즘은 다음과 같은 과정으로 진행됩니다.

1. 시작 노드를 선택합니다. 이 노드를 방문했다고 표시하고, 큐에 넣습니다.
2. 큐에서 노드를 꺼냅니다.
3. 꺼낸 노드의 인접 노드들을 방문했다고 표시하고, 큐에 넣습니다.
4. 큐가 비어있지 않다면 2번으로 돌아가 반복합니다.

### 그래프가 인접 행렬로 저장된 경우

그래프가 인접 행렬로 저장되어 있을 경우의 BFS에 대해 알아보겠습니다.

DFS에서 사용했던 stack 대신에 queue 자료구조를 사용하는 것이 주요한 차이입니다.

이를 코드로 나타내면 아래와 같습니다.

```python
from collections import deque


def bfs_adj_matrix(graph):
  visited = set()
  queue = deque()

  queue.append(graph[0])
  visited.add(0)

  while queue:
    node = queue.popleft()
    print(node)

    for i, connected in enumerate(node):
      if connected and i not in visited:
        queue.append(graph[i])
        visited.add(i)
```

모든 정점을 방문하며 각 정점이 갖고 있는 다른 정점에 대한 연결을 확인하기 위해 정점만큼의
길이를 또 반복하기 때문에 `O(V^2)` 시간 복잡도를 갖습니다.

### 그래프가 인접 리스트로 저장된 경우

그래프가 인접 리스트로 저장되어 있을 경우의 BFS에 대해 알아보겠습니다.

코드는 아래와 같습니다.

```python
from collections import deque


def bfs_adj_list(graph):
  visited = set()
  queue = deque()

  queue.append(graph[0])
  visited.add(0)

  while queue:
    node = queue.popleft()
    print(node)

    for i in node:
      if i not in visited:
        queue.append(graph[i])
        visited.add(i)
```

모든 정점을 방문하되 각 정점과 실제 연결된 정점만 방문하므로 `O(V+E)`
시간복잡도를 갖습니다.

### 끊어져 있는 그래프

DFS를 사용하여 끊어져 있는 그래프를 탐색한 것과 마찬가지로 BFS를 사용하여 
끊어져 있는 그래프를 탐색해 봅시다.

끊어진 그래프에서 모든 노드를 순회하는 BFS 코드는 아래와 같습니다.

그래프는 인접 리스트로 저장되어 있다고 가정합니다.

```python
from collections import deque


def bfs(graph):
  visited = set()

  def bfs_helper(start):
    queue = deque()
    queue.append(graph[start])
    visited.add(start)

    while queue:
      node = queue.popleft()
      print(node)

      for idx in node:
        if idx not in visited:
          queue.append(graph[idx])
          visited.add(idx)

  for i in range(len(graph)):
    if i in visited:
      continue
    bfs_helper(i)
```

시간 복잡도는 `O(V + E)`입니다.

BFS를 구현하는 방법을 배웠으니, BFS를 사용하여 실제 문제를 풀어봅시다.

### 네트워크

DFS의 실전 문제 풀이에서 풀었던 네트워크 문제는 DFS로 풀릴 뿐만 아니라, BFS로도 풀리는 문제입니다.
왜냐하면 DFS를 사용하든 BFS를 사용하든 어쨌든 둘 다 그래프를 전부 순회하기 때문입니다.

#### 문제 풀이

끊어진 그래프를 주므로, 해당 그래프의 총그룹의 개수를 세면 됩니다.

아래는 이를 구현한 코드입니다.

```python
from collections import deque


def count_groups_in_graph(n, graph):
    visited = set()
    count = 0

    def bfs_helper(start):
        nonlocal count
        queue = deque([graph[start]])
        count += 1
        visited.add(start)

        while queue:
            node = queue.popleft()

            for idx, connected in enumerate(node):
                if connected and idx not in visited:
                    queue.append(graph[idx])
                    visited.add(idx)

    for i in range(n):
        if i in visited:
            continue
        bfs_helper(i)

    return count


def solution(n, computers):
    return count_groups_in_graph(n, computers)
```

### 게임 맵 최단거리

ROR 게임은 두 팀으로 나누어서 진행하며, 
상대 팀 진영을 먼저 파괴하면 이기는 게임입니다. 
따라서, 각 팀은 상대 팀 진영에 최대한 일찍 도착하는 것이 유리합니다.

지금부터 당신은 한 팀의 팀원이 되어 게임을 진행하려고 합니다. 
다음은 5 x 5 크기의 맵에, 당신의 캐릭터가 (행: 1, 열: 1) 위치에 있고, 
상대 팀 진영은 (행: 5, 열: 5) 위치에 있는 경우의 예시입니다.

![1](https://user-images.githubusercontent.com/50406129/233122062-597b0a17-61cf-402c-bb15-7072244ab23b.png)

위 그림에서 검은색 부분은 벽으로 막혀있어 갈 수 없는 길이며, 
흰색 부분은 갈 수 있는 길입니다. 
캐릭터가 움직일 때는 동, 서, 남, 북 방향으로 한 칸씩 이동하며, 
게임 맵을 벗어난 길은 갈 수 없습니다.

아래 예시는 캐릭터가 상대 팀 진영으로 가는 두 가지 방법을 나타내고 있습니다.

첫 번째 방법은 11개의 칸을 지나서 상대 팀 진영에 도착했습니다.

![2](https://user-images.githubusercontent.com/50406129/233122294-1a07f3ce-6db9-42db-9240-fdc1c9fa83a6.PNG)

두 번째 방법은 15개의 칸을 지나서 상대 팀 진영에 도착했습니다.

![3](https://user-images.githubusercontent.com/50406129/233122440-e8341a01-d120-4f44-83e3-7ef678bf6adc.PNG)

위 예시에서는 첫 번째 방법보다 더 빠르게 상대 팀 진영에 도착하는 방법은 없으므로, 
이 방법이 상대 팀 진영으로 가는 가장 빠른 방법입니다.

만약, 상대 팀이 자신의 팀 진영 주위에 벽을 세워두었다면 
상대 팀 진영에 도착하지 못할 수도 있습니다. 
예를 들어, 다음과 같은 경우에 당신의 캐릭터는 상대 팀 진영에 도착할 수 없습니다.

![4](https://user-images.githubusercontent.com/50406129/233122676-69e49da0-a7c1-4542-9a35-42bab6382914.PNG)

게임 맵의 상태 maps가 매개변수로 주어질 때, 
캐릭터가 상대 팀 진영에 도착하기 위해서 
지나가야 하는 칸 개수의 최솟값을 return 하도록 solution 함수를 완성해 주세요. 
단, 상대 팀 진영에 도착할 수 없을 때는 -1을 return 해주세요.

#### 제한사항
- maps는 n x m 크기의 게임 맵의 상태가 들어있는 2차원 배열로, 
n과 m은 각각 1 이상 100 이하의 자연수입니다.
  - n과 m은 서로 같을 수도, 다를 수도 있지만, 
  n과 m이 모두 1인 경우는 입력으로 주어지지 않습니다.
- maps는 0과 1로만 이루어져 있으며, 0은 벽이 있는 자리, 
1은 벽이 없는 자리를 나타냅니다.
- 처음에 캐릭터는 게임 맵의 좌측 상단인 (1, 1) 위치에 있으며, 
상대방 진영은 게임 맵의 우측 하단인 (n, m) 위치에 있습니다.

#### 입출력 예

|maps|answer|
|---------------------------------|--------|
|[[1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 1, 1], [1, 1, 1, 0, 1], [0, 0, 0, 0, 1]]|11|
|[[1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 0, 1]]|-1|

#### 입출력 예 설명

입출력 예 #1: 주어진 데이터는 다음과 같습니다.

![5](https://user-images.githubusercontent.com/50406129/233123634-9b57a886-fa44-485e-9d9b-1b46d22c961c.PNG)

캐릭터가 적 팀의 진영까지 이동하는 가장 빠른 길은 다음 그림과 같습니다.

![6](https://user-images.githubusercontent.com/50406129/233123766-1bdf6e3e-a4d8-4cae-9de6-9368a520d346.PNG)

따라서 총 11칸을 캐릭터가 지나갔으므로 11을 return 하면 됩니다.

입출력 예 #2: 문제의 예시와 같으며, 상대 팀 진영에 도달할 방법이 없습니다. 
따라서 -1을 return 합니다.

#### 문제 풀이

주어진 맵의 (n, m) 위치에 도달할 수 있는지 없는지, 만약 도달할 수 있다면 최단 거리로 
도달했을 때의 총이동 거리는 몇인지를 구하는 문제입니다.

BFS를 사용하여 목표 지점까지의 모든 경우의 이동 거리를 구하고 그 중 최솟값을 찾으면 
그것이 목표 지점까지의 최단 거리입니다. 그리고 만약 값이 없다면, -1을 반환하면 됩니다.

다만 제한사항이 아래 제한사항을 지켜야 해서 코드가 약간 길어집니다.

1. `상-하-좌-우`로 움직일 수 있습니다.
   1. 아래 `moves = ((0, -1), (0, 1), (-1, 0), (1, 0))` 에서 moves의 원소는 각각 
   `좌`, `우`, `상`, `하` 로의 움직임을 표현합니다.
2. 움직인 뒤의 나의 위치가 맵을 벗어나면 안 됩니다. 이동 전에 이를 확인해야 합니다.
   1. 아래의 `is_between()` 함수가 해당 확인을 진행합니다.
3. 맵을 벗어나지 않았더라도 해당 인덱스의 요소가 `1`이 아니라면 그곳으로는 이동하지 못합니다.
   1. 아래의 `maps[new_col][new_row] == 1` 이 해당 확인을 진행합니다.

문제 풀이 코드는 아래와 같습니다.

```python
from collections import deque


def solution(maps):
    queue = deque()
    max_c, max_r = len(maps) - 1, len(maps[0]) - 1
    moves = (
        # 좌, 우
        (0, -1), (0, 1),
        # 상, 하
        (-1, 0), (1, 0)
    )

    queue.append((0, 0, 1))

    while queue:
        # 현재 위치를 나타내는 c, r
        c, r, distance = queue.popleft()

        # 현재 위치가 목표 지점이면 distance를 반환합니다.
        if c == max_c and r == max_r:
            return distance

        # 현재 위치가 목표 지점이 아니면 움직입니다.
        for move_c, move_r in moves:
            new_c, new_r = c + move_c, r + move_r

            # 여러 제약 조건을 확인합니다.
            if 0 <= new_c <= max_c \
                    and 0 <= new_r <= max_r \
                    and maps[new_c][new_r] == 1:
                # 같은 위치를 계속 중복으로 이동하지 않도록 요소의 값을 변경합니다. 
                # 이렇게 변경하면 위 maps[new_c][new_r] == 1 에서 
                # 중복 이동을 걸러낼 수 있습니다.
                maps[new_c][new_r] = distance + 1
                queue.append((new_c, new_r, distance + 1))

    return -1
```

## 출처

- 인접 행렬/인접 리스트
  - [그림 툴: 1](https://csacademy.com/app/graph_editor/)
  - [그림 툴: 2](https://app.diagrams.net/)
- DFS
  - [위키](https://ko.wikipedia.org/wiki/%EA%B9%8A%EC%9D%B4_%EC%9A%B0%EC%84%A0_%ED%83%90%EC%83%89)
  - [타켓 넘버: 프로그래머스](https://school.programmers.co.kr/learn/courses/30/lessons/43165?language=python3)
  - [네트워크: 프로그래머스](https://school.programmers.co.kr/learn/courses/30/lessons/43162)
- BFS
  - [위키: gif](https://commons.wikimedia.org/wiki/File:Breadth-First-Search-Algorithm.gif)
  - [네트워크: 프로그래머스](https://school.programmers.co.kr/learn/courses/30/lessons/43162)
  - [게임 맵 최단거리: 프로그래머스](https://school.programmers.co.kr/learn/courses/30/lessons/1844)

## 참고하면 좋은 영상

- [자료구조 알고리즘: Graph 검색 DFS, BFS 구현 in Java](https://www.youtube.com/watch?v=_hxFgg7TLZQ)
