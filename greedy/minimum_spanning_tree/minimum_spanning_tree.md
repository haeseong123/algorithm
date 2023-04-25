# 최소 신장 트리

`최소 신장 트리(Minimum Spanning Tree)` 는 `신장 트리(Spanning Tree)`
중 가중치의 합이 최소인 신장 트리를 일컫는 말입니다.

`신장 트리(Spanning Tree)` 란 그래프에서 모든 정점에 대한 최소한의 연결만을 남긴 그래프입니다. 한 곳으로 도달하는 경우가 두 개 이상 존재하는 경우 즉, 사이클이 존재하는 경우 최소한의 연결이라 말할
수 없기 때문에 신장 트리에는 사이클이 포함되면 안 됩니다.

예를 들어 아래와 같은 그래프가 있을 때

![1](https://user-images.githubusercontent.com/50406129/234200227-f3704848-99a5-4293-9ba7-c26d5c8c0fe7.png)

여러 신장 트리가 나올 수 있고, 그중 아래와 같은 신장 트리도 나올 수 있습니다.

![2](https://user-images.githubusercontent.com/50406129/234200224-08a46966-42e7-4972-8c86-8193b0facd60.png)

많은 신장 트리 중 가중치의 합이 제일 작은 신장 트리(최소 신장 트리)는 아래와 같습니다.

![3](https://user-images.githubusercontent.com/50406129/234200225-56beb90a-2c3a-4404-8c2c-128c463e7e10.png)

이러한 최소 신장 트리를 구하는 대표적인 두 알고리즘으로 크러스컬 알고리즘(Kruskal's algorithm)과 프림 알고리즘(Prim's algorithm)이 있습니다. 우리는 이중 크러스컬 알고리즘에 대해 배울
것입니다.

## 크러스컬 알고리즘

크러스컬 알고리즘의 순서는 다음과 같습니다.

1. 그래프의 `각 정점이 별도의 트리인 F(트리 세트)`를 만듭니다.
2. 그래프의 `모든 간선을 포함하는 정렬된 집합 S`를 만듭니다.
3. S가 비거나 MST가 완성될 때까지 다음을 반복합니다.
    1. S에서 가중치가 최소인 간선을 제거합니다.
    2. 제거된 간선이 독립적인 두 개의 트리를 연결하는 경우 F에 두 트리를 추가하여 두 트리를 단일 트리로 결합합니다.

최소 신장 트리를 구하는 크러스컬 알고리즘은 간선의 가중치를 오름차순으로 정렬한 뒤, 간선을 순회하며 순간순간 최선의 선택을하고 한번한 선택은 두번 다시 바꾸지 않으므로 그리디 알고리즘을 활용한 알고리즘입니다.

크러스컬 알고리즘의 대부분은 쉽게 구현할 수 있겠는데 3-2번의 동작, 즉 제거된 간선이 독립적인 두 트리를 연결하는지 여부를 확인하는 것이 어렵습니다.
이는 `서로소 집합 자료 구조(disjoint-set data structure)` 를 사용하여 손쉽게 해결할 수 있습니다.

크러스컬 알고리즘을 구현하기 이전에 `서로소 집합 자료 구조` 에 대해 배워봅시다.

### 서로소 집합 자료 구조

서로소 집합(disjoint-set) 자료 구조, 또는 합집합-찾기(union-find) 자료 구조, 병합-찾기 집합(merge-find set)은 많은 `서로소 부분 집합을 저장하고 조작하는 자료 구조`입니다.
서로소 집합 자료 구조는 두 개의 유용한 연산을 제공합니다.

- Find(x): x가 속한 집합을 찾습니다. find(x)는 일반적으로 x가 속한 집합을 "대표" 하는 원소를 반환합니다.
    - x와 y가 같은 집합에 속한다면: find(x) == find(y)
    - x와 y가 같은 집합에 속하지 않는다면: find(x) != find(y)
- Union(x, y): 두 개의 집합을 하나의 집합으로 합칩니다.
- MakeSet(x): x가 들어가 있는 새로운 집합을 만듭니다.

![MakeSet과 Union 사용 결과](https://user-images.githubusercontent.com/50406129/234212093-783e00f6-ee49-4f6b-87e8-233b9ed507e2.PNG)

### Union-Find 사용 예시

Union-Find는 전체 집합이 주어지고 각 구성 원소들이 겹치지 않도록 해야할 때 자주 사용됩니다.

- 크러스컬 알고리즘에서 새로 추가할 간선의 양 끝 양끝 정점이 같은 집합에 속해 있는지 여부에 대해 검사하는 경우
- 어떤 사이트의 친구 관계가 생긴 순서대로 주어졌을 때, 가입한 두 사람의 친구 네트워크에 몇 명이 있는지 구하는 프로그램을 작성하는
  경우 ([친구 네트워크](https://www.acmicpc.net/problem/4195))

구현은 배열과 트리 두 가지로 구현할 수 있습니다. 이 둘의 차이를 알아보겠습니다.

#### 배열로 구현

배열로 구현하는 것은 매우 간단합니다. 배열을 하나 생성하고 요소의 수를 index로 사용하고 속한 집합을 실제 값으로 넣으면 됩니다.

만약, 숫자가 아니라 객체 형태의 원소라면 속한 집합의 번호를 저장하는 배열과 별도로 객체를 저장하는 배열을 만들어 상호 연동 시키는 등의 추가적인 작업을 수행하면 됩니다.

숫자로만 이루어진 원소를 갖는 경우를 생각해 봅시다. 1부터 9까지의 숫자가 주어졌을 때 모든 수에대해 makeSet 연산을 진행하면 아래와 같은 배열이 만들어집니다.

![11](https://user-images.githubusercontent.com/50406129/234243789-e3f4ed5e-1bed-43ae-9e6e-f45b5fdbd8a3.PNG)

여기서 집합1과 집합2를 하나의 집합으로 구성하고 싶다면 다시 말해, union 연산을 하고 싶다면, 배열의 모든 원소를 뒤져서 num 값이 2인 원소를 1로 바꿔줘야 합니다. 따라서 배열로
구현한 `union 연산의 시간 복잡도는 O(N)`입니다.

마지막으로, `find 연산은` index를 통해 접근하므로 `O(1)`의 시간 복잡도를 갖습니다.

#### 트리로 구현

트리로 구현했을 때의 구성을 나타내면 아래와 같이 나타낼 수 있습니다.

![eee](https://user-images.githubusercontent.com/50406129/234245417-ba47e01b-2bb6-4e4c-932a-9f2f0dab97ea.png)

빨간색은 해당 집합을 대표하는 노드입니다. 그 아래의 자식 노드들은 빨간색 루트 노드에 대한 포인터를 갖거나 재귀적으로 연결되어 최상위 루트 노드를 가리키도록 구현되어 있습니다.

union(x, y)를 실행하면 집합 x와 집합 y의 루트 노드를 찾고`(find(x), find(y))`
두 루트 노드가 다르면 집합 y를 집합 x의 자손으로 넣어 두 트리를 합칩니다. 트리를 합치는 것은 집합 y의 루트 노드를 집합 x의 노드에 연결만 하면 되므로 O(1) 입니다. 따라서 전체 union 연산의 속도는
`O(find(x)의 시간 복잡도 + find(y)의 시간 복잡도) = O(find()의 시간 복잡도)` 입니다.

find() 연산은 해당 노드서부터 루트 노드까지 쭉 올라가는 연산이므로 만약 아래와 같이 한쪽으로 쏠려있으면, 시간 복잡도가 O(N)이 되지만, 트리가 한쪽으로 치우쳐져 있지 않다면 O(N) 보다 빠릅니다. 따라서
트리의 쏠림 정도에 따라 시간 복잡도가 달라집니다. 한마디로 트리의 높이와 시간 복잡도가 비례합니다.

![wwww](https://user-images.githubusercontent.com/50406129/234254878-720f0097-4a90-432a-a700-47ab9d98e8be.PNG)

트리로 구현하는 코드는 다음과 같습니다.

```python
class DisjointSet:
    def __init__(self, size):
        self.tree = [i for i in range(size)]

    # 재귀로 자신의 root 노드를 찾습니다.
    def find(self, x):
        if self.tree[x] == x:
            return x
        else:
            return self.find(self.tree[x])

    # find를 이용하여 루트 노드를 찾고
    # 루트 노드가 서로 다르면 집합 y를 집합 x에 합칩니다.
    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        self.tree[y] = x
```

아래 사진은 초기화부터 여러 번의 Union-Find 수행을 나타냅니다.

![aa](https://user-images.githubusercontent.com/50406129/234249061-d5acba15-fd6c-4c3e-a2fb-d6c290114251.PNG)

위 코드는 트리 구조가 한쪽으로 완전히 쏠리는 경우를 방지할 수 없습니다. 트리 구조가 한쪽으로 완전히 쏠린다면 find()가 O(N)의 시간 복잡도를 갖게 되므로 union()도 O(N) 시간 복잡도를 갖게
됩니다. 배열로 구현했을 때 find, union이 각각 O(1), O(N) 시간 복잡도였다는 것을 생각해보면, 트리가 한쪽으로 쏠렸을 때는 트리로 구현하는 것의 장점이 없어지는 것입니다.

최적화를 통해 이를 해결해 봅시다.

우선 find 연산 최적화는 find()를 수행하며 만나는 모든 노드의 부모 노드를 root 노드로 변경하는 것입니다. 이를 경로 압축(Path Compression) 기법이라 부르며, 그림으로 나타내면 다음과
같습니다.

![qq](https://user-images.githubusercontent.com/50406129/234256700-b8c7cabd-36dd-46c7-a4a2-d31981746b8e.PNG)

코드는 다음과 같습니다.

```python
def find(self, x):
    if self.tree[x] == x:
        return x
    else:
        self.tree[x] = self.find(self.tree[x])
        return self.tree[x]
```

union 최적화는 rank에 트리의 높이를 저장하고, 항상 높이가 더 낮은 트리를 높은 트리 밑에 넣는 기법입니다. 이를 union-by-rank 혹은 union-by-height라고 부릅니다.

코드는 다음과 같습니다.

```python
def union(self, x, y):
    x = self.find(x)
    y = self.find(y)

    # 두 값의 root 가 같으면 
    # 이미 같은 집합이라는 것이므로
    # 합치지 않아도 됩니다.
    if x == y:
        return

    # 높이가 높은 트리에 높이가 낮은 트리를 붙입니다.
    # 자신보다 높이가 낮은 트리를 붙이므로 
    # 높이는 변하지 않습니다.

    # 단, 둘의 높이가 같다면 붙였을 때 
    # 높이가 1 증가하므로 높이를 +1 해줍니다.
    if self.rank[x] < self.rank[y]:
        self.tree[x] = y
    elif self.rank[x] > self.rank[y]:
        self.tree[y] = x
    else:
        self.tree[y] = self.tree[x]
        self.rank[x] += 1
```

최적화된 find와 최적화된 union의 전체 코드는 아래와 같습니다.

```python
class DisjointSet:
    def __init__(self, size):
        self.tree = [i for i in range(size)]
        self.rank = [0 for _ in range(size)]

    def find(self, x):
        if self.tree[x] == x:
            return x
        else:
            self.tree[x] = self.find(self.tree[x])
            return self.tree[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.rank[x] < self.rank[y]:
            self.tree[x] = y
        elif self.rank[x] > self.rank[y]:
            self.tree[y] = x
        else:
            self.tree[y] = x
            self.rank[x] += 1
```

이러한 최적화를 거친 find 연산과 union 연산의 시간 복잡도는 `O(a(N))` 으로, a(N)은 극도로 빠르게 성장하는 아커만 함수의 역함수를 뜻합니다. 이 a(N)은 현실적인 모든 입력에 대해 5보다
작으며, 따라서 최적화를 거친 find 연산과 union 연산은 사실상 상수 시간에 동작하는 시간 복잡도를 갖습니다.

아래는 disjoint-set 관련 문제입니다.

- [집합의 표현](https://www.acmicpc.net/problem/1717)
- [여행 가자](https://www.acmicpc.net/problem/1976)
- [친구 네트워크](https://www.acmicpc.net/problem/4195)

***

크러스컬 알고리즘을 작성하기 위한 disjoint-set 자료구조와 Union-Find를 배웠으니 크러스컬 알고리즘을 작성해 봅시다. 앞서 순서를 작성했지만 크러스컬 알고리즘의 순서는 아래와 같습니다.

1. 그래프의 `각 정점이 별도의 트리인 F(트리 세트)`를 만듭니다.
2. 그래프의 `모든 간선을 포함하는 정렬된 집합 S`를 만듭니다.
3. S가 비거나 MST가 완성될 때까지 다음을 반복합니다.
    1. S에서 가중치가 최소인 간선을 제거합니다.
    2. 제거된 간선이 독립적인 두 개의 트리를 연결하는 경우 F에 두 트리를 추가하여 두 트리를 단일 트리로 결합합니다.

이를 코드로 구현하면 아래와 같습니다.

```python
class DisjointSet:
    def __init__(self, size):
        self.tree = [i for i in range(size)]
        self.rank = [0 for _ in range(size)]

    def find(self, x):
        if self.tree[x] == x:
            return x
        else:
            self.tree[x] = self.find(self.tree[x])
            return self.tree[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.rank[x] < self.rank[y]:
            self.tree[x] = y
        elif self.rank[x] > self.rank[y]:
            self.tree[y] = x
        else:
            self.tree[y] = x
            self.rank[x] += 1


# edges = [(vertex1, vertex2, cost), ... ]
def solution(edges, size):
    disjoint_set = DisjointSet(size)
    edges.sort(key=lambda x: x[2])

    mst_cost = 0
    mst_edge_cnt = 0

    for u, v, w in edges:
        pu, pv = disjoint_set.find(u), disjoint_set.find(v)
        if pu == pv:
            continue

        disjoint_set.union(pu, pv)
        mst_cost += w
        mst_cost += 1

        # 최소 신장 트리의 간선은 항상 '정점의 개수 - 1'입니다.
        # 따라서 msg_edge_cnt가 size - 1이라는 것은
        # MST가 완성되었다는 것을 의미합니다.
        if mst_edge_cnt == size - 1:
            break

    return mst_cost
```

아래는 크러스컬 알고리즘 활용 문제입니다. 꼭 풀어보세요.

[1251. [S/W 문제해결 응용] 4일차 - 하나로](https://swexpertacademy.com/main/solvingProblem/solvingProblem.do)

<details>
<summary>파이썬 풀이 코드</summary>

```python
T = int(input())


class DisjointSet:
    def __init__(self, size):
        # tree = [부모 노드의 인덱스, 현재 노드의 높이]
        self.tree = [[i, 0] for i in range(size)]

    def find(self, x):
        if self.tree[x][0] == x:
            return x
        else:
            # 경로 압축
            self.tree[x][0] = self.find(self.tree[x][0])
            return self.tree[x][0]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        x_height = self.tree[x][1]
        y_height = self.tree[y][1]

        # union-by-height
        if x_height < y_height:
            self.tree[x][0] = y
        elif x_height > y_height:
            self.tree[y][0] = x
        else:
            self.tree[y][0] = x
            self.tree[x][1] += 1


def get_cost(pos1, pos2, tex_rate):
    return ((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2) * tex_rate


for test_case in range(1, T + 1):
    n = int(input())  # 총 정점의 개수입니다.
    xs = list(map(int, input().split()))  # 정점의 x 좌표 입니다.
    ys = list(map(int, input().split()))  # 정점의 y 좌표 입니다.
    positions = [[x, y] for x, y in zip(xs, ys)]  # 정점의 x,y 좌표입니다.
    disjoint_set = DisjointSet(n)  # Union-Find 연산을 위한 disjoint_set입니다.
    tex_rate = float(input())  # 세율입니다.
    edges = []  # 간선 정보를 담습니다. [(비용, 정점1을 나타내는 인덱스, 정점2를 나타내는 인덱스), ...]

    # 간선 정보를 채우기 위한 반복문입니다.
    for i in range(n - 1):
        for j in range(i + 1, n):
            edges.append((
                get_cost(positions[i], positions[j], tex_rate),
                i,
                j
            ))

    # 크러스컬 알고리즘 ...
    edges.sort(key=lambda x: x[0])
    answer = 0
    count = 0
    for (cost, i, j) in edges:
        if disjoint_set.find(i) == disjoint_set.find(j):
            continue

        disjoint_set.union(i, j)
        answer += cost
        count += 1

        if count == n - 1:
            break

    # 소수 첫째 자리에서 반올림을 합니다.
    answer = int(answer + 0.5)

    # 출력
    print(f"#{test_case} {answer}")
```

</details>

## 출처

- [알고리즘 - 크루스칼 알고리즘(Kruskal Algorithm), 최소 신장 트리(MST)](https://chanhuiseok.github.io/posts/algo-33/)
- [Kruskal's algorithm](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm)
- [서로소 집합 자료 구조](https://ko.wikipedia.org/wiki/%EC%84%9C%EB%A1%9C%EC%86%8C_%EC%A7%91%ED%95%A9_%EC%9E%90%EB%A3%8C_%EA%B5%AC%EC%A1%B0)
- [[알고리즘] Union-Find 알고리즘](https://gmlwjd9405.github.io/2018/08/31/algorithm-union-find.html)
- [서로소 집합(Disjoint Set) & 유니온 파인드(Union find)](https://yoongrammer.tistory.com/102)
- [[자료구조]Union-Find: Disjoint Set의 표현](https://bowbowbow.tistory.com/26)
- [1251. [S/W 문제해결 응용] 4일차 - 하나로](https://swexpertacademy.com/main/solvingProblem/solvingProblem.do)