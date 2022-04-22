def startwith(start: int, mgraph: list):
    passed = [start]
    nopass = [x for x in range(len(mgraph)) if x != start]
    dis = mgraph[start]
    path_1 = [[0]] * len(mgraph)
    while len(nopass):
        idx = nopass[0]
        for i in nopass:
            if dis[i] < dis[idx]:
                idx = i
        print(idx)
        a = path_1[idx]
        path_1[idx] = a + [idx]
        print(path_1)
        nopass.remove(idx)
        passed.append(idx)
        for i in nopass:
            if dis[idx] + mgraph[idx][i] < dis[i]:
                dis[i] = dis[idx] + mgraph[idx][i]
                path_1[i] = path_1[idx]
                print(path_1)
    return [dis, path_1]


if __name__ == "__main__":
    inf = 10086
    mgraph = [[0, 13, inf, inf, inf, inf, inf, 18, 15],
              [13, 0, 17, inf, inf, inf, inf, inf, inf],
              [inf, 17, 0, 18, inf, inf, inf, inf, inf],
              [inf, inf, 18, 0, 10, inf, inf, inf, 13],
              [inf, inf, inf, 10, 0, 15, inf, inf, inf],
              [inf, inf, inf, 20, 15, 0, 18, 10, inf],
              [inf, inf, inf, inf, inf, 18, 0, 7, inf],
              [18, inf, inf, inf, inf, 10, 7, 0, 15],
              [15, inf, inf, 13, inf, inf, inf, 15, 0]]
    backs = startwith(0, mgraph)
    dis = backs[0]
    path_1 = backs[1]
    path_2 = [i + 1 for i in path_1[4]]
    print(f'节点1到5的路径为：{path_2}，距离为{dis[4]}')
