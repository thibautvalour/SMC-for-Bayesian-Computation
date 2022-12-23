def init_Y():
    Y = []
    L = [[30,1], [23,1] ,[15,1], [10,1] ,[8,1],
         [5,2], [4,4], [3,13], [2,20], [1,282]]
    for k, m in L:
        for i in range(m):
            Y.append(k)
    return Y