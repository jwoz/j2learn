def matrix_product(a, b):
    try:
        if len(a[0]) != len(b):
            print('not equal')
    except IndexError:
        raise

    assert len(a[0]) == len(b)
    m = []
    for i in range(len(a)):
        r = []
        for k in range(len(b[0])):
            this_sum = 0
            for j in range(len(a[0])):
                this_sum += a[i][j] * b[j][k]
            r.append(this_sum)
        m.append(r)
    assert len(m) == len(a) and len(m[0]) == len(b[0])
    return m
