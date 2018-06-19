with open("R_matrix_Luyu_20180619.txt", "r") as f:
    while True:
        t = f.readline()
        if not t:
            break
        print(t, end="")
