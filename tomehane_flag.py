import numpy as np
np.set_printoptions(threshold=np.inf)
s = input("読み込むデータは？：")
data = np.load(s+".npy")

###data準備###
#Z軸dataの準備・追加 -100のz軸を元データに追加#
kaku_all = data.shape
z = np.full(((kaku_all[0], 1000, 1)), -100)
data_z_add = np.append(data,z,axis=2)

###Z軸変更プログラム###
for line in range(kaku_all[0]):
    #lineのデータ数を取得
    c=0
    for i in range(1000):
        if data_z_add[line][i][0] != -100:
            c+=1

    #Z軸の値を変更
    #はらい flag=1
    #2/3で値を変更
    kind_line = input(str(line+1)+"画目はとめ？はらい？（とめ:0 はらい:1 を入力）：")
    k_line = int(kind_line)
    if k_line == 1:
        change_pt = int(2 / 3 * c)
        for i in range(c):
            if i <= change_pt:
                data_z_add[line][i][2] = 0
            else:
                data_z_add[line][i][2] = 1
    #とめ flag=2
    elif k_line == 0:
        end_pt = c-20
        for i in range(c):
            if i <= end_pt:
                data_z_add[line][i][2] = 0
            else:
                data_z_add[line][i][2] = 2
        for i in range(end_pt*(-1)+c):
            data_z_add[line][c+i][0] = data_z_add[line][c-i-1][0]
            data_z_add[line][c+i][1] = data_z_add[line][c-i-1][1]
            data_z_add[line][c+i][2] = 2
ans = s+"_plus_flag"
np.save(ans,data_z_add)