import numpy as np
np.set_printoptions(threshold=np.inf)
data = np.load("a_test.npy")

###data準備###
#Z軸dataの準備・追加 -100のz軸を元データに追加#
kaku_all = data.shape
z = np.full(((kaku_all[0], 1000, 1)), -100)
data_z_add = np.append(data,z,axis=2)

###Z軸変更プログラム###
change_line = input("変更したい画数は：")
kind_line = input("とめ:0 はらい:1 を入力：")
line = int(change_line)
k_line = int(kind_line)

#lineのデータ数を取得
c=0
for i in range(1000):
    if data_z_add[line][i][0] != -100:
        c+=1

#Z軸の値を変更

#はらい
#2/3で値を変更
if k_line == 1:
    change_pt = int(2 / 3 * c)
    for i in range(c):
        if i <= change_pt:
            data_z_add[line][i][2] = 0
        else:
            data_z_add[line][i][2] = 1
#とめ
elif k_line == 0:
    end_pt = 20
    for i in range(c+end_pt):
        data_z_add[line][i][2] = 0
    for i in range(end_pt):
        data_z_add[line][c+i][0] = data_z_add[line][c-i-1][0]
        data_z_add[line][c+i][1] = data_z_add[line][c-i-1][1]
print(data_z_add[line])