import numpy as np
np.set_printoptions(threshold=np.inf)
#input_data = input("読み込む文字データ：")
#data = np.load(input_data)
data = np.load("a_test.npy")

###data準備###
#Z軸dataの準備・追加 -100のz軸を元データに追加#
kaku_all = data.shape
z = np.full(((kaku_all[0], 1000, 1)), -100)
data_z_add = np.append(data,z,axis=2)

###Z軸変更プログラム###
change_line = input("変更したい画数は：")
line = int(change_line)
print(data_z_add[line][0])
print(data_z_add[line][0][0])

#lineのデータ数を取得
c=0
for i in range(1000):
    if data_z_add[line][i][0] != -100:
        c+=1
print(c)

#Z軸の値を変更
#2/3で値を変更
change_pt = int(2/3*c)
for i in range(c):
    if i <= change_pt:
        data_z_add[line][i][2] = 0
    else:
        data_z_add[line][i][2] = 1
print(data_z_add[line])