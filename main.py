import os
import hashlib
import numpy as np
from PIL import Image
import math
image_path = "image/dog512.png"
# tool funtions
def calculate_sha256():
    # message_bytes = message.encode('utf-8')
    with open(image_path, 'rb') as file:
        image_bytes = file.read()
        sha256_hash = hashlib.sha256(image_bytes)
        hash_result = sha256_hash.hexdigest()
    return hash_result
def change_image_size(image_path, image_size, rename):
    img = Image.open(image_path)
    resize_image = img.resize((image_size, image_size))
    resize_image.save(f'image/{rename}.png')
    print(f"Finished resize {rename}")
def show_image(image_array):
    test = Image.fromarray(image_array)
    test.show()
# for step's function
def step4tostep6(V): # 因為step4到stpe6有重複所以在此做一個function
    # step 4
    B1, B2 = np.hsplit(V, 2)
    b1 = np.fliplr(B1)
    b2 = np.fliplr(B2)
    # step 5
    F = np.hstack((b1, b2))
    F_up, F_down = np.vsplit(F, 2)
    # 原本是分成四塊，分批上下反轉，但其實分成兩塊翻轉就好，所以直接跳到第六步，然後結合 np.flipud(F_up), np.flipud(F_down)
    # step 6
    G = np.vstack((np.flipud(F_up), np.flipud(F_down)))
    D = list(step6_cut_16(G)) # D1~D16
    d = [np.rot90(i,2) for i in D]
    # step 7
    H = np.vstack([
        np.hstack(d[:4]),
        np.hstack(d[4:8]),
        np.hstack(d[8:12]),
        np.hstack(d[12:])
    ])
    return H
def step6_cut_16(np_array): # step6's function
    vs_image = np.vsplit(np_array,4) # 切4等分，垂直等分
    y = []
    for i in vs_image:
        hs = np.hsplit(i,4) # 上面垂直切4等分的，每一段在切4等分，就有16等分了
        y.append(hs[0])
        y.append(hs[1])
        y.append(hs[2])
        y.append(hs[3])
    for i in y:
        yield i
def PWLCM_system(u , p):
    if 0 <= u and u < p:
        return u / p
    if p <= u and u < 0.5:       
        return (u-p) / (0.5-p)
    if 0.5 <= u and u < 1:
        return (1-u)
    
def PWLCM(u_init,p, iterations):
    final_result = [u_init]

    for _ in range(iterations+200):
        result = PWLCM_system(final_result[-1], p)
        final_result.append(result)

    final_result = final_result[201:]
    return final_result
def sort_list(deal_list): # sort tools output => (value, 原始陣列的位置)
    dl_sort = sorted(deal_list) # dl_sort is deal_list_sort
    for i in range(len(deal_list)):
        yield dl_sort[i], deal_list.index(dl_sort[i])
def step11_switch(sorted_list, image_array): # step 11 switch
    image_array_copy = image_array.copy()
    M = len(image_array)
    N = len(image_array[0])
    for i in range(M):
        if sorted_list[i][1] % 2 == 0:
            shift_value = int((sorted_list[i][0] * (2**12)) % N)
            image_array_copy[i] = np.roll(image_array[i], shift = shift_value)
        else:
            shift_value = int((sorted_list[i][0] * (2**12)) % N)
            image_array_copy[i] = np.roll(image_array[i], shift = -(shift_value))
    return image_array_copy
def step16_row_diffusion(image_array, sorted_list): # step 16 use diffusion
    image_array_copy = image_array.copy()
    M = len(image_array)
    N = len(image_array[0])
    for i in range(M):
        if i ==0:    
            for j in range(N):
                image_array_copy[i][j] = image_array[i][j] ^ (sorted_list[j][1])
                
        else:
            for j in range(N):
                image_array_copy[i][j] = image_array[i][j] ^ (image_array_copy[i-1][j])
    return image_array_copy
def step19_column_diffusion(image_array, sorted_index): # stpe 18 use diffusion
    image_array_copy = image_array.copy()
    M = len(image_array)
    N = len(image_array[0])
    for i in range(N):
        if i ==0:    
            for j in range(M):
                image_array_copy[j][i] = image_array[j][i] ^ (sorted_index[j])
                
        else:
            for j in range(M):
                image_array_copy[j][i] = image_array[j][i] ^ (image_array_copy[j][i-1])
    return image_array_copy
def generat_block4x4(sorted_list_value): # step 20 use function
    sorted_list_value_copy = sorted_list_value.copy()
    for i in range(len(sorted_list_value)):
        if sorted_list_value[i] > 0.25:
            sorted_list_value_copy[i] = 1
        else:
            sorted_list_value_copy[i] = 0
    s = []
    for j in range(16):
        s.append(sorted_list_value_copy[8*j:8*j+8])
    def list_to_str(list_value):
        test = ''
        for i in range(8):
            test += str(list_value[i])
        return test
    for i in range(len(s)):
        s[i] = int(list_to_str(s[i]),2)
    return s
def change_array_block4x4(array): # step21 use fuction
    # 獲取數組的形狀（shape）
    height, width = array.shape
    # 定義子數組的大小 (4x4)
    subarray_height = 4
    subarray_width = 4
    # 創建一個空列表，用於存放子數組
    subarrays = []
    # 使用嵌套迴圈遍歷原始數組，將其劃分為 4x4 的子數組
    for i in range(0, height, subarray_height):
        for j in range(0, width, subarray_width):
            subarray = array[i:i+subarray_height, j:j+subarray_width]
            subarrays.append(subarray)
    return subarrays
# parameters
p1,p2,p3,p4,p5 = 0.352831074644317, 0.364731890162837, 0.371982540643799, 0.382137610925881, 0.391772603534118
u1,u2,u3,u4,u5 = 0.254738905481226, 0.261129543860916, 0.274521866927734, 0.289213722445016, 0.299768932581012
hash = calculate_sha256()
# step 1 open image 
M, N = 512, 512
image = Image.open(image_path)
# step 2
red_channel, green_channel, blue_chnnel = image.split()
R = np.array(red_channel)
G = np.array(green_channel)
B = np.array(blue_chnnel)
# step 3
V1 = np.vstack((R, G, B))
H = step4tostep6(V1)
R1, G1, B1 = np.vsplit(H, 3)
# step 8
V2 = np.hstack((R1, G1, B1))
# step 9
H1 = step4tostep6(V2)
# step 10
p11 = abs(p1 - ((sum(int(i, 16) for i in hash[0:6])) - (math.ceil(sum(int(i, 16) for i in hash[0:6])/(10**15)))) * 0.01)
u11 = abs(u1 - ((sum(int(i, 16) for i in hash[6:12])) - (math.ceil(sum(int(i, 16) for i in hash[6:12])/(10**15)))) * 0.01)
u11_squence = PWLCM(u11, p11, M)
sort_u11_squence = list(sort_list(u11_squence))
# step 11
RS1 = step11_switch(sort_u11_squence, H1)
# steo 12
R2, G2, B2 = np.hsplit(RS1,3)
V3 = np.vstack((R2,G2,B2))
# step 13
p21 = abs(p2 - ((sum(int(i, 16) for i in hash[12:18])) - (math.ceil(sum(int(i, 16) for i in hash[12:18])/(10**15)))) * 0.01)
u21 = abs(u2 - ((sum(int(i, 16) for i in hash[18:24])) - (math.ceil(sum(int(i, 16) for i in hash[18:24])/(10**15)))) * 0.01)
u21_squence = PWLCM(u21, p21, N)
sort_u21_squence = list(sort_list(u21_squence))
# step 14
rotv3 = np.rot90(V3, -1) # 因為up down的shift 和 右轉90度的shift 應該是一樣的，所以在這裡就先右轉90度，然後等shift好在轉回去就好
rotv3shift = step11_switch(sort_u21_squence, rotv3) # output是左右shift完後的，所以要在rot回去
CS1 = np.rot90(rotv3shift, 1)
# step 15
p31 = abs(p3 - ((sum(int(i, 16) for i in hash[24:30])) - (math.ceil(sum(int(i, 16) for i in hash[24:30])/(10**15)))) * 0.01)
u31 = abs(u3 - ((sum(int(i, 16) for i in hash[30:36])) - (math.ceil(sum(int(i, 16) for i in hash[30:36])/(10**15)))) * 0.01)
u31_squence = PWLCM(u31, p31, N)
sort_u31_squence = list(sort_list(u31_squence))
# step 16
RD1 = step16_row_diffusion(CS1, sort_u31_squence)
# step 17
R3 , G3 , B3 = np.vsplit(RD1,3)
V4 = np.hstack((R3,G3,B3))
# step 18
p41 = abs(p4 - ((sum(int(i, 16) for i in hash[36:42])) - (math.ceil(sum(int(i, 16) for i in hash[36:42])/(10**15)))) * 0.01)
u41 = abs(u4 - ((sum(int(i, 16) for i in hash[42:48])) - (math.ceil(sum(int(i, 16) for i in hash[42:48])/(10**15)))) * 0.01)
u41_squence = PWLCM(u41, p41, M)
sort_u41_squence = list(sort_list(u41_squence))
sort_u41_squence_index =[sort_u41_squence[i][1] for i in range(len(sort_u41_squence))] # 先取得 u41_index
# step 19
CD1 = step19_column_diffusion(V4, sort_u41_squence_index)
# step 20 
p51 = abs(p5 - ((sum(int(i, 16) for i in hash[36:42])) - (math.ceil(sum(int(i, 16) for i in hash[36:42])/(10**15)))) * 0.01)
u51 = abs(u5 - ((sum(int(i, 16) for i in hash[42:48])) - (math.ceil(sum(int(i, 16) for i in hash[42:48])/(10**15)))) * 0.01)
u51_squence = PWLCM(u51, p51, (8*(4*4)))
k_1d = np.array(generat_block4x4(u51_squence))
k = k_1d.reshape((4,4))
# stpe 21
BB = change_array_block4x4(CD1)
# step 22
def step22_4x4_diffusion(K, BB):
    def test(array1, array2):
        array_copy = array1.copy()
        for i in range(4):
            for j in range(4):
                array_copy[i][j] = array1[i][j] ^ array2[i][j]
        return array_copy
    BB_copy = BB.copy()
    for i in range(len(BB)):
        if i == 0:
            BB_copy[i] = test(BB[i], K)
        else:
            BB_copy[i] = test(BB[i], BB[i-1])
    return BB_copy
BD = step22_4x4_diffusion(k, BB)
def reconstructed_array_block(BD):
    # 獲取子數組的大小
    subarray_height, subarray_width = 4,4
    # 獲取原始數組的大小
    height = 512
    width = 1536
    # 初始化大的 NumPy 數組
    reconstructed_array = np.zeros((height, width))
    # 使用嵌套迴圈將子數組放入大的數組中
    for i in range(0, height, subarray_height):
        for j in range(0, width, subarray_width):
            subarray = BD.pop(0)  # 取出列表中的第一個子數組
            reconstructed_array[i:i+subarray_height, j:j+subarray_width] = subarray
    return reconstructed_array
BD1 = reconstructed_array_block(BD)
R4 , G4, B4 = np.hsplit(BD1,3)
test = np.stack([R4, G4, B4], axis=-1)
test_image = Image.fromarray(np.uint8(test))
test_image.show()