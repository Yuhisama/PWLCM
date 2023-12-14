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
test = PWLCM(u11, p11, 10)