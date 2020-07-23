import cv2
import numpy as np
import SimpleITK as sitk
import os
import prettytable as pt

slice_information = []
total_tumor_num = -1
start_slice_list = []
end_slice_list = []
total_tumor_size_list = []
tumor_no_size_per_slice_list = []


def mha2jpg(case_no):
    path = 'C:/Users/lrz/Desktop/jpg_label'
    os.mkdir(path + '/' + case_no)
    slices = sitk.ReadImage('C:/Users/lrz/Desktop/mha_label/data2_' + case_no + '_lesion_label.mha')
    slices_data = sitk.GetArrayFromImage(slices)
    for i in range(len(slices_data)):
        slices_data[i] *= 255
        cv2.imwrite('C:/Users/lrz/Desktop/jpg_label/' + case_no + '/' + str(i) + '.jpg', slices_data[i])
    print(len(slices_data))


'''
获取切片的信息，格式为三维：
第一维：每张切片
第二维：每个肿瘤
第三维：肿瘤信息：[所在切片（可删），该切片第n个肿瘤，肿瘤序号，定位信息]
'''
def collect_tumor_information(case_no, slice_num):   # Path
    global slice_information
    for slice_no in range(slice_num):
        img = cv2.imread('C:/Users/lrz/Desktop/jpg_label/' + case_no + '/' + str(slice_no) + '.jpg') # C:\Users\lrz\Desktop\jpg_label\0213
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inter_information = []
        if len(contours) != 0:
            for tumor_num in range(len(contours)):
                # draw_img = cv2.drawContours(img.copy(), contours, tumor_num, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(contours[tumor_num])  # [slice_num, [x, y], [x + w, y], [x, y + h], [x + w, y + h]]
                inter_information.append([slice_no, tumor_num, 0, x, y, w, h]) # [slice_no, tumor_num, tumor_no, location(4)]
            slice_information.append(inter_information)
        else:
            inter_information.append([slice_no, -1, -1, 0, 0, 0, 0])
            slice_information.append(inter_information)
    # print(slice_information)

'''
获取第一个和最后一个带有肿瘤的slice_no
返回值：first_slice：第一个带有肿瘤的切片；last_slice：第一个肿瘤消失的切片
'''
def first_and_last_slice(slice_num):
    global slice_information
    first_slice = 0
    for i in range(slice_num):
        if slice_information[i][0][1] != -1:
            first_slice = i
            break
    last_slice = 0
    for i in range(first_slice, slice_num):
        if slice_information[i][0][1] == -1:
            last_slice = i
            break
    return first_slice, last_slice

'''
将第一个有肿瘤切片中n个肿瘤标号：0,1,2,3...
'''
def first_tumor_no(first):
    global slice_information
    global total_tumor_num
    for i in range(slice_information[first][0][1] + 1):
        slice_information[first][i][2] = slice_information[first][i][1]
        total_tumor_num += 1


'''
判断两肿瘤重叠率。
input：两肿瘤位置信息
output：返回overlap / before
'''
def rec_overlap_rate(x0, y0, w0, h0, x1, y1, w1, h1):     # x0, y0, w0, h0, x1, y1, w1, h1
    black0 = np.zeros((512, 512),dtype=np.uint8)
    black1 = np.zeros((512, 512), dtype=np.uint8)
    white0_pixel_num = 0
    for i in range(x0, x0 + w0):
        for j in range(y0, y0 + h0):
            black0[i][j] = 255
            white0_pixel_num += 1
    for i in range(x1, x1 + w1):
        for j in range(y1, y1 + h1):
            black1[i][j] = 255
    overlap_pixel_num = 0
    for i in range(x0, x0 + w0):
        for j in range(y0, y0 + h0):
            if black0[i][j] == black1[i][j]:
                overlap_pixel_num += 1
    overlap_rate = overlap_pixel_num / white0_pixel_num
    return overlap_rate


'''
修改每个切片上每个肿瘤的编号
'''
def get_tumor_no(first, last):
    global slice_information
    global total_tumor_num
    for i in range(first + 1, last - 1):
        for j in range(len(slice_information[i + 1])):
            for k in range(len(slice_information[i])):
                if rec_overlap_rate(slice_information[i][k][3], slice_information[i][k][4], slice_information[i][k][5], slice_information[i][k][6], slice_information[i + 1][j][3], slice_information[i + 1][j][4], slice_information[i + 1][j][5], slice_information[i + 1][j][6]) > 0:
                    slice_information[i + 1][j][2] = slice_information[i][k][2]
                    break
            else:
                total_tumor_num += 1
                slice_information[i + 1][j][2] = total_tumor_num

'''
按标号给出每个肿瘤的起始和结束点
'''
def tumor_no_start_slice(slice_num):
    global total_tumor_num
    global start_slice_list
    global end_slice_list
    if total_tumor_num == -1:
        start_slice_list = [None]
        end_slice_list = [None]
    else:
        for i in range(total_tumor_num + 1):
            flag = False
            for j in range(slice_num):
                for k in range(len(slice_information[j])):
                    if slice_information[j][k][2] == i:
                        start_slice_list.append(j)
                        flag = True
                        break
                if flag:
                    break
        for i in range(total_tumor_num + 1):
            end = 0
            for j in range(start_slice_list[i], slice_num):
                for k in range(len(slice_information[j])):
                    if slice_information[j][k][2] == i:
                        end = j
            end_slice_list.append(end)


'''
计算外接矩阵内肿瘤大小
返回值：肿瘤所占像素点数量
'''
def size_in_rec(case_no, slice_no, x, y, w, h): # , x, y, w, h
    img = cv2.imread('C:/Users/lrz/Desktop/jpg_label/' + case_no + '/' + str(slice_no) + '.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    count = 0
    for i in range(y, y + h):
        for j in range(x, x + w):
            if binary[i][j] == 255:
                count += 1
    return count

def tumor_no_total_size(case_no, tumor_no):
    start = start_slice_list[tumor_no]
    end = end_slice_list[tumor_no] + 1
    size = 0
    for i in range(start, end):
        for j in range(len(slice_information[i])):
            if slice_information[i][j][2] == tumor_no:
                size += size_in_rec(case_no, i, slice_information[i][j][3],  slice_information[i][j][4], slice_information[i][j][5], slice_information[i][j][6])
    return size
'''
列表：按肿瘤标号获取肿瘤总体大小
'''
def get_total_tumor_size_list(case_no):
    global total_tumor_size_list
    if total_tumor_num == -1:
        total_tumor_size_list = [None]
    else:
        for i in range(total_tumor_num + 1):
            temp = tumor_no_total_size(case_no, i)
            total_tumor_size_list.append(temp)


def get_tumor_no_size_per_slice_list():
    global start_slice_list
    global end_slice_list
    global total_tumor_size_list
    global tumor_no_size_per_slice_list
    if total_tumor_num == -1:
        tumor_no_size_per_slice_list = [None]
    else:
        for i in range(total_tumor_num + 1):
            temp = total_tumor_size_list[i] / (end_slice_list[i] + 1 - start_slice_list[i])
            tumor_no_size_per_slice_list.append(temp)
    # print(tumor_no_size_per_slice_list)

# def get_tumor_no_location():
#     global tumor_no_location_list
#     if total_tumor_num == -1:
#         tumor_no_location_list = [None]
#     else:
#         for i in range(total_tumor_num + 1):
#             temp = []
#             for j in range(start_slice_list[i], end_slice_list[i] + 1):
#                 location = []
#                 for k in range(len(slice_information[j])):
#                     if slice_information[j][k][2] == i:
#                         location.append(j)
#                         location.append([(slice_information[j][k][3], slice_information[j][k][4]), (slice_information[j][k][3] + slice_information[j][k][5], slice_information[j][k][4] + slice_information[j][k][6])])
#                         temp.append(location)
#             tumor_no_location_list.append(temp)

stop = False

def find_else_tumor(start, end):
    global slice_information
    global stop
    else_first_slice = 0
    for i in range(start, end):
        stop = True
        if slice_information[i][0][1] != -1:
            else_first_slice = i
            stop = False
            break

    else_last_slice = 0
    for i in range(else_first_slice, slice_num):
        if slice_information[i][0][1] == -1:
            else_last_slice = i
            break
    return else_first_slice, else_last_slice

def else_first_tumor_no(else_first):
    global slice_information
    global total_tumor_num
    for i in range(len(slice_information[else_first])):
        if stop:
            break
        else:
            total_tumor_num += 1
            slice_information[else_first][i][2] = total_tumor_num






def show_in_table():
    tb = pt.PrettyTable()
    tb.field_names = ["tumor NO.", "tumor size", "slices with tumor", "tumor size per slice"]
    for i in range(total_tumor_num + 1):
        # print(i)
        # print(total_tumor_size_list[i])
        # print(end_slice_list[i] + 1 - start_slice_list[i])
        # print(tumor_no_size_per_slice_list[i])
        # print(tumor_no_location_list[i])
        tb.add_row([i, total_tumor_size_list[i], end_slice_list[i] + 1 - start_slice_list[i], tumor_no_size_per_slice_list[i]])
    print(tb)



if __name__ == '__main__':
    case_no = input('case_no:')
    mha2jpg(case_no)
    slice_num = int(input("slice_number:"))
    collect_tumor_information(case_no, slice_num)
    first, last = first_and_last_slice(slice_num)
    first_tumor_no(first)
    get_tumor_no(first, last)

    while not stop:
        first, last = find_else_tumor(last, slice_num)
        else_first_tumor_no(first)
        get_tumor_no(first - 1, last)

    tumor_no_start_slice(slice_num)
    get_total_tumor_size_list(case_no)
    get_tumor_no_size_per_slice_list()
    # print(total_tumor_size_list)
    # print(start_slice_list)
    # print(end_slice_list)
    # print(tumor_no_size_per_slice_list)
    show_in_table()