import cv2
import copy
import numpy

def SavePic(Img, Path):
    cv2.imwrite(Path, Img)
    print("[I] Save at: ", Path)

def GetAllBlock(img, height, width, patchSize, step):
    block_ls = []
    row_idx_ls = []   # 记录block的左上角起始像素的索引
    col_idx_ls = []

    for i in range(0, height - patchSize + 1, step):    # 最后一个block的起始索引: height - patchSize
        row_idx_ls.append(i)
    
    for j in range(0, width - patchSize + 1, step):
        col_idx_ls.append(j)
    
    for i in range(len(row_idx_ls)):
        for j in range(len(col_idx_ls)):
            tmp = copy.deepcopy(img[row_idx_ls[i]:row_idx_ls[i]+patchSize, 
                                    col_idx_ls[j]:col_idx_ls[j]+patchSize])
            block_ls.append(tmp)
    return row_idx_ls, col_idx_ls, block_ls


def cal_distance(a, b):
    sum = ((a - b) ** 2).mean()
    return sum

def getSimilarPatch(block_ls, i, j, blockRowNum, blockColNum,
                    searchArea, patchSize, step, maxMatch, threshold):
    '''搜寻与第i行第j列的block相似的block'''
    sim_patch = []
    sim_idx = []

    blockNumInsideSearchArea = int((searchArea - patchSize) / step) + 1   # 搜寻面积searchArea内有多少个block

    row_min = max(0, i - (blockNumInsideSearchArea - 1) / 2)
    row_max = min(blockRowNum - 1, i + (blockNumInsideSearchArea - 1) / 2)
    row_length = int(row_max - row_min + 1)

    col_min = max(0, j - (blockNumInsideSearchArea - 1) / 2)
    col_max = min(blockColNum - 1, j + (blockNumInsideSearchArea - 1) / 2)
    col_length = int(col_max - col_min + 1)

    # print('[Debug] Block number in search area: ', row_length * col_length)

    current = block_ls[i * blockColNum + j]
    distance = numpy.zeros((row_length * col_length))  # 记录searchArea内的block与当前block的相似度
    idx = numpy.zeros((row_length * col_length)).astype(numpy.uint8)  # 记录searchArea内的block的索引
    for p in range(row_length):
        for q in range(col_length):
            tmp = block_ls[int((p + row_min) * blockColNum + (q + col_min))]
            distance[p * col_length + q] = cal_distance(current, tmp)
            idx[p * col_length + q] = p * col_length + q
    
    # distance 升序排列，idx保存排序后每个元素在原数组中的索引
    # 与前面比较，若小于前面的distance，一直往前移动

    # print("[Debug] distance before sort: ", distance)
    # print("[Debug] idx before sort: ", idx)

    for k in range(1, row_length * col_length):
        value = distance[k]
        l = k - 1
        while value < distance[l] and l >= 0:
            distance[l + 1] = distance[l]
            idx[l + 1] = idx[l]
            l = l - 1
        distance[l + 1] = value
        idx[l + 1] = k

    # print("[Debug] distance after sort: ", distance)
    # print("[Debug] idx after sort: ", idx)

    selectedNum = maxMatch
    while row_length * col_length < selectedNum:
        selectedNum /= 2
    while distance[int(selectedNum - 1)] > threshold:
        selectedNum /= 2

    for k in range(int(selectedNum)):
        Row = int(row_min + idx[k] / col_length)
        Col = int(col_min + idx[k] % col_length)
        tmp = copy.deepcopy(block_ls[Row * blockColNum + Col])
        sim_patch.append(tmp)
        sim_idx.append(Row * blockColNum + Col)
    return sim_patch, sim_idx


def wavedec(signal):
    length = len(signal)
    N = int(numpy.log2(length))

    for _ in range(N):
        tmp = numpy.copy(signal)
        for k in range(length // 2):
            signal[k] = (tmp[2 * k] + tmp[2 * k + 1]) / numpy.sqrt(2)
            signal[k + length // 2] = (tmp[2 * k] - tmp[2 * k + 1]) / numpy.sqrt(2)
        length = length // 2

    return signal

def waverec(signal, length, N):
    for _ in range(N):
        tmp = numpy.copy(signal)
        for k in range(length // 2):
            signal[2 * k] = (tmp[k] + tmp[k + length // 2]) / numpy.sqrt(2)
            signal[2 * k + 1] = (tmp[k] - tmp[k + length // 2]) / numpy.sqrt(2)
        length = length * 2
    return signal

def Tran1D(sim_patch, patchSize):
    '''一维离散Haar小波变换'''
    size = len(sim_patch)
    data = numpy.zeros((size))
    for i in range(patchSize):
        for j in range(patchSize):
            for k in range(size):
                data[k] = sim_patch[k][i, j]
            data = wavedec(data)
            for k in range(size):
                sim_patch[k][i, j] = data[k]
    return sim_patch

def DetectZero(input, threshold):
    for k in range(len(input)):
        for i in range(input[k].shape[0]):
            for j in range(input[k].shape[1]):
                if numpy.fabs(input[k][i, j]) < threshold:
                    input[k][i, j] = 0
    return input

def calculate_weight_ht(input, sigma):
    num = 0
    for k in range(len(input)):
        for i in range(input[k].shape[0]):
            for j in range(input[k].shape[1]):
                if input[k][i, j] != 0:
                    num += 1
    # print("[Debug] 3d filter keep {}/{}".format(num, len(input)*input[k].shape[0]*input[k].shape[1]))
    if num == 0:
        return 1
    else:
        return 1.0 / (sigma * sigma * num)

def aggregation(numerator_ht, denominator_ht, sim_idx_row, sim_idx_col, sim_patch, weight, patchSize, window):
    for k in range(len(sim_patch)):
        x = sim_idx_row[k]
        y = sim_idx_col[k]
        numerator_ht[x:x + patchSize, y:y + patchSize] = numerator_ht[x:x + patchSize, y:y + patchSize] + \
                                                     weight * (sim_patch[k] * window)
        denominator_ht[x:x + patchSize, y:y + patchSize] = denominator_ht[x:x + patchSize, y:y + patchSize] + \
                                                         weight * window

    return numerator_ht, denominator_ht

def hamming_window(N):
    n = numpy.arange(0, N)
    hamming1d =  0.54 - 0.46 * numpy.cos(2 * numpy.pi * n / (N - 1))
    hamming2d = numpy.outer(hamming1d, hamming1d)
    return hamming2d

def Inver3Dtrans(sim_patch, patchSize):
    size = len(sim_patch)
    layer = int(numpy.log2(size))
    data = numpy.zeros((size))
    for i in range(patchSize):
        for j in range(patchSize):
            for k in range(size):
                data[k] = sim_patch[k][i, j]
            data = waverec(data, 2, layer)
            for k in range(len(sim_patch)):
                sim_patch[k][i, j] = data[k]
    for k in range(size):
        sim_patch[k] = cv2.idct(sim_patch[k])
    return sim_patch


def gen_wienFilter(input, sigma):
    sigma = numpy.full(input[0].shape, sigma * sigma, dtype=numpy.float32)
    
    for k in range(len(input)):
        tmp = input[k] * input[k] + sigma
        input[k] = (input[k] * input[k]) / tmp
    
    return input

def calculate_weight_wien(coefficients, sigma, patchSize):
    sum_squared = 0.0
    for k in range(len(coefficients)):
         for i in range(patchSize):
            for j in range(patchSize):
                sum_squared += coefficients[k][i, j] * coefficients[k][i, j]
    if sum_squared == 0:
        return 1
    else:
        return 1.0 / (sigma * sigma * sum_squared)