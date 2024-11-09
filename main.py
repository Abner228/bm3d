import cv2
import numpy
import tool
import copy
if __name__ == "__main__":
    # param
    patchSize = 7
    step = 3
    searchArea = 39     # 搜索范围
    maxMatch = 16       # 最多匹配数
    threshold = 2500    # 块匹配阈值 (频域)
    thresholdDenoise = 400   #去噪后的块匹配阈值 （空域）
    threshold3d = 75    # 三维空间滤波后的硬阈值
    sigma = 3
    # run
    img_bgr = cv2.imread('python/BM3D/pic.jpg')

    print("[I] Origin Size: ", img_bgr.shape)
    img_bgr=cv2.resize(img_bgr,(img_bgr.shape[1] // 8, img_bgr.shape[0] // 8), 
                       interpolation=cv2.INTER_LINEAR)
    print("[I] Downscale to : ", img_bgr.shape)

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # uint8
    tool.SavePic(img_gray, 'python/BM3D/pic_gray.png')

    img_gray = img_gray.astype(numpy.float32)
    height, width = img_gray.shape

    row_idx_ls, col_idx_ls, block_ls = tool.GetAllBlock(img_gray, height, width, patchSize, step)
    blockRowNum = len(row_idx_ls)
    blockColNum = len(col_idx_ls)

    hamming = tool.hamming_window(patchSize)  # 2-D hamming window
    denominator_ht = numpy.zeros_like(img_gray)
    numerator_ht = numpy.zeros_like(img_gray)

    for i in range(len(block_ls)):
        block_ls[i] = cv2.dct(block_ls[i])

    for i in range(blockRowNum):
        for j in range(blockColNum):
            # print("[I] Processing: {}/{}".format(i*blockColNum+j+1, blockRowNum*blockColNum))
            sim_patch, sim_idx = tool.getSimilarPatch(block_ls, i, j, blockRowNum, blockColNum,
                                                      searchArea, patchSize, step, maxMatch, threshold)
            
            sim_idx_row = []
            sim_idx_col = []
            for k in range(len(sim_idx)):
                sim_idx_row.append(row_idx_ls[int(sim_idx[k] / blockColNum)])
                sim_idx_col.append(col_idx_ls[int(sim_idx[k] % blockColNum)])
            
            sim_patch = tool.Tran1D(sim_patch, patchSize)
            sim_patch = tool.DetectZero(sim_patch, threshold3d)
            weight_ht = tool.calculate_weight_ht(sim_patch, sigma)
            sim_patch = tool.Inver3Dtrans(sim_patch, patchSize)
            numerator_ht, denominator_ht = tool.aggregation(numerator_ht, denominator_ht, sim_idx_row, sim_idx_col,
                                                            sim_patch, weight_ht, patchSize, hamming)
    denominator_ht = denominator_ht + 1e-6
    basic = numerator_ht / denominator_ht

    basic_copy = copy.deepcopy(basic)
    basic_copy = numpy.clip(basic_copy, 0, 255)
    basic_copy = basic_copy.astype(numpy.uint8)
    tool.SavePic(basic_copy, 'python/BM3D/pic_gray_basic.png')


    # second
    denominator_wie = numpy.zeros_like(basic)
    numerator_wie = numpy.zeros_like(basic)

    basic_row_idx_ls, basic_col_idx_ls, basic_block_ls = tool.GetAllBlock(basic, height, width, patchSize, step)
    for i in range(blockRowNum):
        for j in range(blockColNum):
            # print("[I] Processing: {}/{}".format(i*blockColNum+j+1, blockRowNum*blockColNum))
            basic_sim_patch, basic_sim_idx = tool.getSimilarPatch(basic_block_ls, i, j, blockRowNum, blockColNum,
                                                                  searchArea, patchSize, step, maxMatch, thresholdDenoise)
            sim_idx_row = []
            sim_idx_col = []
            noisy_sim_patch = []
            for k in range(len(basic_sim_idx)):
                sim_idx_row.append(row_idx_ls[int(basic_sim_idx[k] / blockColNum)])
                sim_idx_col.append(col_idx_ls[int(basic_sim_idx[k] % blockColNum)])
                tmp = copy.deepcopy(img_gray[sim_idx_row[k]:sim_idx_row[k]+patchSize,
                                             sim_idx_col[k]:sim_idx_col[k]+patchSize])
                noisy_sim_patch.append(tmp)

            for k in range(len(basic_sim_patch)):
                basic_sim_patch[k] = cv2.dct(basic_sim_patch[k])
            for k in range(len(noisy_sim_patch)):
                noisy_sim_patch[k] = cv2.dct(noisy_sim_patch[k])
            
            basic_sim_patch = tool.Tran1D(basic_sim_patch, patchSize)
            noisy_sim_patch = tool.Tran1D(noisy_sim_patch, patchSize)

            wienFilter = tool.gen_wienFilter(basic_sim_patch, sigma)
            weight_wie = tool.calculate_weight_wien(wienFilter, sigma, patchSize)
            
            for k in range(len(noisy_sim_patch)):
                noisy_sim_patch[k] = noisy_sim_patch[k] * wienFilter[k]
            
            noisy_sim_patch = tool.Inver3Dtrans(noisy_sim_patch, patchSize)

            numerator_wie, denominator_wie = tool.aggregation(numerator_wie, denominator_wie, sim_idx_row, sim_idx_col,
                                                              noisy_sim_patch, weight_wie, patchSize, hamming)
    denominator_wie = denominator_wie + 1e-6
    image_denoised = numerator_wie / denominator_wie
    
    image_denoised = numpy.clip(image_denoised, 0, 255)
    image_denoised = image_denoised.astype(numpy.uint8)
    tool.SavePic(image_denoised, 'python/BM3D/pic_gray_denoise.png')
