from cProfile import label
import numpy as np
from scipy import fftpack as ft
import PIL.Image
import matplotlib.pyplot as plt
import time

start = time.perf_counter()

intObjectNum = 300
intProbePix = 64
intScanPix = 10
intXScanNum = 15
intYScanNum = 15

intIterationNum = 1000
fltAlpha = 0.5
fltBeta = 0.5

strFileName = "toshimab.jpg"
img_int = PIL.Image.open(strFileName)
img_int = img_int.convert('L')
img_int = img_int.resize((intObjectNum, intObjectNum), PIL.Image.LANCZOS)
img_int = np.array(img_int)

filename = "fe_xmcd.csv"
aryData = np.loadtxt(filename, delimiter=',', encoding='utf-8-sig')
stop = -1
step = 10
energy = aryData[:stop:step, 0] # np.linspace(690, 710, 20)
spcOrig = aryData[:stop:step, 1] # 1/((energy - 700)**2 + 2) 

aryEnergy = np.repeat(energy, intObjectNum**2)
aryEnergy = aryEnergy.reshape([len(energy), intObjectNum, intObjectNum])

def physical_model(object_image, energy_list, _interpolation): # 3D (energy, x, y) image to 3D image
    n = np.copy(len(energy_list))
    if _interpolation == 1:
        appendnum = 500
        tmp = np.linspace(energy_list[-1]*2 - energy_list[-2], 1000, appendnum)
        energy_appended = np.append(energy_list, tmp)
        tmp = np.linspace(100, energy_list[0]*2 - energy_list[1], appendnum)
        energy_appended = np.append(tmp, energy_appended)
        energy_list = energy_appended
    hilbert_image = np.empty((len(energy_list), object_image.shape[1], object_image.shape[2]))
    for k in range(object_image.shape[1]):
        for l in range(object_image.shape[2]):
            transparence = np.abs(object_image[:, k, l])
            if _interpolation == 1:
                spc_appended = np.append([transparence[0]] * appendnum, transparence)
                spc_appended = np.append(spc_appended, [transparence[-1]] * appendnum)
                transparence = spc_appended
            hilbert_image[:, k, l] = ft.hilbert(np.log(transparence) * energy_list[:] / 1.240)
    if _interpolation == 1:
        tmp = np.empty((n, object_image.shape[1], object_image.shape[2]))
        tmp = hilbert_image[appendnum-1:-appendnum-1]
        hilbert_image = tmp - hilbert_image[-appendnum-1]
    return hilbert_image

def generate_complex_image(image_int, energy_list, spectrum_exp, _interpolation): # 2D to 3D
    image_tmp = np.empty((len(energy_list), image_int.shape[0], image_int.shape[1]))
    for i in range(len(energy_list)):
        image_tmp[i] = np.exp(-spectrum_exp[i] * image_int)
    hil_img = physical_model(image_tmp, energy_list, _interpolation)
    image_cx = np.empty_like(image_tmp) + 0j
    for i in range(len(energy_list)):
        image_cx[i] = image_tmp[i] * np.exp(1.0j * (-hil_img[i] - 0.01) * 1.240 / energy_list[i])
    return image_cx

# img_ph = np.empty((len(energy), intObjectNum, intObjectNum))
# img_tmp = np.empty((len(energy), intObjectNum, intObjectNum))
# for i in range(len(energy)):
#     img_int = img_int / np.max(img_int)
#     img_tmp[i] = np.exp(-spcOrig[i] * img_int)
# for k in range(intObjectNum):
#     for l in range(intObjectNum):
#         img_ph[:, k, l] = (ft.hilbert(np.log(img_tmp[:, k, l]) * aryEnergy[:, k, l] / 1.240)) * 1.240 / aryEnergy[:, k, l]
interpolation = 0
img_int = img_int / np.max(img_int)
# img_ph = generate_phase_image(img_int, energy, spcOrig)

listExpImg = []
img = generate_complex_image(img_int, energy, spcOrig, interpolation)

plt.plot(energy, np.angle(img[:, 130, 130]))
plt.title("Phase")
plt.show()
# vac_pix = 5
# vac_area = (intObjectNum//2 - vac_pix, intObjectNum//2 + vac_pix)
# img[:, vac_area[0]:vac_area[1], vac_area[0]:vac_area[1]] = 1.0 + 0j
for i in range(len(energy)):

    # img_int = img_int / np.max(img_int)
    # img_ph[i] = img_ph[i] / np.max(img_ph[i])

    aryExpObject = img[i]

    y, x = np.indices((intProbePix, intProbePix))

    y -= intProbePix // 2
    x -= intProbePix // 2
    r = intProbePix // 20

    aryCircle = np.where(x ** 2 + y ** 2 <= r ** 2, 1.0, 0.0)
    aryExpProbe = np.fft.ifftshift(np.fft.fft2(aryCircle))

    aryExpPos = np.empty(shape = [intXScanNum * intYScanNum, 2], dtype = int)

    x, y = np.indices((intXScanNum, intYScanNum))
    x = x.flatten()
    y = y.flatten()

    aryExpPos[:, 0] = x * intScanPix + intObjectNum // 2 - intProbePix // 2 - intScanPix * (intXScanNum // 2)
    aryExpPos[:, 1] = y * intScanPix + intObjectNum // 2 - intProbePix // 2 - intScanPix * (intYScanNum // 2)

    listExpDiffraction = np.empty((intXScanNum * intYScanNum, intProbePix, intProbePix))
    SN = 10**3
    noise = np.random.normal(intProbePix, intProbePix) / SN

    for k in range(intXScanNum * intYScanNum):
        sub = np.fft.fft2(
            aryExpProbe * aryExpObject[
                aryExpPos[k, 0]:aryExpPos[k, 0] + intProbePix, 
                aryExpPos[k, 1]:aryExpPos[k, 1] + intProbePix]
        ) #+ noise
        listExpDiffraction[k] = np.abs(sub) ** 2
    
    listExpImg.append(listExpDiffraction)

plt.imshow(np.abs(aryExpObject))
plt.colorbar()
plt.title("Object.")
plt.show()
plt.imshow(np.angle(aryExpObject))
plt.colorbar()
plt.title("Phase.")
plt.show()

check_pixel = (130, 130)
spcExp = img[:, check_pixel[0], check_pixel[1]]
const_KKR = -physical_model(spcExp.reshape(len(spcExp), 1, 1), energy, interpolation)[:, 0, 0] - (np.angle(spcExp) * energy / 1.240)
# if interpolation == 1:
#     appendnum = 500
#     tmp = np.linspace(energy[-1]*2 - energy[-2], 1000, appendnum)
#     energy_append = np.append(energy, tmp)
#     tmp = np.linspace(100, energy[0]*2 - energy[1], appendnum)
#     energy_append = np.append(tmp, energy_append)
#     spcExp_append = np.append([spcExp[0]] * appendnum, spcExp)
#     spcExp_append = np.append(spcExp_append, [spcExp[-1]] * appendnum)
#     const_KKR = -ft.hilbert(np.log(np.abs(spcExp_append)) * energy_append / 1.240) - (np.angle(spcExp_append) * energy_append / 1.240)
#     const_KKR = const_KKR[appendnum-1:-appendnum-1]

plt.plot(energy, np.angle(spcExp), label = 'arg')
plt.plot(energy, np.abs(spcExp), label = 'int')
plt.plot(energy, np.exp(- spcOrig * img_int[check_pixel[0], check_pixel[1]]), label = 'orig_int')
plt.plot(energy, -physical_model(np.exp(- spcOrig * img_int[check_pixel[0], check_pixel[1]]).reshape(len(spcExp), 1, 1), energy, interpolation)[:, 0, 0] * 1.240 / energy, label = 'orig_arg')
plt.plot(energy, const_KKR, label = 'const')
plt.legend()
plt.show()

y, x = np.indices((intProbePix, intProbePix))
y = y - intProbePix // 2
x = x - intProbePix // 2
r = intProbePix // 10
aryProbe = np.empty((len(energy), intProbePix, intProbePix), dtype = complex)
aryProbe[:] = np.copy(aryExpProbe) # np.where(x ** 2 + y ** 2 < r ** 2, 1.0, 0.0)
# np.copy(aryExpProbe) #
plt.imshow(np.abs(aryProbe[2]))
plt.title("Probe.")
plt.show()

# aryObject = np.random.rand(len(energy), intObjectNum, intObjectNum) + 0.0j
aryObject = np.ones(shape = [len(energy), intObjectNum, intObjectNum], dtype = complex)
aryObjectBefore = np.empty(shape = [intProbePix, intProbePix], dtype = complex)
aryExitWave = np.empty(shape = [intProbePix, intProbePix], dtype = complex)
aryExitWaveBefore = np.empty(shape = [intProbePix, intProbePix], dtype = complex)
arySubComplex = np.empty(shape = [intProbePix, intProbePix], dtype = complex)
aryHilSpc = np.empty((len(energy), intObjectNum, intObjectNum))

intIterationSamplePos = aryExpPos.shape[0]

listconst = []
listHil = []
listArg = []
for Iteration1 in range(intIterationNum):
    print(Iteration1)
    for energyi in range(len(energy)):
        # print(energyi)
        for iSamplePos in range(intIterationSamplePos):
            
            aryObjectBefore = np.copy(
                aryObject[energyi, 
                    aryExpPos[iSamplePos, 0] : aryExpPos[iSamplePos, 0] + intProbePix,
                    aryExpPos[iSamplePos, 1] : aryExpPos[iSamplePos, 1] + intProbePix
                ]
            )
            aryProbeBefore = np.copy(aryProbe[energyi])

            aryExitWaveBefore = aryProbeBefore * aryObjectBefore

            arySubComplex = np.fft.fft2(aryProbe[energyi] * aryObjectBefore)

            arySubComplex = np.sqrt(listExpImg[energyi][iSamplePos]) * np.exp(1.0j * np.angle(arySubComplex))

            aryExitWave = np.fft.ifft2(arySubComplex)

            aryProbe[energyi] = aryProbeBefore + fltBeta * np.conjugate(aryObjectBefore) / np.max(np.abs(aryObjectBefore)) ** 2 * (aryExitWave - aryExitWaveBefore)

            aryObject[energyi, 
                aryExpPos[iSamplePos, 0] : aryExpPos[iSamplePos, 0] + intProbePix,
                aryExpPos[iSamplePos, 1] : aryExpPos[iSamplePos, 1] + intProbePix
            ] = aryObjectBefore + fltAlpha * np.conjugate(aryProbeBefore) / np.max(np.abs(aryProbeBefore)) ** 2 * (aryExitWave - aryExitWaveBefore)

            # aryObject[energyi, vac_area[0]:vac_area[1], vac_area[0]:vac_area[1]] = 1.0 + 0j

        # aryObject = aryObject / np.max(np.real(aryObject))

        if (Iteration1 + 1) % 10 == 0:
            aryProbe[energyi] = np.roll(aryProbe[energyi], -np.argmax(np.sum(np.abs(aryProbe[energyi]), axis = 1)) + intProbePix // 2, axis = 0)
            aryProbe[energyi] = np.roll(aryProbe[energyi], -np.argmax(np.sum(np.abs(aryProbe[energyi]), axis = 0)) + intProbePix // 2, axis = 1)
    # if (Iteration1 + 1) % 10 == 0 and (Iteration1 + 1) > 300 and (Iteration1 + 1) <= 400:
    #     aryHilSpc = physical_model(np.abs(aryObject), energy)
    #     argBefore = np.angle(aryObject) * aryEnergy / 1.240
    #     constKKR = -aryHilSpc - argBefore
    #     listconst.append(constKKR[:, 130, 130])

    if (Iteration1 + 1) % 100 == 0: #and Iteration1 > 90:
        # spcPty = np.log(np.abs(aryObject)) * aryEnergy / 1.240
        # for l in range(intObjectNum):
        #     for m in range(intObjectNum):
        #         aryHilSpc[:, m, l] = ft.hilbert(spcPty[:, m, l])
        aryHilSpc = physical_model(np.abs(aryObject), energy, 1)
        argBefore = np.angle(aryObject) * aryEnergy / 1.240
        listHil.append(aryHilSpc[:, 130, 130])
        listArg.append(argBefore[:, 130, 130])
        constKKR = -aryHilSpc - argBefore
        listconst.append(constKKR[:, 130, 130])
        aveKKR = np.average(constKKR, axis=0) # [:, aryExpPos[0, 0]:aryExpPos[-1, 0]+intProbePix, aryExpPos[0, 1]:aryExpPos[-1, 1]+intProbePix]

        # aveKKR = np.pad(aveKKR, [aryExpPos[0, 0], aryExpPos[0, 0]], "constant")
        argObject = (-aryHilSpc[:] - aveKKR) * 1.240 / aryEnergy[:]
        # plt.imshow(aveKKR)
        # plt.colorbar()
        # plt.show()
        aryObject = np.abs(aryObject) * np.exp(1.0j * argObject)
        # plt.imshow(np.abs(aryObject[0]))
        # plt.show()

end = time.perf_counter()
elapsed_time = end - start
print(elapsed_time)
print(len(listconst))
for i in range(len(listHil)):
    plt.plot(energy, listHil[i], label=i.__str__())
plt.legend()
plt.show()
for i in range(len(listArg)):
    plt.plot(energy, listArg[i], label=i.__str__())
plt.legend()
plt.show()
for i in range(len(listconst)):
    plt.plot(energy, listconst[i], label=i.__str__())
plt.legend()
plt.show()
# plt.plot(energy, constKKR[:, 130, 130], label='const')
# plt.plot(energy, aryHilSpc[:, 130, 130], label='hilbert')
# plt.plot(energy, argBefore[:, 130, 130], label='arg')
# plt.legend()
# plt.show()
# print((aryExpProbe[0, 0] * aryExpObject[aryExpPos[0, 0] + 1, aryExpPos[0, 1] + 1]).__abs__() / noise)

# for i in range(len(energy)):
#     fig = plt.figure(dpi=250, figsize=(14,4))

#     ax1 = fig.add_subplot(1, 3, 1)
#     ax1.set_title('object')
#     ax2 = fig.add_subplot(1, 3, 2)
#     ax2.set_title('phase')
#     ax3 = fig.add_subplot(1, 3, 3)
#     ax3.set_title('probe')
#     im1 = ax1.imshow(np.abs(aryObject[i]))
#     im2 = ax2.imshow(np.angle(aryObject[i]))
#     im3 = ax3.imshow(np.abs(aryProbe[i]))
#     fig.colorbar(im1, ax=ax1)
#     fig.colorbar(im2, ax=ax2)
#     fig.colorbar(im3, ax=ax3)
#     fig.tight_layout()
#     fig.savefig(i.__str__() + ".jpg", dpi=250)

plt.imshow(np.abs(aryObject[0]))
plt.colorbar()
plt.title("Object.")
plt.show()
plt.imshow(np.angle(aryObject[0]))
plt.colorbar()
plt.title("Phase.")
plt.show()
plt.imshow(np.abs(aryProbe[0]))
plt.title("Probe.")
plt.show()

# tmp = np.empty(len(energy))
# for p in range(len(energy)):
#     tmp[p] = np.average(aryObject[p, 130:150, 130:150])
# plt.plot(energy, np.abs(tmp))
# plt.show()