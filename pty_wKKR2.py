from cProfile import label
import numpy as np
from scipy import fftpack as ft
import PIL.Image
import matplotlib.pyplot as plt
import time

start = time.perf_counter()

intObjectNum = 300
intProbePix = 100
intScanPix = 10
intXScanNum = 15
intYScanNum = 15

intIterationNum = 100
fltAlpha = 0.5
fltBeta = 0.5

strFileName = "/Users/furunobo/Pictures/toshimab.jpg"
img_int = PIL.Image.open(strFileName)
img_int = img_int.convert('L')
img_int = img_int.resize((intObjectNum, intObjectNum), PIL.Image.LANCZOS)
img_int = np.array(img_int)

# strFileName = "/Users/furunobo/Pictures/ikemen6.jpg"
# img_ph = PIL.Image.open(strFileName)
# img_ph = img_ph.convert('L')
# img_ph = img_ph.resize((intObjectNum, intObjectNum), PIL.Image.LANCZOS)

filename = '/Users/furunobo/Library/CloudStorage/OneDrive-TheUniversityofTokyo/lab_study/fe_xmcd.csv'
aryData = np.loadtxt(filename, delimiter=',', encoding='utf-8-sig')
energy = aryData[:120:10, 0]
spcOrig = aryData[:120:10, 1]
aryEnergy = np.repeat(energy, intObjectNum**2)
aryEnergy = aryEnergy.reshape([len(energy), intObjectNum, intObjectNum])

img_ph = np.empty((len(energy), intObjectNum, intObjectNum))
img_tmp = np.empty((len(energy), intObjectNum, intObjectNum))
for i in range(len(energy)):
    img_int = img_int / np.max(img_int)
    img_tmp[i] = np.exp(-spcOrig[i] * img_int)
for k in range(intObjectNum):
    for l in range(intObjectNum):
        img_ph[:, k, l] = (ft.hilbert(np.log(img_tmp[:, k, l]) * aryEnergy[:, k, l] / 1.240)) * 1.240 / aryEnergy[:, k, l]

plt.plot(energy, img_ph[:, 150, 150])
# plt.imshow(np.abs(img_ph[0]))
# plt.colorbar()
# plt.title("Phase")
plt.show()
aryExpObject_sub = np.empty((len(energy), intObjectNum, intObjectNum), dtype = complex)

listExpImg = []
for i in range(len(energy)):

    img_int = img_int / np.max(img_int)
    # img_ph[i] = img_ph[i] / np.max(img_ph[i])

    img = np.exp(-spcOrig[i] * img_int) * np.exp(1.0j * img_ph[i])

    aryExpObject = img
    aryExpObject_sub[i] = img

    y, x = np.indices((intProbePix, intProbePix))

    y -= intProbePix // 2
    x -= intProbePix // 2
    r = intProbePix // 20

    aryCircle = np.where(x ** 2 + y ** 2 < r ** 2, 1.0, 0.0)
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
plt.plot(energy, np.angle(aryExpObject_sub)[:, 150, 150], label = 'arg')
plt.plot(energy, np.abs(aryExpObject_sub)[:, 150, 150], label = 'int')
plt.plot(energy, np.exp(- spcOrig * 0.392), label = 'orig_int')
plt.plot(energy, ft.hilbert((np.log(np.exp(- spcOrig * 0.392)) * aryEnergy[:, 150, 150] / 1.240)) * 1.240 / aryEnergy[:, 150, 150], label = 'orig_arg')
plt.legend()
plt.show()

y, x = np.indices((intProbePix, intProbePix))
y = y - intProbePix // 2
x = x - intProbePix // 2
r = intProbePix // 10
aryProbe = np.empty((len(energy), intProbePix, intProbePix), dtype = complex)
aryProbe[:] = np.copy(aryExpProbe) # np.where(x ** 2 + y ** 2 < r ** 2, 1.0, 0.0)

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

for Iteration1 in range(intIterationNum):
    print(Iteration1)
    for energyi in range(len(energy)):
        print(energyi)
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

        if (Iteration1 + 1) % 10 == 0:
            aryProbe[energyi] = np.roll(aryProbe[energyi], -np.argmax(np.sum(np.abs(aryProbe[energyi]), axis = 1)) + intProbePix // 2, axis = 0)
            aryProbe[energyi] = np.roll(aryProbe[energyi], -np.argmax(np.sum(np.abs(aryProbe[energyi]), axis = 0)) + intProbePix // 2, axis = 1)

    if (Iteration1 + 1) % 100 == 0:
        spcPty = np.log(np.abs(aryObject)) * aryEnergy / 1.240
        for l in range(intObjectNum):
            for m in range(intObjectNum):
                aryHilSpc[:, m, l] = ft.hilbert(spcPty[:, m, l])
        constKKR = aryHilSpc - (np.angle(aryObject) * aryEnergy / 1.240)
        aveKKR = np.average(constKKR, axis=0)

        argObject = (aryHilSpc[:] - aveKKR) * 1.240 / aryEnergy[:]
        aryObject = np.sqrt(aryObject) * np.exp(1.0j * argObject)
        # plt.imshow(np.abs(aryObject[0]))
        # plt.show()

# plt.plot(energy, constKKR[:, 150, 150])
# plt.show()
end = time.perf_counter()
elapsed_time = end - start
print(elapsed_time)
# print((aryExpProbe[0, 0] * aryExpObject[aryExpPos[0, 0] + 1, aryExpPos[0, 1] + 1]).__abs__() / noise)

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