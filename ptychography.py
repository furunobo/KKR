from re import A
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import time

start = time.perf_counter()

intObjectNum = 300
intProbePix = 64
intScanPix = 10
intXScanNum = 15
intYScanNum = 15

intIterationNum = 300
fltAlpha = 0.5
fltBeta = 0.5

strFileName = "/Users/furunobo/Pictures/toshimab.jpg" #Object image
img_int = PIL.Image.open(strFileName)
img_int = img_int.convert('L')
img_int = img_int.resize((intObjectNum, intObjectNum), PIL.Image.LANCZOS)

strFileName = "/Users/furunobo/Pictures/IMG_1089.jpeg" #Phase image
img_ph = PIL.Image.open(strFileName)
img_ph = img_ph.convert('L')
img_ph = img_ph.resize((intObjectNum, intObjectNum), PIL.Image.LANCZOS)

filename = "/Users/furunobo/Library/CloudStorage/OneDrive-TheUniversityofTokyo/lab_study/fe_xmcd.csv" #Spectrum data
aryData = np.loadtxt(filename, delimiter=',', encoding='utf-8-sig')
energy = aryData[:, 0]
spcExp = aryData[:, 1]

img_int = img_int / np.max(img_int)
img_ph = np.array(img_ph)
img_ph = img_ph / np.max(img_ph) - 0.5

img = np.exp(-aryData[30, 1] * img_int) * np.exp(1.0j * img_ph * 2*np.pi / 10)
aryExpObject = img #/ np.max(np.abs(img))
plt.imshow(np.abs(aryExpObject))
plt.colorbar()
plt.title("Object.")
plt.show()
plt.imshow(np.angle(aryExpObject))
plt.colorbar()
plt.title("Phase.")
plt.show()

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

listExpDiffraction = []
SN = 10**3
noise = np.random.normal(intProbePix, intProbePix) / SN

for i in range(intXScanNum * intYScanNum):
    sub = np.fft.fft2(
        aryExpProbe * aryExpObject[
            aryExpPos[i, 0]:aryExpPos[i, 0] + intProbePix, 
            aryExpPos[i, 1]:aryExpPos[i, 1] + intProbePix]
    ) + noise
    listExpDiffraction.append(np.abs(sub) ** 2)

y, x = np.indices((intProbePix, intProbePix))
y = y - intProbePix // 2
x = x - intProbePix // 2
r = intProbePix // 10
aryProbe = np.copy(aryExpProbe) #np.where(x ** 2 + y ** 2 < r ** 2, 1.0, 0.0)
plt.imshow(aryProbe.__abs__())
plt.title("Probe.")
plt.show()

# aryObject = np.random.rand(intObjectNum, intObjectNum) + 0j
aryObject = np.ones([intObjectNum, intObjectNum]) + 0j
aryObjectBefore = np.empty(shape = [intProbePix, intProbePix], dtype = complex)
aryExitWave = np.empty(shape = [intProbePix, intProbePix], dtype = complex)
aryExitWaveBefore = np.empty(shape = [intProbePix, intProbePix], dtype = complex)
arySubComplex = np.empty(shape = [intProbePix, intProbePix], dtype = complex)

intIterationSamplePos = aryExpPos.shape[0]

for Iteration1 in range(intIterationNum):
    print(Iteration1)
    for iSamplePos in range(intIterationSamplePos):

        aryObjectBefore = np.copy(
            aryObject[
                aryExpPos[iSamplePos, 0] : aryExpPos[iSamplePos, 0] + intProbePix,
                aryExpPos[iSamplePos, 1] : aryExpPos[iSamplePos, 1] + intProbePix
            ]
        )
        aryProbeBefore = np.copy(aryProbe)
        aryExitWaveBefore = aryProbeBefore * aryObjectBefore

        arySubComplex = np.fft.fft2(aryProbe * aryObjectBefore)

        arySubComplex = np.sqrt(listExpDiffraction[iSamplePos]) * np.exp(1.0j * np.angle(arySubComplex))

        aryExitWave = np.fft.ifft2(arySubComplex)

        if Iteration1 > 9:
            aryProbe = aryProbeBefore + fltBeta * np.conjugate(aryObjectBefore) / np.max(np.abs(aryObjectBefore)) ** 2 * (aryExitWave - aryExitWaveBefore)
        
        aryObject[
            aryExpPos[iSamplePos, 0] : aryExpPos[iSamplePos, 0] + intProbePix,
            aryExpPos[iSamplePos, 1] : aryExpPos[iSamplePos, 1] + intProbePix
        ] = aryObjectBefore + fltAlpha * np.conjugate(aryProbeBefore) / np.max(np.abs(aryProbeBefore)) ** 2 * (aryExitWave - aryExitWaveBefore)

    if (Iteration1+1) % 10 == 0:
        aryProbe = np.roll(aryProbe, -np.argmax(np.sum(np.abs(aryProbe), axis = 1)) + intProbePix // 2, axis = 0)
        aryProbe = np.roll(aryProbe, -np.argmax(np.sum(np.abs(aryProbe), axis = 0)) + intProbePix // 2, axis = 1)

end = time.perf_counter()
elapsed_time = end - start
print(elapsed_time)
# print((aryExpProbe[0, 0] * aryExpObject[aryExpPos[0, 0] + 1, aryExpPos[0, 1] + 1]).__abs__() / noise)

plt.imshow(np.abs(aryObject))
plt.colorbar()
plt.title("Object.")
plt.show()
plt.imshow(np.angle(aryObject))
plt.colorbar()
plt.title("Phase.")
plt.show()
plt.imshow(aryProbe.__abs__())
plt.title("Probe.")
plt.show()