import cv2
import numpy as np
import time


a = np.random.rand(100,100)

start = time.time()
for i in range(1, 100):
    b1 = np.fft.fft2(a)

end = time.time()
print(end - start)


start = time.time()
for i in range(1, 100):
    b2 =np.fft.fft(np.fft.fft(a, axis=0), axis=1)

end = time.time()
print(end - start)


start = time.time()
for i in range(1, 100):
    c = cv2.dft(a)
end = time.time()
print(end - start)

assert(np.all(b==c))
