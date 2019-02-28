import numpy as np
import fastvio
import sys

import time

"""
#print(fastvio.open('/output.mkv'))
foo = '/output.mkv'
print(sys.getrefcount(foo))
ret = fastvio.open('/output.mkv')
print(hex(ret))
print(ret)
print(sys.getrefcount(foo))
print(foo)

#help(fastvio);

print('Now trying to open an non-existing file')
ret = fastvio.open('/output1232.mkv')
"""
print(fastvio)

vid_path = './testvga.mkv'

vio = fastvio.open(vid_path)


print("Duration: {}us".format(fastvio.get_duration(vio)))
frame, pts = fastvio.grab_frame(vio)
print("Pts of first frame: {}".format(pts))
print("Try seeking. seek() will seek to the closest frame that has pts >= given pts.")
fastvio.seek(vio, 6300000)
frame, pts = fastvio.grab_frame(vio)
print(pts)
print("Do some consecutive reading. This is much more efficient than randomly seeking.")
frame, pts = fastvio.grab_frame(vio)
print(pts)
frame, pts = fastvio.grab_frame(vio)
print(pts)
frame, pts = fastvio.grab_frame(vio)
print(pts)
print("This is the last frame. Note that its pts is lower than duration since it has its own duration.")
fastvio.seek(vio, 10660000) # This is the pts of the last frame of testvga.mkv
frame, pts = fastvio.grab_frame(vio)
print(pts)
print("============ More debugging info follows ===========")
fastvio.print_dbg(vio)
print("Release resources by calling close(ctx).")
print("FastVIO allows co-exist of multiple contexts.")
fastvio.close(vio)

num_samples = 200
print("============ Benchmark: Context creating and destroying ===========")
start = time.time()
for i in range(num_samples): # 3.32s!!!
    vio = fastvio.open(vid_path)
    fastvio.close(vio)
end = time.time()
print("{:.3f}ms / ops".format((end-start)/num_samples*1000))

print("============ Benchmark: Read first 10 frames ===========")
start = time.time()
for i in range(num_samples):
    vio = fastvio.open(vid_path)
    for j in range(10):
        fastvio.grab_frame(vio)
    fastvio.close(vio)
end = time.time()
print("{:.3f}ms / ops".format((end-start)/num_samples*1000))

print("============ Benchmark: Seek ===========")
start = time.time()
for i in range(num_samples):
    vio = fastvio.open(vid_path)
    fastvio.seek(vio, 6300000)
    fastvio.close(vio)
end = time.time()
print("{:.3f}ms / ops".format((end-start)/num_samples*1000))

print("============ Benchmark: Seek and grab KeyFrame ===========")
start = time.time()
for i in range(num_samples):
    vio = fastvio.open(vid_path)
    fastvio.seek(vio, 7547000)
    fastvio.grab_frame(vio)
    fastvio.close(vio)
end = time.time()
print("{:.3f}ms / ops".format((end-start)/num_samples*1000))

print("============ Benchmark: Seek and grab non-keyframe (worst case) ===========")
start = time.time()
for i in range(num_samples):
    vio = fastvio.open(vid_path)
    fastvio.seek(vio, 9200000) # Last frame before another keyframe, thus worst case ^_^
    fastvio.grab_frame(vio)
    fastvio.close(vio)
end = time.time()
print("{:.3f}ms / ops".format((end-start)/num_samples*1000))

print("============ Benchmark: Reading the whole video frame by frame ===========")
start = time.time()
for i in range(5):
    vio = fastvio.open(vid_path)
    while True:
        ret = fastvio.grab_frame(vio)
        if ret is None:
            break # end of file
    fastvio.close(vio)
end = time.time()
print("{:.3f}ms / ops".format((end-start)/5*1000))

print("============ Benchmark: Seek and grab non-keyframe (worst case) (slice level Parallel) ===========")
start = time.time()
for i in range(num_samples):
    vio = fastvio.open(vid_path, thread_mode='thread_slice', num_threads=8)
    fastvio.seek(vio, 9200000) # Last frame before another keyframe, thus worst case ^_^
    fastvio.grab_frame(vio)
    fastvio.close(vio)
end = time.time()
print("{:.3f}ms / ops".format((end-start)/num_samples*1000))

print("============ Benchmark: Reading the whole video frame by frame (slice level Parallel) ===========")
start = time.time()
for i in range(5):
    vio = fastvio.open(vid_path, thread_mode='thread_slice', num_threads=8)
    while True:
        ret = fastvio.grab_frame(vio)
        if ret is None:
            break # end of file
    fastvio.close(vio)
end = time.time()
print("{:.3f}ms / ops".format((end-start)/5*1000))

print("============ Benchmark: Seek and grab non-keyframe (worst case) (frame level Parallel) ===========")
start = time.time()
for i in range(num_samples):
    vio = fastvio.open(vid_path, thread_mode='thread_frame', num_threads=8)
    fastvio.seek(vio, 9200000) # Last frame before another keyframe, thus worst case ^_^
    fastvio.grab_frame(vio)
    fastvio.close(vio)
end = time.time()
print("{:.3f}ms / ops".format((end-start)/num_samples*1000))

print("============ Benchmark: Reading the whole video frame by frame (frame level Parallel) ===========")
start = time.time()
for i in range(5):
    vio = fastvio.open(vid_path, thread_mode='thread_frame', num_threads=8)
    while True:
        ret = fastvio.grab_frame(vio)
        if ret is None:
            break # end of file
    fastvio.close(vio)
end = time.time()
print("{:.3f}ms / ops".format((end-start)/5*1000))

print("============ Visual inspection ===========")
import matplotlib.pyplot as plt
fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False)

drawn = 0
vio = fastvio.open(vid_path)
vio2 = fastvio.open('test4k.mkv')
while True:
    for j in range(20):
        ret = fastvio.grab_frame(vio)
        ret2 = fastvio.grab_frame(vio2)
        if ret is None or ret2 is None:
            break;
        frame = ret[0]
        frame2 = ret2[0]
        if drawn==0:
            im1 = axs[0].imshow(frame)
            im2 = axs[1].imshow(frame2)
            drawn=1
        else:
            im1.set_data(frame)
            im2.set_data(frame2)
        plt.show(block=False)
        plt.pause(0.05)
    fastvio.seek(vio, 10000)
fastvio.close(vio)
fastvio.close(vio2)
