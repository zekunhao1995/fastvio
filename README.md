# FastVIO

## To make FastVIO:
1. check python version in Makefile
2. make

## To test:
`python3.6 test.py`

## To remux any video to mkv container (and discard audio):
`ffmpeg -hide_banner -loglevel panic -y -i src_vid_path -map 0:v:0 -codec copy dst_vid_path`

## Dependencies
Python >= 3.3 
Numpy 
FFmpeg >= 3.3 

## Benchmark on my PC
`
============ Benchmark: Context creating and destroying ===========
0.902ms / ops
============ Benchmark: Read first 10 frames ===========
44.866ms / ops
============ Benchmark: Seek ===========
1.085ms / ops
============ Benchmark: Seek and grab KeyFrame ===========
7.780ms / ops
============ Benchmark: Seek and grab non-keyframe (worst case) ===========
58.713ms / ops
============ Benchmark: Reading the whole video frame by frame ===========
626.692ms / ops
`
