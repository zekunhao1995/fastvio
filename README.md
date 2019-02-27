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


