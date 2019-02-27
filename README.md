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
> ============ Benchmark: Context creating and destroying ===========  
> 0.322ms / ops  
> ============ Benchmark: Read first 10 frames ===========  
> 16.183ms / ops  
> ============ Benchmark: Seek ===========  
> 0.333ms / ops  
> ============ Benchmark: Seek and grab KeyFrame ===========  
> 3.028ms / ops  
> ============ Benchmark: Seek and grab non-keyframe (worst case) ===========  
> 25.574ms  / ops  
> ============ Benchmark: Reading the whole video frame by frame ===========  
> 485.156ms / ops  
