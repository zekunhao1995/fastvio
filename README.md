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

## QAs and Recommendations
1. Why no frame-based access?
  * Most video formats don't have a natural way to access by frame.
  * Different videos have different frame rate. Frame-based access cannot ensure the temporal uniformness of your samples.
  * Some videos even have variable frame rate.
  * Recommendation: scan videos once, save their frame-to-timecode table. Reuse them later.
2. Too slow when accessing non-keyframes.
  * There is no shortcut for decoding a random non-keyframe from a video. To get a non-keyframe, all the frames between previous keyframe and target frame need to be decoded.
  * Recommendation: Transcode the videos to have denser keyframes if necessary.
  * Trade-off is always there. If your task is IO or storage space bounded, trade CPU for them by reading from compressed videos. Another option is to go and get more disks and decompress all the videos before use.
3. Too slow when opening some video formats.
  * Some video containers such as MPEG and some flv, lack header or necessary metadata.
  * Recommendation: Remux them to mkv.
4. Why no hardware acceleration?
  * This library is designed for the scenario that many individual instances are running in parallel reading many individual videos (for example, dataloader for deep learning). In this scenario, overall thoughtput, instead of the speed of a single instance, is of utmost importance. Threading and opening hardware contexts introduce overheads and may not work well when there are a lot of running instances.
5. Are there any video formats that are:
  * Friendly for random access;
  * Low in decoding complexity;
  * Leveraging temporal smoothness for better compression?


