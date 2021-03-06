#include "fastvio.h"
#include "fastvio_common.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MY_BIND_C
#include <numpy/arrayobject.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <time.h>
#include <stdlib.h>

#define FASTVIO_FLAG_OUT_OF_PACKET (0)
#define FASTVIO_FLAG_GOT_FIRST_FRAME (1)
#define FASTVIO_FLAG_GOT_LAST_FRAME (2)
#define FASTVIO_FLAG_FLUSHED (3)
#define TEST_FLAG(number,bit) ((number>>bit)&(1UL))
#define SET_FLAG(number,bit) (number|=(1UL<<bit))
#define CLEAR_FLAG(number,bit) (number&=(~(1UL<<bit)))
//#define Toggle(number,bit) (number ^ (1<<bit))

struct FastvioCtx
{
    AVFormatContext *fmt_ctx;
    AVCodecContext *video_dec_ctx;
    struct SwsContext *sws_context;
    int width, height;
    enum AVPixelFormat pix_fmt;
    const char *src_filename;
    uint8_t *video_dst_data[4];
    int      video_dst_linesize[4];
    int video_dst_bufsize;
    int video_stream_idx;
    AVFrame *frame;
    AVPacket pkt;
    int64_t current_pts;
    int64_t seek_target_pts;
    unsigned int flags;
};

/* initialize fields of FastvioCtx */
void init_fastvio_ctx(struct FastvioCtx* ctx) {
    ctx->fmt_ctx = NULL;
    ctx->video_dec_ctx = NULL;
    ctx->sws_context = NULL;
    ctx->video_dst_data[0] = NULL;
    ctx->video_dst_data[1] = NULL;
    ctx->video_dst_data[2] = NULL;
    ctx->video_dst_data[3] = NULL;
    ctx->frame = NULL;
    ctx->current_pts = -1;
    ctx->seek_target_pts = 0;
    ctx->flags = 0;
}

struct FastvioCtx* create_fastvio_ctx() {
    struct FastvioCtx* ctx = (struct FastvioCtx*)malloc(sizeof(struct FastvioCtx));
    init_fastvio_ctx(ctx);
    return ctx;
}

/* 224 bytes, 128 elements, 28KB memory if left on heap. */
//static struct FastvioCtx ctx_stor[128];
//static int ctx_stor_cnt = 0;

static void init_swscale(struct FastvioCtx* ctx, enum AVPixelFormat src_pix_fmt)
{
    ctx->sws_context = sws_getContext(ctx->video_dec_ctx->width, ctx->video_dec_ctx->height, src_pix_fmt, // src
                                    ctx->video_dec_ctx->width, ctx->video_dec_ctx->height, AV_PIX_FMT_RGB24,   // dst
                                    0, NULL, NULL, NULL);
    
    //printf("swscale srcRange: %d\n",ctx->sws_context->srcRange);
}
/*
 * It will be more natural if we split this into send_packet and receive_packet.
 * Better readability & much cleaner decoding state machine.
 */
static int decode_packet(struct FastvioCtx* ctx, int *got_frame, int cached)
{
    AVPacket* pkt = &ctx->pkt;
    AVCodecContext *video_dec_ctx = ctx->video_dec_ctx;
    AVFrame *frame = ctx->frame;
    AVStream *st = ctx->fmt_ctx->streams[ctx->video_stream_idx];

    int ret = 0;
    int decoded = pkt->size;
    *got_frame = 0;
    if (pkt->stream_index == ctx->video_stream_idx) {
        /* decode video frame */
        // ret = avcodec_decode_video2(video_dec_ctx, frame, got_frame, &pkt);
        if (!cached) {
            ret = avcodec_send_packet(video_dec_ctx, pkt);
            if (ret < 0) {
                fprintf(stderr, "Error sending package (%s)\n", av_err2str(ret));
                return ret;
            }
        }
        ret = avcodec_receive_frame(video_dec_ctx, frame);
        if (ret == 0)
            *got_frame = 1;
        else if (ret == AVERROR_EOF)
            return 0;
        else if (ret == AVERROR(EAGAIN)) { // Got B-frame (which need to be decoded from both sides)
            return 0;
        }
        else if (ret < 0) {
            fprintf(stderr, "Error decoding video frame (%s)\n", av_err2str(ret)); // Resource temporarily unavailable
            return ret;
        }
        // AV_PIX_FMT_YUV420P
        if (*got_frame) {
            /* If this is the first frame we get, init swscale and set ctx->pix_fmt */
            if (ctx->sws_context == NULL) {
                //printf("color_range: %d\n",ctx->video_dec_ctx->color_range);
                init_swscale(ctx, frame->format);
                ctx->pix_fmt = frame->format;
            }
            if (frame->width != ctx->width || frame->height != ctx->height ||
                frame->format != ctx->pix_fmt) {
                /* To handle this change, one could call av_image_alloc again and
                 * decode the following frames into another rawvideo file. */
                fprintf(stderr, "Error: Width, height and pixel format have to be "
                        "constant in a rawvideo file, but the width, height or "
                        "pixel format of the input video changed:\n"
                        "old: width = %d, height = %d, format = %s\n"
                        "new: width = %d, height = %d, format = %s\n",
                        ctx->width, ctx->height, av_get_pix_fmt_name(ctx->pix_fmt),
                        frame->width, frame->height,
                        av_get_pix_fmt_name(frame->format));
                return -1;
            }
            /*
            printf("video_frame%s n:%d coded_n:%d pts:%s\n",
                   cached ? "(cached)" : "",
                   video_frame_count++, frame->coded_picture_number,
                   av_ts2timestr(frame->pts, &video_dec_ctx->time_base));
                   */
            /* copy decoded frame to destination buffer:
             * this is required since rawvideo expects non aligned data */
            /* YUV420: 1 1/2 1/2 0 The linesize may be larger than the size of usable data – there may be extra padding present for performance reasons. */
            /* In the case of pixel data (RGB24), there is only one plane (data[0]) and linesize[0] == width * channels (320 * 3 for RGB24) */
            //printf("dst linesize: %d %d %d %d %d\n",(ctx->video_dst_linesize)[0],(ctx->video_dst_linesize)[1],(ctx->video_dst_linesize)[2],(ctx->video_dst_linesize)[3],ctx->video_dst_bufsize); // 960 0 0 0 230400
            //printf("frame linesize: %d %d %d %d\n",frame->linesize[0],frame->linesize[1],frame->linesize[2],frame->linesize[3]); // 320 160 160 0
            //av_image_copy(video_dst_data, video_dst_linesize,
            //              (const uint8_t **)(frame->data), frame->linesize,
            //              pix_fmt, width, height);
            //sws_scale(ctx->sws_context, (const uint8_t * const *)(frame->data), frame->linesize, 0, 
            //          ctx->height, ctx->video_dst_data, ctx->video_dst_linesize);
            
            // a * b / c convert frame pts to pts in AV_TIME_BASE
            ctx->current_pts = av_rescale(frame->pts, (int64_t)st->time_base.num * AV_TIME_BASE, st->time_base.den);
            
            /* write to rawvideo file */
            //fwrite(video_dst_data[0], 1, video_dst_bufsize, video_dst_file);
            //fwrite((const uint8_t * const *)(frame->data), 1, video_dst_bufsize, video_dst_file);
        }
    }
    return decoded;
}

static void yuv420p_2_rgb24(struct FastvioCtx* ctx) {
    sws_scale(ctx->sws_context, (const uint8_t * const *)(ctx->frame->data), ctx->frame->linesize, 0, 
              ctx->height, ctx->video_dst_data, ctx->video_dst_linesize);
}

static int open_codec_context(int stream_index, AVCodecContext **dec_ctx, AVFormatContext *fmt_ctx, int thread_mode, int num_threads)
{
    int ret;
    AVStream *st;
    AVCodec *dec = NULL;
    AVDictionary *opts = NULL;

    st = fmt_ctx->streams[stream_index];
    
    /* find decoder for the stream */
    dec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!dec) {
        fprintf(stderr, "Failed to find video codec\n");
        return AVERROR(EINVAL);
    }
    
    /* Allocate a codec context for the decoder */
    *dec_ctx = avcodec_alloc_context3(dec);
    if (!*dec_ctx) {
        fprintf(stderr, "Failed to allocate the video codec context\n");
        return AVERROR(ENOMEM);
    }
    
    /* Copy codec parameters from input stream to output codec context */
    if ((ret = avcodec_parameters_to_context(*dec_ctx, st->codecpar)) < 0) {
        fprintf(stderr, "Failed to copy video codec parameters to decoder context\n");
        return ret;
    }

    if (thread_mode) {
        (*dec_ctx)->thread_type = thread_mode;
        (*dec_ctx)->thread_count = num_threads;

    }
    //(*dec_ctx)->thread_type = FF_THREAD_SLICE; FF_THREAD_FRAME;
    /* Init the decoders, without reference counting */
    av_dict_set(&opts, "refcounted_frames", "0", 0); // No ref counting. Frame will be freed on the next call to decode()
    if ((ret = avcodec_open2(*dec_ctx, dec, &opts)) < 0) {
        fprintf(stderr, "Failed to open video codec\n");
        return ret;
    }
    return 0;
}


void release_decoder(struct FastvioCtx* ctx) {
    sws_freeContext(ctx->sws_context);
    avcodec_free_context(&ctx->video_dec_ctx);
    avformat_close_input(&ctx->fmt_ctx);
    av_frame_free(&ctx->frame);
    //av_free((ctx->video_dst_data)[0]); // we use python memory instead
}

/* Initialize demuxer, decoder, swscale */
int init_decoder(struct FastvioCtx* ctx, int thread_mode, int num_threads) {
    //int ret = 0;
    /* open input file, and allocate format context */
    if (avformat_open_input(&ctx->fmt_ctx, ctx->src_filename, NULL, NULL) < 0) { // 1/10000s
        fprintf(stderr, "Could not open source file %s\n", ctx->src_filename);
        return -1;
    }
    
    /* find the first video stream and discard others */
    ctx->video_stream_idx = -1;
    for (int i = 0; i < ctx->fmt_ctx->nb_streams; i++) {
        AVStream *st = ctx->fmt_ctx->streams[i];
        if (ctx->video_stream_idx == -1 && st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            ctx->video_stream_idx = i;
        }
        else
            st->discard = AVDISCARD_ALL;
    }
    if (ctx->video_stream_idx == -1) {
        fprintf(stderr, "No video stream in the input file\n");
        release_decoder(ctx);
        return -1;
    }
    
    if (open_codec_context(ctx->video_stream_idx, &ctx->video_dec_ctx, ctx->fmt_ctx, thread_mode, num_threads) >= 0) {
        /* allocate image where the decoded image will be put */
        ctx->width = ctx->video_dec_ctx->width;
        ctx->height = ctx->video_dec_ctx->height;
        //printf("w: %d, h: %d, fmt: %d\n",ctx->width, ctx->height, AV_PIX_FMT_YUV420P);
        //ret = av_image_alloc(video_dst_data, video_dst_linesize,
        //                     width, height, pix_fmt, 1);
        
        /* No longer need to allocate storage ad we will use numpy array instead. */
        //ret = av_image_alloc(ctx->video_dst_data, ctx->video_dst_linesize,
        //                     ctx->width, ctx->height, AV_PIX_FMT_RGB24, 1);                  
        //if (ret < 0) {
        //    fprintf(stderr, "Could not allocate raw video buffer\n");
        //    release_decoder(ctx);
        //    return -1;
        //}
        //ctx->video_dst_bufsize = ret;
        ctx->video_dst_linesize[0] = ctx->width * 3;
        ctx->video_dst_linesize[1] = 0;
        ctx->video_dst_linesize[2] = 0;
        ctx->video_dst_linesize[3] = 0;
        ctx->video_dst_bufsize = ctx->video_dst_linesize[0];
    }
    else {
        fprintf(stderr, "Could not find video stream in the input, aborting\n");
        release_decoder(ctx);
        return -1;
    }
    
    //enum AVPixelFormat pix_fmt = ctx->video_dec_ctx->pix_fmt;
    //printf("Pixel Format: %s\n", av_get_pix_fmt_name(pix_fmt));
    
    /* dump input information to stderr */
    //av_dump_format(ctx->fmt_ctx, 0, ctx->src_filename, 0);
    
    /* Init SwScale for converting YUV420 to RGB */
    //ctx->sws_context = sws_getContext(ctx->video_dec_ctx->width, ctx->video_dec_ctx->height, AV_PIX_FMT_YUV420P, // src
    //                            ctx->video_dec_ctx->width, ctx->video_dec_ctx->height, AV_PIX_FMT_RGB24,   // dst
    //                            0, NULL, NULL, NULL);
    
    ctx->frame = av_frame_alloc();
    if (!ctx->frame) {
        fprintf(stderr, "Could not allocate frame\n");
        //ret = AVERROR(ENOMEM);
        release_decoder(ctx);
        return -1;
    }
    /* initialize packet, set data to NULL, let the demuxer fill it */
    av_init_packet(&ctx->pkt);
    (ctx->pkt).data = NULL;
    (ctx->pkt).size = 0;
    
    return 0;
}

/* Initialize FFMPEG instance */
void init_ffmpeg() {
    av_register_all();
    // Set FFmpeg log level
    /*
     *  { "quiet"  , AV_LOG_QUIET   },
     *  { "panic"  , AV_LOG_PANIC   },
     *  { "fatal"  , AV_LOG_FATAL   },
     *  { "error"  , AV_LOG_ERROR   },
     *  { "warning", AV_LOG_WARNING },
     *  { "info"   , AV_LOG_INFO    },
     *  { "verbose", AV_LOG_VERBOSE },
     *  { "debug"  , AV_LOG_DEBUG   },
        { "trace"  , AV_LOG_TRACE   },
     */
    av_log_set_level(AV_LOG_ERROR);
    printf("[FastVIO] init done!\n");
}

PyObject * fastvio_open(PyObject *self, PyObject *args, PyObject *kwargs) {
	char *filename;
	char *thread_mode_str = NULL;
	int num_threads = 0; // auto thread count by default
    static char* kwlist[] = {"","thread_mode","num_threads",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "s|si", kwlist, &filename, &thread_mode_str, &num_threads))
        return NULL;

	//if(!PyArg_ParseTuple(args, "s|si", &filename,&thread_mode,&num_threads))
	//	return NULL;
	//printf("Trying to open %s\n", filename);
    int thread_mode = 0;
	if (thread_mode_str) {
	    if (strcmp(thread_mode_str, "thread_slice") == 0) {
	        thread_mode = FF_THREAD_SLICE;
	    }
        else if (strcmp(thread_mode_str, "thread_frame") == 0) {
            thread_mode = FF_THREAD_FRAME;
        }
        else {
	        PyErr_SetString(MyError, "[FastVIO Open()] Unrecognized thread_mode. Only supports: thread_slice and thread_frame.");
	        return NULL;
        }
        // set num_threads here
	}
	
	
    struct FastvioCtx* ctx = create_fastvio_ctx();
    Py_INCREF(args); // Borrow filename
	ctx->src_filename = filename;
	
	if(init_decoder(ctx, thread_mode, num_threads) < 0) {
	    PyErr_SetString(MyError, "[FastVIO Open()] Error when initializing demuxer and decoder.");
	    return NULL;
	}
	
	
	PyObject* ret = PyLong_FromVoidPtr((void*)ctx); // do not lose it! 
	// Retrive by PyLong_AsVoidPtr()
	return ret;
}

PyObject * fastvio_close(PyObject *self, PyObject *args) {
    PyObject *handle;
    if(!PyArg_ParseTuple(args, "O", &handle))
		return NULL;
	struct FastvioCtx* ctx = (struct FastvioCtx*)PyLong_AsVoidPtr(handle);
	
	release_decoder(ctx);
	free(ctx);
    Py_RETURN_NONE;
}

PyObject * fastvio_grab_frame(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char* kwlist[] = {"","keyframe_only",NULL};
    PyObject *handle;
    int kf_only = 0;
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", kwlist, &handle, &kf_only)) {
        return NULL;
    }
	struct FastvioCtx* ctx = (struct FastvioCtx*)PyLong_AsVoidPtr(handle);        

    int ret = 0;
    int got_frame = 0;
    while (1) {
        if (!TEST_FLAG(ctx->flags, FASTVIO_FLAG_GOT_LAST_FRAME)) {
            while (1) {
                if (!TEST_FLAG(ctx->flags, FASTVIO_FLAG_OUT_OF_PACKET)) {
                    ret = av_read_frame(ctx->fmt_ctx, &ctx->pkt);
                    //printf("         packet: %ld\n",ctx->pkt.pts);
                    if (ret < 0) {
                        SET_FLAG(ctx->flags, FASTVIO_FLAG_OUT_OF_PACKET);
                        (ctx->pkt).data = NULL;
                        (ctx->pkt).size = 0;
                    //    printf("# Out of pocket\n");
                    }
                    if(kf_only && !((ctx->pkt).flags & AV_PKT_FLAG_KEY)) {
                        avcodec_flush_buffers(ctx->video_dec_ctx);
                        continue; // Discard non-keyframe packages
                    }
                }
                if (!TEST_FLAG(ctx->flags, FASTVIO_FLAG_OUT_OF_PACKET)) {
                    //printf("consumed packet: %ld\n",ctx->pkt.pts);
                    ret = decode_packet(ctx, &got_frame, 0);
                    if (ret < 0)
                        break;
                    if (TEST_FLAG(ctx->flags, FASTVIO_FLAG_GOT_FIRST_FRAME) && got_frame) {
                    //    printf("# Got a frame    %ld\n", ctx->frame->pts);
                        break;
                    } // Now haven't got first frame or haven't got any frame
                    else if (got_frame) {
                    //    printf("# Got first frame\n");
                        SET_FLAG(ctx->flags, FASTVIO_FLAG_GOT_FIRST_FRAME);
                        break;
                    }
                    // In this round nothing is returned by the decoder. continue sending package.
                }
                else {
                    decode_packet(ctx, &got_frame, TEST_FLAG(ctx->flags, FASTVIO_FLAG_FLUSHED)); // Flush decoder
                    SET_FLAG(ctx->flags, FASTVIO_FLAG_FLUSHED);
                    //printf("# Flushing \n");
                    if (!got_frame) {
                        SET_FLAG(ctx->flags, FASTVIO_FLAG_GOT_LAST_FRAME);
                    //    printf("# Got last frame \n");
                    }
                    //else
                    //    printf("# Got a frame    %ld\n", ctx->frame->pts);
                    break;
                }
            }
        }
        
        if (kf_only) { // For some reason, some keyframe packets produce non-keyframe frames... Filter them out.
            if (got_frame && !ctx->frame->key_frame)
                continue;
        }
        else {
            if (got_frame && (ctx->current_pts < ctx->seek_target_pts))
                continue;
        }
        break;
    }
    
    if (got_frame) {
        // fwrite(video_dst_data[0], 1, video_dst_bufsize, video_dst_file);
	    npy_intp dims[] = {ctx->height, ctx->width, 3};
	    PyObject *img = PyArray_SimpleNew(3, dims, NPY_UINT8);
	    uint8_t* data_ptr = (uint8_t*)PyArray_BYTES((PyArrayObject*)img);
	    ctx->video_dst_data[0] = data_ptr;
	    ctx->video_dst_data[1] = NULL;
	    ctx->video_dst_data[2] = NULL;
	    ctx->video_dst_data[3] = NULL;
	    
	    //memcpy((void*)data_ptr, (const void*)(ctx->video_dst_data)[0], ctx->video_dst_bufsize);
	    ////memcpy((void*)data_ptr, (const void*)(ctx->video_dst_data)[0], ctx->height*ctx->width*3);
	    yuv420p_2_rgb24(ctx);
	    PyObject *pts = PyLong_FromLongLong(ctx->current_pts);
	    PyObject *is_kf = PyBool_FromLong(ctx->frame->key_frame);
	    PyObject *ret_tup = PyTuple_Pack(3, img, pts, is_kf);
	    Py_DECREF(img);
	    Py_DECREF(pts);
	    Py_DECREF(is_kf);
        return ret_tup;
    }
    else
        Py_RETURN_NONE;
}

/*
 * Here we have a possible optimization: do not actually seek if current_pts < target_pts < next_keyframe_pts
 * However, seems that FFmpeg doesn't have a api for getting keyframe pts.
 */
PyObject * fastvio_seek(PyObject *self, PyObject *args) {
    PyObject* handle;
    PyObject* pts_obj;
    if(!PyArg_ParseTuple(args, "OO", &handle, &pts_obj))
		return NULL;
	struct FastvioCtx* ctx = (struct FastvioCtx*)PyLong_AsVoidPtr(handle);
	AVStream *st = ctx->fmt_ctx->streams[ctx->video_stream_idx];
	int64_t target_pts = PyLong_AsLongLong(pts_obj);
    
    int64_t target_pts_stream = av_rescale(target_pts, st->time_base.den, AV_TIME_BASE * (int64_t) st->time_base.num);
    
    int ret;
	ret = av_seek_frame(ctx->fmt_ctx, ctx->video_stream_idx, target_pts_stream, AVSEEK_FLAG_BACKWARD);
	if (ret < 0) {
	    PyErr_SetString(MyError, "Seek failed.");
	    return NULL;
	}
	
	ctx->seek_target_pts = target_pts;
	avcodec_flush_buffers(ctx->video_dec_ctx);
	CLEAR_FLAG(ctx->flags, FASTVIO_FLAG_FLUSHED);
	CLEAR_FLAG(ctx->flags, FASTVIO_FLAG_GOT_LAST_FRAME);
	CLEAR_FLAG(ctx->flags, FASTVIO_FLAG_OUT_OF_PACKET);
	CLEAR_FLAG(ctx->flags, FASTVIO_FLAG_GOT_FIRST_FRAME);
    Py_RETURN_NONE;
}

/*
 * Get the duration of video, expressed in av_timebase (usually us).
 * Sorry but there is no easy way to get the exact length of a video.
 */
PyObject * fastvio_get_duration(PyObject *self, PyObject *args) {
    PyObject* handle;
    if(!PyArg_ParseTuple(args, "O", &handle))
		return NULL;
	struct FastvioCtx* ctx = (struct FastvioCtx*)PyLong_AsVoidPtr(handle);

    //AVStream *st = ctx->fmt_ctx->streams[ctx->video_stream_idx];
    
    int64_t duration = ctx->fmt_ctx->duration;
    //duration = st->duration;
    
    //  av_rescale: a * b / c,
    // int64_t timestamp = av_rescale(timestamp, st->time_base.den, AV_TIME_BASE * (int64_t) st->time_base.num);
	
    PyObject* ret = PyLong_FromLongLong(duration);
	return ret;
}

PyObject * fastvio_print_dbg(PyObject *self, PyObject *args) {
    PyObject* handle;
    if(!PyArg_ParseTuple(args, "O", &handle))
		return NULL;
	struct FastvioCtx* ctx = (struct FastvioCtx*)PyLong_AsVoidPtr(handle);

    AVStream *st = ctx->fmt_ctx->streams[ctx->video_stream_idx];
    
    av_dump_format(ctx->fmt_ctx, 0, ctx->src_filename, 0);
    
    if (ctx->fmt_ctx->iformat->read_seek)
        printf("FAST SEEKING AVAILABLE!!!\n");
    if (ctx->fmt_ctx->iformat->read_seek2)
        printf("FAST SEEKING PLUS AVAILABLE!!!\n");
    if (ctx->fmt_ctx->iformat->read_timestamp)
        printf("read_timestamp available!\n");
    
    printf("Matroska nb_index_entries: %d\n", ctx->fmt_ctx->streams[ctx->video_stream_idx]->nb_index_entries);
    AVIndexEntry* entries = ctx->fmt_ctx->streams[ctx->video_stream_idx]->index_entries;
	for (int i=0;i<ctx->fmt_ctx->streams[ctx->video_stream_idx]->nb_index_entries;i++) {
	    int64_t timestamp = entries[i].timestamp;
	    int64_t timestamp_avtimebase = av_rescale(timestamp, AV_TIME_BASE * (int64_t) st->time_base.num, st->time_base.den);
	    printf("Matroska index_entries[%d].timestamp: %ld(%ld), pos: %ld\n", i, timestamp, timestamp_avtimebase, entries[i].pos);
	}

    Py_RETURN_NONE;
}

//    int got_frame = 0;
//    /* read frames from the file */
//    while (av_read_frame(ctx->fmt_ctx, &ctx->pkt) >= 0) {
//        ret = decode_packet(ctx, &got_frame, 0);
//        if (ret < 0)
//            break;
//        av_packet_unref(&ctx->pkt);
//        //printf("frame_pts: %ld\n", frame->pts);
//    }
//    /* flush cached frames */
//    (ctx->pkt).data = NULL;
//    (ctx->pkt).size = 0;
//    do {
//        decode_packet(ctx, &got_frame, 1);
//    } while (got_frame);
    
