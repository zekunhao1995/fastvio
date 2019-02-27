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
        ret = avcodec_send_packet(video_dec_ctx, pkt);
        if (ret < 0) {
            fprintf(stderr, "Error sending package (%s)\n", av_err2str(ret));
            return ret;
        }
        ret = avcodec_receive_frame(video_dec_ctx, frame);
        if (ret == 0)
            *got_frame = 1;
        else if (ret == AVERROR_EOF)
            return 0;
        else if (ret < 0) {
            fprintf(stderr, "Error decoding video frame (%s)\n", av_err2str(ret));
            return ret;
        }
        if (*got_frame) {
            if (frame->width != ctx->width || frame->height != ctx->height ||
                frame->format != AV_PIX_FMT_YUV420P) {
                /* To handle this change, one could call av_image_alloc again and
                 * decode the following frames into another rawvideo file. */
                fprintf(stderr, "Error: Width, height and pixel format have to be "
                        "constant in a rawvideo file, but the width, height or "
                        "pixel format of the input video changed:\n"
                        "old: width = %d, height = %d, format = %s\n"
                        "new: width = %d, height = %d, format = %s\n",
                        ctx->width, ctx->height, av_get_pix_fmt_name(AV_PIX_FMT_YUV420P),
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

static int open_codec_context(int stream_index, AVCodecContext **dec_ctx, AVFormatContext *fmt_ctx)
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
int init_decoder(struct FastvioCtx* ctx) {
    int ret = 0;
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
    
    if (open_codec_context(ctx->video_stream_idx, &ctx->video_dec_ctx, ctx->fmt_ctx) >= 0) {
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
    
    /* dump input information to stderr */
    //av_dump_format(ctx->fmt_ctx, 0, ctx->src_filename, 0);
    
    /* Init SwScale for converting YUV420 to RGB */
    ctx->sws_context = sws_getContext(ctx->video_dec_ctx->width, ctx->video_dec_ctx->height, AV_PIX_FMT_YUV420P, // src
                                ctx->video_dec_ctx->width, ctx->video_dec_ctx->height, AV_PIX_FMT_RGB24,   // dst
                                0, NULL, NULL, NULL);
    
    ctx->frame = av_frame_alloc();
    if (!ctx->frame) {
        fprintf(stderr, "Could not allocate frame\n");
        ret = AVERROR(ENOMEM);
        release_decoder(ctx);
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
    printf("[FastVIO] init done!\n");
}

PyObject * fastvio_open(PyObject *self, PyObject *args) {
	char *filename;
	if(!PyArg_ParseTuple(args, "s", &filename))
		return NULL;
	//printf("Trying to open %s\n", filename);
	
    struct FastvioCtx* ctx = create_fastvio_ctx();
    Py_INCREF(args); // Borrow filename
	ctx->src_filename = filename;
	
	if(init_decoder(ctx) < 0) {
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

PyObject * fastvio_grab_frame(PyObject *self, PyObject *args) {
    PyObject *handle;
    if(!PyArg_ParseTuple(args, "O", &handle))
		return NULL;
	struct FastvioCtx* ctx = (struct FastvioCtx*)PyLong_AsVoidPtr(handle);
    
    int ret = 0;
    int got_frame = 0;
    do {
        if (!TEST_FLAG(ctx->flags, FASTVIO_FLAG_GOT_LAST_FRAME)) {
            while (1) {
                if (!TEST_FLAG(ctx->flags, FASTVIO_FLAG_OUT_OF_PACKET)) {
                    ret = av_read_frame(ctx->fmt_ctx, &ctx->pkt);
                    if (ret < 0) {
                        SET_FLAG(ctx->flags, FASTVIO_FLAG_OUT_OF_PACKET);
                        (ctx->pkt).data = NULL;
                        (ctx->pkt).size = 0;
                    }
                }
                if (!TEST_FLAG(ctx->flags, FASTVIO_FLAG_OUT_OF_PACKET)) {
                    ret = decode_packet(ctx, &got_frame, 0);
                    if (ret < 0)
                        break;
                    if (TEST_FLAG(ctx->flags, FASTVIO_FLAG_GOT_FIRST_FRAME) && got_frame) {
                        break;
                    } // Now haven't got first frame or haven't got any frame
                    else if (got_frame) {
                        SET_FLAG(ctx->flags, FASTVIO_FLAG_GOT_FIRST_FRAME);
                        break;
                    }
                    continue;
                }
                else {
                    decode_packet(ctx, &got_frame, 1);
                    if (!got_frame) {
                        SET_FLAG(ctx->flags, FASTVIO_FLAG_GOT_LAST_FRAME);
                    }
                    break;
                }
            }
        }
    } while (got_frame && (ctx->current_pts < ctx->seek_target_pts));
    
    if (got_frame) {
        // fwrite(video_dst_data[0], 1, video_dst_bufsize, video_dst_file);
	    npy_intp dims[] = {ctx->height, ctx->width, 3};
	    PyObject *img = PyArray_SimpleNew(3, dims, NPY_UINT8);
	    char* data_ptr = PyArray_BYTES((PyArrayObject*)img);
	    ctx->video_dst_data[0] = data_ptr;
	    ctx->video_dst_data[1] = NULL;
	    ctx->video_dst_data[2] = NULL;
	    ctx->video_dst_data[3] = NULL;
	    
	    //memcpy((void*)data_ptr, (const void*)(ctx->video_dst_data)[0], ctx->video_dst_bufsize);
	    ////memcpy((void*)data_ptr, (const void*)(ctx->video_dst_data)[0], ctx->height*ctx->width*3);
	    yuv420p_2_rgb24(ctx);
	    PyObject *pts = PyLong_FromLongLong(ctx->current_pts);
	    PyObject *ret_tup = PyTuple_Pack(2, img, pts);
	    Py_DECREF(img);
	    Py_DECREF(pts);
        return ret_tup;
    }
    else
        Py_RETURN_NONE;
}

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

    AVStream *st = ctx->fmt_ctx->streams[ctx->video_stream_idx];
    
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
    
