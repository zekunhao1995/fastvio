#!/usr/bin/env python3

from distutils.core import setup, Extension

module1 = Extension("fastvio",
                    include_dirs = ['/usr/include/x86_64-linux-gnu'],
                    library_dirs = ['/usr/lib/x86_64-linux-gnu'],
                    libraries = ['avformat','avfilter','avcodec','avdevice','swresample','swscale','avutil','z','m'],
                    sources = ["bind.c", "fastvio.c"])

setup(
	name = "fastvio",
	version = "0.1",
	ext_modules = [module1]
	);
