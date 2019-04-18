from distutils.core import setup, Extension
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
setup(name = "ompdModule", version = "1.0", \
	ext_modules = [Extension("ompdModule", [dir_path+'/ompdModule.c'])])
