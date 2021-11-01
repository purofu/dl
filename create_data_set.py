import os
import shutil

directory = 'set/'
des_0 = 'set/0/'
des_1 = 'set/1/'

for filename in os.listdir(directory):
	if filename.endswith("0.jpg"):
		shutil.move(directory+filename, des_0 + filename)
	elif filename.endswith("1.jpg"):
		shutil.move(directory+filename, des_1 + filename)
