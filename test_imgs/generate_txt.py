import os

samples_path = "samples-will/image"
images = os.listdir(samples_path)
f = open("samples-will/images_list.txt","w")
for i in range(len(images)):
	f.write(images[i]+"\n")
f.close()
