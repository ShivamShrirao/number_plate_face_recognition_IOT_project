import os
from PIL import Image

lbl = open("labels.csv","w")
lbl.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")

"""
GENERATE NEGATIVE IMAGES TOO NOW
"""

lsdir=os.listdir("../number_plates/")
for idx,file in enumerate(lsdir):
	if file.endswith(".jpg"):
		print("\r",idx+2,"/",len(lsdir),end="")
		width,height=map(str,Image.open("../number_plates/"+file).size)
		with open("../number_plates/"+file[:-3]+"txt","r") as f:
			b=list(map(int,f.read().split()[1:5]))
		xmin=str(b[0])
		ymin=str(b[1])
		xmax=str(b[0]+b[2])
		ymax=str(b[1]+b[3])
		lbl.write(file+","+width+","+height+","+"number_plate"+","+xmin+","+ymin+","+xmax+","+ymax+"\n")

print()
lbl.close()