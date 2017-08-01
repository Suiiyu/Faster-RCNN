# get the gt's x_min,y_min,x_max,y_max and replace the xml's values
# first traverse the image to find the pixel==255's position
# second find the x_min,y_min,x_max,y_max
# third read the xml file and replace the values
# by LYS 6/28/2017 
from PIL import Image,ImageDraw
import xml.etree.cElementTree as ET
import os

def get_positions(image_path):
    im = Image.open(image_path).convert('L')# convert to gray image
    draw = ImageDraw.Draw(im)
    width = im.size[0]
    height = im.size[1]
    x = []
    y = []
  
    for w in range(0, width):
        for h in range(0, height):
	    pixel = im.getpixel((w, h))
	    if pixel>=120.0:
	            x.append(w)
	            y.append(h)
    
    if(len(x)!=0 and len(y)!=0):
	for i in x[::-1]:
	    if i < 50 or i >300:
		x.remove(i)
		continue
    
	for j in y[::-1]:
	    if j < 50 or j >300:
		y.remove(j)
		continue
    	x_min = min(x)
    	y_min = min(y)
    	x_max = max(x)
    	y_max = max(y)
	    
    	box = [x_min, y_min, x_max, y_max]
	    #crop_image(box)
      #draw.rectangle(box,outline=255)
      #im.save(image_path)
	
    	modify_xmlfile(box, xml_path, im)
	
#   	print  x_min,y_min,x_max,y_max
#     im.show()

def modify_xmlfile(box, xml_path, image):
    tree = ET.ElementTree(file='000001.xml')
    for elem in tree.iter(tag = 'xmin'):
	      elem.text = `box[0]`
    for elem in tree.iter(tag = 'ymin'):
	      elem.text = `box[1]`
    for elem in tree.iter(tag = 'xmax'):
	      elem.text = `box[2]`
    for elem in tree.iter(tag = 'ymax'):
	      elem.text = `box[3]`
    for elem in tree.iter(tag = 'filename'):
	      elem.text = image_name.split('.')[0]
    for elem in tree.iter(tag = 'width'):
	      elem.text = `image.size[0]`
    for elem in tree.iter(tag = 'height'):
	      elem.text = `image.size[1]`
    tree.write(xml_path)

def crop_image(box):
    image_cropped = Image.open(os.path.join('/home/lys/lys/deform/aug_merge/',image_name))
    crop_saveDir = os.path.join('/home/lys/lys/deform/crop/',image_name)
    crop_im = image_cropped.crop(box)
    crop_im.save(crop_saveDir)
    
if __name__ == '__main__':
    image_read_path = '/home/lys/lys/tes012/aug_gt/'
    xml_save_path = '/home/lys/lys/tes012/ann_aug/'
    if not os.path.lexists(xml_save_path):
	      os.mkdir(xml_save_path)
    image_names = os.listdir(image_read_path)
    for image_name in image_names:
	      if image_name[-3:]=='jpg':
	          xml_name = image_name.split('.')[0] + '.xml'
	          xml_path = os.path.join(xml_save_path, xml_name)
	          image_path = os.path.join(image_read_path, image_name)
	          get_positions(image_path)
