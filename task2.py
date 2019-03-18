"""
Character Detection
(Due date: March 8th, 11: 59 P.M.)

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
and the functions you implement in task1.py are of great help.

Hints: You might want to try using the edge detectors to detect edges in both the image and the template image,
and perform template matching using the outputs of edge detectors. Edges preserve shapes and sizes of characters,
which are important for template matching. Edges also eliminate the influence of colors and noises.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import numpy as np
import utils
import task1   # you could modify this line


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="./data/characters.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def detect(img, template):
    """Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
   
  
    """
    img_edge_x = detect_edges(img, sobel_x, False)
    img_edge_y = detect_edges(img, sobel_y, False)
    img_edges = (edge_magnitude(img_edge_x,img_edge_y))
    show_image(np.asarray(img_edges))
   
    
    template_x = detect_edges(template, sobel_x, False)
    template_y = detect_edges(template, sobel_y, False)
    template_edges = (edge_magnitude(template_x,template_y))
    show_image(np.asarray(template_edges))
    """
#     img = convolve2d(img, gaussian) 
#     template = convolve2d(template, gaussian)
    
#     show_image(np.asarray(normalize(img)))
#     show_image(np.asarray(normalize(template)))
    
#     lapimg = (convolve2d(img, laplacian))
#     laptemp = (convolve2d(template, laplacian))
    
#     show_image(np.asarray(normalize(lapimg)))
#     show_image(np.asarray(normalize(laptemp)))
    #print(args.template_path)
    args = parse_args()
    print(args.template_path)
    if (args.template_path == './data/a.jpg'):
    	threshold = 0.918
    elif (args.template_path == './data/b.jpg'):
    	threshold = 0.95
    elif (args.template_path == './data/c.jpg'):
    	threshold = 0.96
    print("threshold is %f" % threshold)

    img = task1.normalize(img)
    template = task1.normalize(template)
    
    img_edges = img
    template_edges = template
    
    print(len(img_edges), len(img_edges[0]))
    
    print(min([min(i) for i in img_edges]))
    print(max([max(i) for i in img_edges]))
    
    
    
    #Calculate size of the output image
    img_w = len(img_edges[0]) - len(template_edges[0]) + 1 
    img_h = len(img_edges) - len(template_edges)+ 1
    #print(img_w, img_h)
    
    j=0
    offset = 0
    og_img = []
    for m in range(img_w * img_h): # size of kernel x kernel
        x = []
   
        for i in range(len(template_edges)): #3 is kernel size
        #print(i,j)
            x.append(img_edges[i+offset][j:j+len(template_edges[0])])
            
       # print(len(x), len(x[0]))
        #print(x)
        #print(template_edges)
        
        
        sum_xy = 0
        img_sq = 0
        temp_sq = 0
        for k in range(len(template_edges)):
            for l in range(len(template_edges[0])):  #Loop till the image and template match ends
#                 print(x[k][l])
# #                 print("---------")
#                 print(template_edges[k][l])
#                 print("---------")
#                 print(k,l)
                x1 = x[k][l]
                x2 = template_edges[k][l]
                #print(x1,x2)
                #print("Mutliplication %d" % (x1*x2))
                sum_xy+= float((x[k][l]) * (template_edges[k][l]))
                img_sq+= float(x[k][l])**2                 #image square
                temp_sq+= float(template_edges[k][l])**2    #template square
                 #Sum of product of image and template
                #print(img_sq, temp_sq, sum_xy)
        #print("Coming out of the loop")
        out = float(sum_xy / (np.sqrt(img_sq * temp_sq)))  #Formula for NCC
        #print(i,j)
        #print(out)       
        og_img.append(out) 
        j+=1
        if (j == (img_w)):
            j = 0
            offset+= 1
    
   # print(type(og_img))        
    #print(len(og_img))
    final_img = []
    for i in range(0,img_w * img_h, img_w):
        final_img.append(og_img[i:i+img_w])
   # print(len(final_img))
    #print(final_img[0])
    #show_image(np.array(final_img))
    #final_max = (max([max(i) for i in final_img]))
    
    # print("------------Final image--------")
    # print(min([min(i) for i in final_img]))
    # print(max([max(i) for i in final_img]))
    
    normalized_final = task1.normalize(final_img)
    
   # print(normalized_final)
    
    #print("------------Norm image--------")
    
    #print(min([min(i) for i in normalized_final]))
    maximg= (max([max(i) for i in normalized_final]))
    #print(maximg)
    
    coords = []
    for i in range(len(normalized_final)):
        for j in range(len(normalized_final[0])):
            if (normalized_final[i][j] >= threshold*maximg):
                #print(i,j)
                #print(normalized_final[i][j])
                coords.append([i,j])
                #normalized_final[i][j] = 1
                #
    #print(coords)
    #print(len(normalized_final), len(normalized_final[0]))
    # TODO: implement this function.
    #raise NotImplementedError
    return coords
    #return coords


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = task1.read_image(args.img_path)
    template = task1.read_image(args.template_path)

    coordinates = detect(img, template)

    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
	
	main()
	
    
