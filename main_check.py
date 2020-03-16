from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2
home_ = os.getcwd()



# test_path = home_ + "/" + "images_data_IIT_Students"

# for i in os.listdir(test_path):
#     for j in os.listdir(test_path+"/"+i):
#         path = test_path+"/"+i+"/"+j

# photo_image = 6.jpeg"
# photo_image = home_ + "/" + "samples" + "/" + "5.png"

# crop_save_folder_horizontal = "cropped_images_horizontal"
crop_save_folder_horizontal = r"C:\Users\Avadhesh Kumar\Desktop\images\cropped_images_horizontal"
crop_save_folder_vertical = r"C:\Users\Avadhesh Kumar\Desktop\images\cropped_images_vertical"
edge_folder_horizontal = r"C:\Users\Avadhesh Kumar\Desktop\images\edge_folder_horizontal"
edge_folder_vertical = r"C:\Users\Avadhesh Kumar\Desktop\images\edge_folder_vertical"
save_folder_horizontal = r"C:\Users\Avadhesh Kumar\Desktop\images\cropped_images_horizontal"
save_folder_vertical = r"C:\Users\Avadhesh Kumar\Desktop\images\cropped_images_vertical"

class new_check:

   
    def crop_func_seperate(self,photo_image):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        print("Face-cas",face_cascade)
        try:
            img = cv2.imread(photo_image)
        #     print("img",img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[:2]
            #cv2.imshow('result',gray)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            print(len(faces))
            if len(faces) == 0:
                face_detection_out = "No-Face-Detected"
                # print(face_detection_out)
                pass
            else:
                face_detection_out = "Face_Detected"
                for (a, b, c, d) in faces:
                    cv2.rectangle(img, (a, b), (a+c, b+d), (255, 0, 0), 2)
                    im = Image.open(photo_image)
                    val = a+c
                    val1 = b+d
                try:

                    im1 = im.crop((val, b, width, 1280))  # right side
                    im2 = im.crop((a, 0, (a+c), b))  # top
                    im3 = im.crop((0, b, a, 1280))  # left side
                    im4 = im.crop((a, (b+d), (a+c), height)) #bottom
                    im1.save(crop_save_folder_vertical + "\\" + "right.jpg")
                    im2.save(crop_save_folder_horizontal + "\\" + "top.jpg")
                    im3.save(crop_save_folder_vertical + "\\" + "left.jpg")
                    im4.save(crop_save_folder_horizontal + "\\" + "bottom.jpg")
                except SystemError:
                    pass
                except:
                    face_detection_out = "No-Face-Detected"

                       
                return face_detection_out


    def canny_edges(self,save_folder_horizontal, save_folder_vertical):    
        flag = 1
        for images in os.listdir(save_folder_horizontal):   
            img = cv2.imread(save_folder_horizontal+"\\"+images, 0)
        #         print("img-print:",img)
            edges = cv2.Canny(img, 100, 200)
            cv2.imwrite(edge_folder_horizontal + "\\"+images, edges)
        for images in os.listdir(save_folder_vertical):
            img = cv2.imread(save_folder_vertical+"\\"+images, 0)
            edges = cv2.Canny(img, 100, 200)
            cv2.imwrite(edge_folder_vertical + "\\"+images, edges)
    # canny_edges(save_folder_horizontal,save_folder_vertical)


    # # In[334]:

    def Draw_HV_Lines_and_Compare(self,face_detected, edge_folder_horizontal, edge_folder_vertical):
        h_path = 'C://Users//Avadhesh Kumar//Desktop//images//horizontal_lines//'
        v_path = 'C://Users//Avadhesh Kumar//Desktop//images//vertical_lines//'
        horv_length = []
        horv_lines = []
        vflag = 0
        hflag = 0
        hflag_list = []
        vflag_list = []
        lh = []
        lv = []

        for image in os.listdir(edge_folder_horizontal):
            gray = cv2.imread(edge_folder_horizontal + "/"+image)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            minLineLength = 100
            lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=100,
                                    lines=np.array([]), minLineLength=minLineLength, maxLineGap=80)
            if lines is not None:
                a, b, c = lines.shape
                for i in range(a):
                    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i]
                                                                      [0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.imwrite(h_path+image, gray)
            else:
                hflag = 0
                hflag_list.append(hflag)
                # print("118 #### Original Image")

        for img in os.listdir(h_path):
            # print("# # 111 ###",img)
            gray = cv2.imread(h_path+img)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            minLineLength = 100
            lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=100,
                                    lines=np.array([]), minLineLength=minLineLength, maxLineGap=80)
            if lines is not None:
                for i in range(len(lines)):
                    horv_line = lines.reshape(len(lines), 4)[i]
                    horv_lines.append(horv_line)
                    horv_length.append(abs(horv_line[0] - horv_line[2]))
                    index_max_h_line = np.argmax(horv_length)
                    im = Image.open(h_path+"\\"+img)
                    width, height = im.size
                    half_size_of_image = width//2
                    longest_horizontal_line = horv_lines[index_max_h_line]
                    longest_horizontal_line_distance = abs(
                        longest_horizontal_line[0] - longest_horizontal_line[2])
                    lh.append(longest_horizontal_line_distance)
                h = max(lh)
                if (h > half_size_of_image):
                    # print("inside if loop")
                    hflag = 1
                    hflag_list.append(hflag)
                else:
                    hflag = 0
                    hflag_list.append(hflag)
        for image in os.listdir(edge_folder_vertical):
            gray = cv2.imread(edge_folder_vertical + "\\"+image)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            minLineLength = 100
            lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=100,
                                    lines=np.array([]), minLineLength=minLineLength, maxLineGap=80)
            if lines is not None:
                a, b, c = lines.shape
                for i in range(a):
                    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i]
                                                                      [0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.imwrite(v_path+image, gray)
            else:
                vflag = 0
                vflag_list.append(vflag)
        for img in os.listdir(v_path):
            gray = cv2.imread(v_path+img)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            minLineLength = 100
            lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=100,
                                    lines=np.array([]), minLineLength=minLineLength, maxLineGap=80)
            if lines is not None:
                for i in range(len(lines)):
                    horv_line = lines.reshape(len(lines), 4)[i]
                    horv_lines.append(horv_line)
                    horv_length.append(abs(horv_line[1] - horv_line[3]))
                    index_max_v_line = np.argmax(horv_length)
                    im = Image.open(v_path+"\\"+img)
                    width, height = im.size
                    half_size_of_image = height//2
                    longest_verti_line = horv_lines[index_max_v_line]
                    longest_vertical_line_distance = abs(
                        longest_verti_line[1] - longest_verti_line[3])
                    lv.append(longest_vertical_line_distance)
                v = max(lv)
                # print(v)
                if (v > half_size_of_image):
                    # print("inside if loop")
                    vflag = 1
                    vflag_list.append(vflag)
                else:
                    vflag = 0
                    vflag_list.append(vflag)
        horizontal_count = hflag_list.count(1)
        vertical_count = vflag_list.count(1)
        if face_detected != "No-Face-Detected":
            final_count = horizontal_count + vertical_count
            # print("Final_count:", final_count)
            if final_count > 2:
                output = "Photo of a Photo"
                # print("Output is:--->    Photo of Photo")
            elif face_detected == "No-Face Detected":
                output = "No-Face-Detected"
            else:
                output = "Original Photo"
                print("Output is:--->    Original Photo")
            return output

    def delete_files_after_check(self):
        import os
        import glob
        home_ = os.getcwd()
        try:
            for folder in os.listdir(home_):
                print("folders", folder)
                if folder == "cropped_images_horizontal" or folder == "cropped_images_vertical" or folder == "edge_folder_horizontal" or folder == "edge_folder_vertical" or folder == "horizontal_lines" or folder == "vertical_lines":
                    print("dirs:", folder)
                    content = glob.glob(home_ + "\\"+folder)
            #        print("contents",content[0])
                    for (root, dirs, files) in os.walk(content[0]):
                        if files != []:
                            print(folder)
                            os.remove(folder+"\\"+files[0])
                            os.remove(folder+"\\"+files[1])
                else:
                    pass
        except:
            pass


    # def write_data(photo_name, face_detected, output):

    #     import logging
    #     import time
    #     import os
    #     photo_of_photo = "photo_photo_out"
    #     logging.basicConfig(filename=photo_of_photo, format='%(asctime)s - %(message)s',
    #                         datefmt='%d-%b-%y %H:%M:%S', filemode='w+', level=logging.INFO)
    #     # start_time = time.time()
    #     # time_taken = time.time() - start_time
    #     logging.info('Image-Name : {}-------Face_Check_output : {}------Image-output : {}'.format(photo_name, face_detected, output))


    #test_path = r"C:\Users\Avadhesh Kumar\Desktop\images\img1"
    def test(self,test_path):
        photo_image = test_path
        #cv2.imshow('output',photo_image)
        #print("photo_image:", photo_image)
        face_detected = self.crop_func_seperate(photo_image)
        self.canny_edges(save_folder_horizontal, save_folder_vertical)
        output = self.Draw_HV_Lines_and_Compare(face_detected,edge_folder_horizontal, edge_folder_vertical)
        return output


test_path = r"C:/Users/hp/Desktop/face/test1.jpg"
obj_new = new_check()
obj_new.test(test_path)