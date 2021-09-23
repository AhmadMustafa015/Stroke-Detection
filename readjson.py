import json
import cv2
import numpy as np

jsonpath = 'C:/Users/RadioscientificOne/PycharmProjects/Stroke-Detection/Baturalp_labels/train-2.json'
with open(jsonpath) as f:
    data = json.load(f)
    imagess = data["labels"]
    im_fold_path = 'C:/Users/RadioscientificOne/PycharmProjects/Stroke-Detection/Baturalp_labels/Selected PNG/Selected PNG/'
    im_shape = (512,512)
    for i,image in enumerate(imagess):


        orig_name = imagess[i]["dataId"] #+ '.png'
        orig_image_path = im_fold_path + orig_name
        final_image = np.zeros(im_shape)
        annots = imagess[i]["annotations"]
        intra_areas = annots["Intraparenchymal"]
        subarach_areas = annots["Subarachnoid"]
        try:
            iskemi_areas = annots["Ischemia"]
        except:
            pass

        for i,objs in enumerate(intra_areas):

            indices_list = intra_areas[i]["data"]["points"]
            image_indices = np.int32(np.array(indices_list)*512)

            filled = np.zeros(im_shape)
            filled = cv2.fillPoly(filled, pts=np.int32([image_indices]), color=(255, 255, 255))
            filled[filled > 0] = 2
            final_image = final_image + filled

        for i,objs in enumerate(subarach_areas):


            indices_list = subarach_areas[i]["data"]["points"]
            image_indices = np.int32(np.array(indices_list)*512)

            filled = np.zeros(im_shape)
            filled = cv2.fillPoly(filled, pts=np.int32([image_indices]), color=(255, 255, 255))
            filled[filled > 0] = 2
            final_image = final_image + filled

        for i, objs in enumerate(iskemi_areas):

            indices_list = iskemi_areas[i]["data"]["points"]
            image_indices = np.int32(np.array(indices_list) * 512)

            filled = np.zeros(im_shape)
            filled = cv2.fillPoly(filled, pts=np.int32([image_indices]), color=(255, 255, 255))
            filled[filled > 0] = 1
            final_image = final_image + filled

        origimage = cv2.imread(orig_image_path)
        '''
        mask = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)
        final2 = cv2.addWeighted(mask, 0.5, origimage, 1 - 0.5, 0, origimage)
        '''
        final_image[final_image > 2] = 0

        writepath = 'C:/Users/RadioscientificOne/PycharmProjects/Stroke-Detection/Baturalp_labels/Selected MASK/'+ 'baturalp@'+  orig_name
        cv2.imwrite(writepath,final_image)
        #cv2.imshow('image1', origimage )
        #cv2.imshow('image', final_image)
        #cv2.imshow('image2', final2)
        #cv2.waitKey(0)
        #c =1


#b =1