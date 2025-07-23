import albumentations as A
import cv2
import json
import os
import glob
from pathlib import Path
import random
from tqdm import tqdm

def augmentation_geo(dir_img, dir_labels):
    for image_path in tqdm(glob.glob(dir_img + '/*.jpg'), desc="Geometric augmentation"):
        image_raw = cv2.imread(image_path)
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        file_name = Path(image_path).stem
        annotation = os.path.join(dir_labels, file_name + '.json')

        with open(annotation) as json_file:
            data = json.load(json_file)

        keypoints = []
        for i in range(len(data['keypoints'])):
            keypoints.append(data['keypoints'][i][0])

        bboxes = []
        for i in range(len(data['bboxes'])):
            bboxes.append(data['bboxes'][i])

        transformation = random.choice([-1, 0, 1, -99])

        if transformation == -99:
            continue

        transformed_image = cv2.flip(image_raw, transformation)
        transformed_bboxes = []
        transformed_keypoints = []

        if transformation == -1:
            for bbox in bboxes:
                transformed_bboxes.append((image_raw.shape[0]-bbox[2],image_raw.shape[0]-bbox[3],image_raw.shape[0]-bbox[0],image_raw.shape[0]-bbox[1]))
            for kds in keypoints:
                transformed_keypoints.append(([image_raw.shape[0]-kds[0],image_raw.shape[0]-kds[1],1]))
        if transformation == 0:
            for bbox in bboxes:
                transformed_bboxes.append((bbox[0],image_raw.shape[0]-bbox[3],bbox[2],image_raw.shape[0]-bbox[1]))
            for kds in keypoints:
                transformed_keypoints.append(([kds[0],image_raw.shape[0]-kds[1],1]))
        if transformation == 1:
            for bbox in bboxes:
                transformed_bboxes.append((image_raw.shape[0]-bbox[2],bbox[1],image_raw.shape[0]-bbox[0],bbox[3]))
            for kds in keypoints:
                transformed_keypoints.append(([image_raw.shape[0]-kds[0],kds[1],1]))

        cv2.imwrite(dir_img + '/' + file_name + '.jpg',
                    cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
        keypoints_format = []
        for kps in transformed_keypoints:
            keypoints_format.append([kps])
        aDict = {'bboxes': transformed_bboxes, 'keypoints': keypoints_format}
        jsonString = json.dumps(aDict)
        jsonFile = open(dir_labels + '/' + file_name + '.json', "w")
        jsonFile.write(jsonString)
        jsonFile.close()

def augmentation(path, number_of_images):
    dir_crop = path + '/augmented'
    if not os.path.exists(dir_crop):
        os.mkdir(dir_crop)

    dir_img = dir_crop + '/images'
    if not os.path.exists(dir_img):
        os.mkdir(dir_img)

    dir_labels = dir_crop + '/labels'
    if not os.path.exists(dir_labels):
        os.mkdir(dir_labels)
    counter = 000
    for image_path in tqdm(glob.glob(path + '/*.jpg'), desc="Augmenting images"):
        try:
            image_raw = cv2.imread(image_path)
            image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            file_name = Path(image_path).stem
            annotation = os.path.join(Path(path).parent, 'labels', file_name + '.json')

            with open(annotation) as json_file:
                data = json.load(json_file)
            keypoints = []
            for i in range(len(data['keypoints'])):
                keypoints.append(data['keypoints'][i])
            bboxes = []
            for i in range(len(data['bboxes'])):
                bboxes.append(data['bboxes'][i])

            for i in range(number_of_images):
                transform = A.Compose([
                    A.ToGray(p=0.05),
                    A.Sharpen(p=0.2),
                    A.FancyPCA(p=0.3),
                    A.OneOf([A.RandomRain(blur_value=3, p=0.2), A.RandomRain(blur_value=1, p=0.4)], p=0.2),
                    A.RandomShadow(p=0.05),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.1),
                    A.GaussianBlur(blur_limit=[1, 3], sigma_limit=0, p=0.1),
                    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.3),
                    A.RandomFog(fog_coef_lower=0, fog_coef_upper=0.1, alpha_coef=0.05, always_apply=False, p=0.1),
                    # A.RandomScale(scale_limit=(-0.3, 0.3), p=0.2),
                    A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, shear=[-15, 15],
                             interpolation=1,
                             mask_interpolation=0, cval_mask=0, mode=1, keep_ratio=False, always_apply=False,
                             p=0.6),
                    A.Perspective(scale=(0.01, 0.3), keep_size=True, pad_mode=1, pad_val=0, mask_pad_val=0,
                                  fit_output=False,
                                  interpolation=1, always_apply=False, p=0.6),
                    A.Resize(512, 512)],

                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=[], min_visibility=0.5),
                    keypoint_params=A.KeypointParams(format='xy'))

                transformed = transform(image=image_raw, keypoints=keypoints, bboxes=bboxes)
                transformed_image = transformed['image']
                transformed_keypoints = transformed['keypoints']
                transformed_bboxes = transformed['bboxes']
                if transformed_keypoints and transformed_bboxes:
                    if len(transformed_bboxes) == len(transformed_keypoints):
                        counter += 1
                        cv2.imwrite(dir_img + '/' + file_name + '_' + str(counter) + '.jpg',
                                    cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
                        keypoints_format=[]
                        for kps in transformed_keypoints:
                            keypoints_format.append([kps])
                        aDict = {'bboxes': transformed_bboxes, 'keypoints': keypoints_format}
                        jsonString = json.dumps(aDict)
                        jsonFile = open(dir_labels + '/' + file_name + '_' + str(counter) + '.json', "w")
                        jsonFile.write(jsonString)
                        jsonFile.close()
            counter = 0
            cv2.imwrite(dir_img + '/' + file_name + '_' + str(counter) + '.jpg',
                        cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR))
            keypoints_format = []
            for kps in keypoints:
                keypoints_format.append([kps])
            aDict = {'bboxes': bboxes, 'keypoints': keypoints_format}
            jsonString = json.dumps(aDict)
            jsonFile = open(dir_labels + '/' + file_name + '_' + str(counter) + '.json', "w")
            jsonFile.write(jsonString)
            jsonFile.close()

        except:
            print('ERROR -> ' + str(counter) + ' Error FIle -> '+ str(annotation))

    return dir_img, dir_labels

# def save_image_labelled(path):
#     for image_path in glob.glob(path + '/augmented/images/*.jpg'):
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image_path = Path(image_path)
#         annotation = str(image_path.parent.parent) + '/labels/' + str(image_path.stem) + '.json'
#         with open(annotation) as json_file:
#             data = json.load(json_file)
#         keypoints = []
#         for kds in data['keypoints']:
#             keypoints.append(kds[0])
#         bboxes = []
#         for bbox in data['bboxes']:
#             bboxes.append(bbox)
#
#         for i in range(len(keypoints)):
#             image = cv2.circle(image.copy(), (int(keypoints[i][0]), int(keypoints[i][1])), radius=0, color=[255, 0, 0],
#                                thickness=3)
#         for i in range(len(bboxes)):
#             image = cv2.rectangle(image.copy(), pt1=(int(bboxes[i][0]), int(bboxes[i][1])), pt2=(int(bboxes[i][2]),
#                                                                                           int(bboxes[i][3])),
#                                   color=(0, 0, 255), thickness=2)
#         dir = str(image_path.parent.parent) + '/augmented_labelled'
#         if not os.path.exists(dir):
#             os.mkdir(dir)
#         cv2.imwrite(dir + '/' + image_path.stem + '.jpg',
#                     cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


path = './dataset/train/images'
dir_img, dir_labels = augmentation(path, 4)
augmentation_geo(dir_img, dir_labels)
# save_image_labelled(path)
