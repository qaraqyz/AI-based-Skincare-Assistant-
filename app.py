from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, isfile, splitext
from scipy import misc
import imageio
import sys
import dlib
import csv

app = Flask(__name__)
model = load_model('/Users/dilnazahmetzan/DataspellProjects/capstone/mode.0.83.h5')
target_img = os.path.join(os.getcwd(), 'static/images')

class_names = ["acne", "blackhead", "carcinoma", "normal", "wrinkles"]
skin_types = ["oily", "dry", "normal", "problematic", "combination"]

@app.route('/')
def index_view():
    return render_template('index.html')


# Allow files with extension png, jpg, and jpeg
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXT
def crop(filename):
    originalFile_Dir = "static/images"
    croppedFaces_Dir = "static/extracted"
    Eye_Cascade_Path = 'models/haarcascade_eye.xml'
    PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'
    filepath = filename
    verb = False
    width_ratio = 1.5
    top_ratio = 1.5
    down_ratio = 4.5
    cheek_width_ratio = 2.8
    forehead_ratio = 0.3

    eye_cascade = cv2.CascadeClassifier(Eye_Cascade_Path)

    SCALE_FACTOR = 1
    FEATHER_AMOUNT = 11

    FACE_POINTS = list(range(17, 70))
    MOUTH_POINTS = list(range(48, 61))
    RIGHT_BROW_POINTS = list(range(17, 22))
    LEFT_BROW_POINTS = list(range(22, 27))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    NOSE_POINTS = list(range(27, 35))
    JAW_POINTS = list(range(0, 17))
    ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                                   RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
    OVERLAY_POINTS = [
        LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
        NOSE_POINTS + MOUTH_POINTS,
    ]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)


    class TooManyFaces(Exception):
        pass


    class NoFaces(Exception):
        pass


    def get_landmarks(im):
        rects = detector(im, 1)

        if len(rects) > 1:
            raise TooManyFaces
        if len(rects) == 0:
            raise NoFaces

        return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


    def annotate_landmarks(im, landmarks):
        im = im.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        return im


    def draw_convex_hull(im, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)


    def get_face_mask(im, landmarks):
        im = np.zeros(im.shape[:2], dtype=np.float64)

        for group in OVERLAY_POINTS:
            draw_convex_hull(im,
                             landmarks[group],
                             color=1)

        im = np.array([im, im, im]).transpose((1, 2, 0))

        im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
        im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

        return im


    def transformation_from_points(points1, points2):
        """
        Return an affine transformation [s * R | T] such that:
            sum ||s*R*p1,i + T - p2,i||^2
        is minimized.
        """
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R,
                                     c2.T - (s2 / s1) * R * c1.T)),
                          np.matrix([0., 0., 1.])])


    def read_im_and_landmarks(fname):
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                             im.shape[0] * SCALE_FACTOR))
        s = get_landmarks(im)

        return im, s
    def warp_im(im, M, dshape):
        output_im = np.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im,
                       M[:2],
                       (dshape[1], dshape[0]),
                       dst=output_im,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)
        return output_im
    COLOUR_CORRECT_BLUR_FRAC = 0.6
    def correct_colours(im1, im2, landmarks1):
        blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                                  np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                                  np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

        return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                    im2_blur.astype(np.float64))
    def infer_cheek_region(eye, width_ratio, down_ratio, left_or_right):
        region1 = [0] * 4
        if left_or_right == 'right':
            region1[0] = int(max(0, int(eye[0] - 0.5 * eye[2])))
            region1[2] = int(0.5 * eye[2])
        else:
            region1[0] = int(eye[0] + eye[2])
            region1[2] = int(0.5 * eye[2])
        region1[1] = int(eye[1] + eye[3])
        region1[3] = int(1.5 * eye[3])
        return region1
    def detect_face_direction(gray, face, eye, down_ratio, cheek_width_ratio):
        region1 = [0] * 4
        region2 = [0] * 4
        region1 = infer_cheek_region(eye[0], cheek_width_ratio, down_ratio, 'left')
        region2 = infer_cheek_region(eye[0], cheek_width_ratio, down_ratio, 'right')
        std1 = np.std(gray[region1[1]:(region1[1]+region1[3]), region1[0]:(region1[0]+region1[2])])
        std2 = np.std(gray[region2[1]:(region2[1]+region2[3]), region2[0]:(region2[0]+region2[2])])
        face_direction = ""
        if std1 > std2:
            face_direction = "right"
        else:
            face_direction = "left"
        return face_direction

    def extract_cheek_region(face_x_min, face_x_max, face_y_max, eye_landmarks, left_or_right):
        if left_or_right == "Left":
            cheek_region_min_x = eye_landmarks[0,0]
            cheek_region_max_x = int(face_x_max - 0.05 * (face_x_max - min(eye_landmarks[:,0])))
        else:
            cheek_region_max_x = eye_landmarks[-1, 0] # right cheek, x direction, cheek starts at the face_x_min + a margin,
                                                      # where the margin is 0.1 * (cheek_region_max_x - face_x_min). This margin is set
                                                      # to avoid hair or other backgrounds, and ends at the most inner point of right eye
            cheek_region_min_x = int(face_x_min + 0.1 * (cheek_region_max_x - face_x_min))

        cheek_region_min_y = int(max(eye_landmarks[:,1]) + 0.2 * (max(eye_landmarks[:,1])  - min(eye_landmarks[:,1])))
        cheek_region_max_y = int(face_y_max - 0.1 * (face_y_max - max(eye_landmarks[:,1])))
        return [cheek_region_min_x, cheek_region_min_y, cheek_region_max_x, cheek_region_max_y]


    imageFiles = [join(originalFile_Dir, f) for f in listdir(originalFile_Dir) if isfile(join(originalFile_Dir, f))]
    num_images = len(imageFiles)
    image_counter = 0
    start_index = 0
    end_index = num_images
    for imagefile in imageFiles[start_index:end_index]:
        imagefile = filepath
        image_counter += 1
        imageName = splitext(os.path.basename(imagefile))[
            0]
        face_detected = False
        try:
            img, landmarks = read_im_and_landmarks(imagefile)
            face_detected = True
        except:
            img = cv2.imread(imagefile)  # if landmark model does not work, read the image
            face_detected = False
        img_height, img_width = img.shape[0:2]  # get the image height and width. Image data is in the format of [height, width, channel]
        min_dim = min(img_height, img_width)
        min_face_size = min_dim * 0.2  # Specify the minimal face size. Heuristic.
        min_eye = min_face_size * 0.2  # specify the minimal eye size.
        min_eye_area = min_eye ** 2  # specify the miniaml area of the eye. This is used screen detected eyes by the OneEye model.
        # Keep in mind, the One Eye model will identify whatever looks like an eye. We need to screen out
        # those that are too small.

        gray = cv2.cvtColor(img,
                            cv2.COLOR_BGR2GRAY)  # Convert image from BGR to GRAY. OpenCV reads images in as BGR. One Eye model
        # works on gray scale images.
        if face_detected: # if face is detected by landmark model
            mask = get_face_mask(img, landmarks)
            face_x_min = int(max(0, np.asarray(min(landmarks[:,0])).flatten()[0])) #Get the minimal value of the detected landmarks in x
            face_x_max = int(min(img_width, np.asarray(max(landmarks[:,0])).flatten()[0])) # Get the maximal value of the detected landmarks in x
            face_y_min = int(max(0, np.asarray(min(landmarks[:,1])).flatten()[0])) # Get the minimal value of the detected landmarks in y
            face_y_max = int(min(img_height, np.asarray(max(landmarks[:,1])).flatten()[0])) # Get the maximal value of the detected landmarks in y
            face_height = face_y_max - face_y_min # Get the height of face
            forehead_height = int(face_height * forehead_ratio) # Ideally, forehead height should be 1/2 of the height between eyebrow and bottom of chin
                                                                # We choose forehead_ratio = 0.3 to avoid hairs on the forehead.
            new_face_y_min = max(0, face_y_min - forehead_height) # new_face_y_min is the top edge of the forehead.
            right_brow_landmarks = landmarks[RIGHT_BROW_POINTS,:]
            left_brow_landmarks = landmarks[LEFT_BROW_POINTS,:]
            right_eye_landmarks = landmarks[RIGHT_EYE_POINTS,:]
            left_eye_landmarks = landmarks[LEFT_EYE_POINTS,:]
            mouse_landmarks = landmarks[MOUTH_POINTS,:]
            ########################
            # Get the forehead patch
            ########################
            [right_brow_min_x, left_brow_max_x] = \
                [max(0, np.min(np.array(right_brow_landmarks[:,0]))), min(img_width, np.max(np.array(left_brow_landmarks[:,0])))]
            brow_min_y = min(np.min(np.array(right_brow_landmarks[:,1])),np.min(np.array(left_brow_landmarks[:,1])))
            forehead_x_min = right_brow_min_x # forehead starts at the left landmark of the right eye brow
            forehead_x_max = left_brow_max_x
            forehead_y_min = max(0, brow_min_y - forehead_height)
            forehead_y_max = min(brow_min_y, forehead_y_min + forehead_height)
            forehead_y_min = int(forehead_y_min)
            forehead_y_max = int(forehead_y_max)
            forehead_x_min = int(forehead_x_min)
            forehead_x_max = int(forehead_x_max)
            forehead_region = img[forehead_y_min:forehead_y_max, forehead_x_min:forehead_x_max, :]
            forehead_file_name = join(croppedFaces_Dir, imageName+"_fh.jpg")
            # BGR image needs to be converted to RGB before saving as image file
            forehead_region = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2RGB)
            imageio.imsave(forehead_file_name, forehead_region)
            chin_x_min = np.max(
                np.array(right_eye_landmarks[:, 0]))  # In x direction, chin patch will be between the two most inner
            # points of eyebrows
            chin_x_max = np.min(np.array(left_eye_landmarks[:, 0]))
            chin_y_min = np.max(np.array(mouse_landmarks[:, 1]))  # In y direction, chin patch starts at the lowest point of mouse landmarks
            chin_y_max = face_y_max  # In y direction, chin patch ends at the lowest point of face
            chin_region = img[chin_y_min:chin_y_max, chin_x_min:chin_x_max, :]
            chin_file_name = join(croppedFaces_Dir, imageName + "_chin.jpg")
            chin_region = cv2.cvtColor(chin_region, cv2.COLOR_BGR2RGB)
            imageio.imsave(chin_file_name, chin_region)
            #########################
            # Get nose patch
            #########################
            nose_landmarks = landmarks[NOSE_POINTS, :]
            nose_x_min = np.min(np.array(nose_landmarks[:, 0]))
            nose_x_max = np.max(np.array(nose_landmarks[:, 0]))
            nose_y_min = np.min(np.array(nose_landmarks[:, 1]))
            nose_y_max = np.max(np.array(nose_landmarks[:, 1]))
            nose_region = img[nose_y_min:nose_y_max, nose_x_min:nose_x_max, :]
            nose_file_name = join(croppedFaces_Dir, imageName + "_nose.jpg")
            nose_region = cv2.cvtColor(nose_region, cv2.COLOR_BGR2RGB)
            imageio.imsave(nose_file_name, nose_region)
            ##########################
            # Get the cheeks patch
            ##########################
            # Decide whether it is a side view or not. If the the left eye is 15% wider than the right eye, we determine
            # that it is the left cheek facing the camera.
            # If the right eye is 15% wider than the left eye, we determine that it is the right cheek facing the camera.
            # Under such situation, we only extract one cheek patch.
            left_eye_width = np.max(np.array(left_eye_landmarks[:, 0])) - np.min(np.array(left_eye_landmarks[:, 0]))
            right_eye_width = np.max(np.array(right_eye_landmarks[:, 0])) - np.min(np.array(right_eye_landmarks[:, 0]))
            right_face = True
            left_face = True
            if float(right_eye_width) / float(
                    left_eye_width) >= 1.15:  # right eye is bigger than left eye, showing the right face
                left_face = False
            elif float(left_eye_width) / float(
                    right_eye_width) >= 1.15:  # left eye is bigger than right eye, showing the left face
                right_face = False
            if right_face:
                right_cheek_region = extract_cheek_region(face_x_min, face_x_max, face_y_max, right_eye_landmarks, "Right")
                cheek_region = img[right_cheek_region[1]:right_cheek_region[3], right_cheek_region[0]:right_cheek_region[2],
                               :]
                cheek_file_name = join(croppedFaces_Dir, imageName + "_rc.jpg")
                cheek_region = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2RGB)
                imageio.imsave(cheek_file_name, cheek_region)
            if left_face:
                left_cheek_region = extract_cheek_region(face_x_min, face_x_max, face_y_max, left_eye_landmarks, "Left")
                cheek_region = img[left_cheek_region[1]:left_cheek_region[3], left_cheek_region[0]:left_cheek_region[2], :]
                cheek_file_name = join(croppedFaces_Dir, imageName + "_lc.jpg")
                cheek_region = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2RGB)
                imageio.imsave(cheek_file_name, cheek_region)
            # if verb == True, display the detected skin patches on the original image, using rectangle to highlight the skin patches
            if verb:
                img_tmp = img
                img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
                cv2.rectangle(img_tmp, (forehead_x_min, forehead_y_min), (forehead_x_max, forehead_y_max), (0, 255, 0), 10)
                if right_face:
                    cv2.rectangle(img_tmp, (right_cheek_region[0], right_cheek_region[1]), \
                                  (right_cheek_region[2], right_cheek_region[3]), (255, 255, 0), 10)
                if left_face:
                    cv2.rectangle(img_tmp, (left_cheek_region[0], left_cheek_region[1]), \
                                  (left_cheek_region[2], left_cheek_region[3]), (255, 255, 0), 10)
                plt.imshow(img_tmp)

        if not face_detected:
            print("Face not detected by landmarks model...")
            # Use the OneEye model to detect one eye, and infer the face region based on the eye location
            eye_detected = False
            roi_gray = gray
            roi_color = img
            roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            max_area = 0
            eye_count = 0
            max_index = 0
            for (ex,ey,ew,eh) in eyes: # there might be multiple eyes detected. Choose the biggest one
                if ew*eh >= max_area and ex >= img_width * 0.1 and ex <= img_width * 0.9:
                    max_area = ew*eh
                    max_index = eye_count
                eye_count += 1
            if max_area >= min_eye_area: # if area of maximal eye is greater than the eye area threshold, take it as a real eye
                eye_detected = True
                (ex, ey, ew, eh) = eyes[max_index]
                if float(ew) / float(img_width) > 0.15 or float(eh) / float(img_height) > 0.15: # detected eye too large
                    # resize the detected eye
                    center_x = ex + ew/2
                    center_y = ey + eh/2
                    resized_w = min(img_width * 0.15, img_height * 0.15) # resize the eye
                    ex = int(center_x - resized_w/2)
                    ey = int(center_y - resized_w/2)
                    ew = int(resized_w)
                    eh = int(resized_w)
                    eyes1 = np.array([ex, ey, resized_w, resized_w]).reshape((1,4))
                else:
                    eyes1 = np.array(eyes[max_index]).reshape((1,4))
                face1 = np.array(())
                face_direction = detect_face_direction(gray, face1, eyes1, down_ratio, cheek_width_ratio)
                if face_direction == "left":
                    print("Left eye detected")
                    face_min_x = eyes1[0, 0]
                    face_max_x = min(img_width, int(eyes1[0,0] + (cheek_width_ratio + 0.5) * eyes1[0, 2]))
                    forehead_max_x = min(img_width, int(eyes1[0,0] + width_ratio * eyes1[0, 2]))
                    forehead_min_x = face_min_x
                    cheek_min_x = int(eyes1[0, 0] + 0.5 * eyes1[0,2])
                    cheek_max_x = face_max_x
                else:
                    print("Right eye detected")
                    face_min_x = max(0, int(eyes1[0, 0] - cheek_width_ratio * eyes1[0, 2]))
                    face_max_x = eyes1[0, 0] + eyes1[0, 2]
                    forehead_min_x = max(0, int(eyes1[0, 0] - width_ratio * eyes1[0, 2]))
                    forehead_max_x = min(img_width, int(eyes1[0, 0] + width_ratio * eyes1[0, 2]))
                    cheek_max_x = int(eyes1[0,0] + 0.5*eyes1[0,2])
                    cheek_min_x = face_min_x
                forehead_min_y = max(0, int(eyes1[0, 1] - top_ratio * eyes1[0,3]))
                forehead_max_y = max(0, int(eyes1[0, 1] - 0.5 * eyes1[0, 3]))
                forehead_ok = False
                if forehead_max_y - forehead_min_y >= 0.7 * eyes1[0, 3]:
                    forehead_ok = True
                    forehead_region = img[forehead_min_y:forehead_max_y, forehead_min_x: forehead_max_x, :]
                    forehead_region = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2RGB)
                    forehead_file_name = join(croppedFaces_Dir, imageName + "_fh.jpg")
                    misc.imsave(forehead_file_name, forehead_region)
                # Get the cheek region
                cheek_min_y = int(eyes1[0, 1] + eyes1[0, 3])
                cheek_max_y = min(img_height, int(eyes1[0, 1] + down_ratio * eyes1[0, 3]))
                cheek_region = img[cheek_min_y: cheek_max_y, cheek_min_x: cheek_max_x, :]
                cheek_region = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2RGB)
                if face_direction == "left":
                    cheek_file_name = join(croppedFaces_Dir, imageName + "_lc.jpg")
                elif face_direction == "right":
                    cheek_file_name = join(croppedFaces_Dir, imageName + "_rc.jpg")
                else:
                    cheek_file_name = join(croppedFaces_Dir, imageName + "_c.jpg")
                misc.imsave(cheek_file_name, cheek_region)
                if verb:
                    image = img
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if forehead_ok:
                        cv2.rectangle(image, (forehead_min_x, forehead_min_y), \
                                      (forehead_max_x, forehead_max_y), (0, 255, 0), 5)
                    cv2.rectangle(image, (cheek_min_x, cheek_min_y), \
                                  (cheek_max_x, cheek_max_y), (255, 255, 0), 5)
                    cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

                    plt.imshow(image)
        if (not face_detected) and (not eye_detected): # no face detected, nor eye detected, save the entire image and write to dest
            print("No cheeks or forehead detected, output the original file %s.jpg"%imageName)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if verb:
                plt.imshow(img)
            outfile = join(croppedFaces_Dir, imageName+".jpg")
            misc.imsave(outfile, img)

        if image_counter % 500 == 0: # Report the progress on image processing every 500 images processed
            print("%d images have been processed."%image_counter)
    def roll_and_plot(img, file_name_wo_ext, x_or_y, pixels, verb):
        img_height, img_width = img.shape[0:2]
        img2 = copy.copy(img)
        if x_or_y == 'x':
            img2[:, 0:(img_width-pixels),:] = img[:,pixels:img_width,:]
            img2[:,(img_width-pixels):img_width,:] = img[:,0:pixels,:]
        else:
            img2[0:(img_height-pixels), :, :] = img[pixels:img_height, :, :]
            img2[(img_height-pixels):img_height, :,:] = img[0:pixels,:, :]
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        if verb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            f, axarr = plt.subplots(2,1)
            axarr[0].imshow(img)
            axarr[1].imshow(img2)
        return
# Function to load and prepare the image in the right shape
def read_image(filename):
    crop(filename)
    parts = []  # List to store the divided image parts
    predictions = []  # List to store the predictions
    dirname, base = os.path.split(filename)
    basename = os.path.splitext(base)[0]
    extension = 'jpg'

    part_filename = [
        f"static/extracted/{basename}_lc.{extension}",
        f"static/extracted/{basename}_rc.{extension}",
        f"static/extracted/{basename}_chin.{extension}",
        f"static/extracted/{basename}_nose.{extension}",
        f"static/extracted/{basename}_fh.{extension}",
    ]

    for i in range(5):
       # Load the part, preprocess it, and make the prediction
       part_img = tf.keras.utils.load_img(part_filename[i], target_size=(224, 224))
       input_arr = tf.keras.utils.img_to_array(part_img)
       input_arr = np.array([input_arr])  # Convert single image to a batch
       input_arr /= 225
       prediction = model.predict(input_arr)
       classes = np.argmax(prediction, axis=1)
       predictions.append(class_names[classes[0]])

    predictions_count = {class_name: predictions.count(class_name) for class_name in class_names}
    skin_type = determine_skin_type(predictions_count)

    return predictions_count, skin_type


def determine_skin_type(predictions_count):
    relevant_counts_sum = sum(
        predictions_count[class_name] for class_name in ["acne", "blackhead", "normal", "carcinoma", "wrinkles"]
    )

    acne_percentage = predictions_count["acne"] / relevant_counts_sum
    blackhead_percentage = predictions_count["blackhead"] / relevant_counts_sum
    normal_percentage = predictions_count["normal"] / relevant_counts_sum
    carcinoma_percentage = predictions_count["carcinoma"] / relevant_counts_sum
    wrinkles_percentage = predictions_count["wrinkles"] / relevant_counts_sum

    if acne_percentage > 0.4:
        return "sensitive"
    elif wrinkles_percentage > 0.2:
        return "dry"
    elif blackhead_percentage > 0.3:
        return "oily"
    elif normal_percentage > 0.2:
        return "normal"
    else:
        return "combination"

def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            data.append(row)
    return data

def recommend_skincare(products, category, price_max, price_min, skin_type, rank):
    recommendations = []
    for product in products:
        label, brand, name, price, product_rank, ingredients, combination, dry, normal, oily, sensitive = product

        # Filter based on category, price range, skin type, and rank
        if label.lower() == category.lower() and \
                price_min <= float(price) <= price_max and \
                (skin_type == 'combination' or combination == '1') and \
                (skin_type == 'dry' or dry == '1') and \
                (skin_type == 'normal' or normal == '1') and \
                (skin_type == 'oily' or oily == '1') and \
                (skin_type == 'sensitive' or sensitive == '1') and \
                float(product_rank) >= rank:
            recommendations.append((brand, name))

    return recommendations
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            predictions_count, skin_type = read_image(file_path)
            return render_template('predict.html', predictions_count=predictions_count, skin_type=skin_type, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"



@app.route('/skincare_assistant', methods=['GET', 'POST'])
def skincare_assistant():
    if request.method == 'POST':
        # Read CSV file
        filename = 'cosmetics.csv'
        data = read_csv(filename)

        # Get user preferences from the form
        price_max = float(request.form.get('price_max'))
        price_min = float(request.form.get('price_min'))
        skin_type = request.form.get('skin_types')
        rank = float(request.form.get('rank'))

        # Recommend skincare routine
        moisturizer = recommend_skincare(data, 'Moisturizer', price_max, price_min, skin_type, rank)
        cleanser = recommend_skincare(data, 'Cleanser', price_max, price_min, skin_type, rank)
        treatment = recommend_skincare(data, 'Treatment', price_max, price_min, skin_type, rank)
        face_mask = recommend_skincare(data, 'Face Mask', price_max, price_min, skin_type, rank)
        eye_cream = recommend_skincare(data, 'Eye cream', price_max, price_min, skin_type, rank)
        spf = recommend_skincare(data, 'Sun protect', price_max, price_min, skin_type, rank)
        # Render the template with recommendations and user input
        return render_template('recommend.html',
                               moisturizer=moisturizer,
                               cleanser=cleanser,
                               treatment=treatment,
                               face_mask=face_mask,
                               eye_cream=eye_cream,
                               spf=spf
                              )

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)
