import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import numpy as np
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
from PIL import Image

def face(IMAGE_FILES):

 """

 Returns cropped image with face given 
 path to an image. If no faces are found
 returns None

 @params:
 IMAGE_FILES(str):path

 Returns:
 None if no heads found/cropped image


 """


 with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:
    
    #print(IMAGE_FILES)
    pil_image = Image.open(IMAGE_FILES).convert('RGB') 
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    image = open_cv_image[:, :, ::-1].copy() 
    image_rows, image_cols, _ = image.shape
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    im_pil=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    if not results.detections:
      return None

    else:
      detection=results.detections[0]
      location = detection.location_data

      relative_bounding_box = location.relative_bounding_box
      rect_start_point = _normalized_to_pixel_coordinates(
      relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
      image_rows)
      rect_end_point = _normalized_to_pixel_coordinates(
      relative_bounding_box.xmin + relative_bounding_box.width,
      relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
      image_rows)
      xleft,ytop=rect_start_point
      xright,ybot=rect_end_point

      crop_img = im_pil[ytop: ybot, xleft: xright]

      return crop_img

    
    
  
   