import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

image_file_path = os.getcwd() + "/fdz3/girls-kimono-input.jpg"
img             = cv2.imread(image_file_path)

app   = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
faces = app.get(np.asarray(img))
rimg  = app.draw_on(img, faces)
cv2.imwrite("./img/girls-kimono-output.jpg", rimg)
