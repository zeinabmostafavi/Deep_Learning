import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from retinaface import RetinaFace
import cv2


for i in range(1, 11):
    faces = RetinaFace.extract_faces(img_path=f"{i}.jpg", align=True)
    for face in faces:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{i}.jpg", face)
