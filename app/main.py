import base64

import cv2
import numpy as np
from fastapi import FastAPI, Request
from app.hog import gethog

app = FastAPI()

def readb64(uri):
    encoded_data = uri.split(",")[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img

@app.get("/api/gethog")
async def read_str(request: Request):
    try:
        data = await request.json()
        img = readb64(data.get("item_str"))
        hog = gethog(img)
        return {"HOG": hog.tolist()}
    except Exception as e:
        return {"error": str(e)}