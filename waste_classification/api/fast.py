from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2
import io
from PIL import Image
from waste_classification.interface.main import load_model, load_glass_model

#from face_rec.face_detection import annotate_face

app = FastAPI()
app.state.model = load_model()
app.state.model_glass = load_glass_model()

# Allow all requests (optional, good for development purposes)
app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],  # Allows all origins
     allow_credentials=True,
     allow_methods=["*"],  # Allows all methods
     allow_headers=["*"])  # Allows all headers


@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/predict')
async def receive_image(img: UploadFile=File(...)):
    ### Step 1: Read the Image File
    contents = await img.read()

    ### Step 2: Convert Bytes to NumPy Array (Fixing `fromstring` issue)
    nparr = np.frombuffer(contents, np.uint8)

    ### Step 3: Decode the Image into OpenCV Format (BGR)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # numpy.ndarray (H, W, C)

    ### Step 4: Convert OpenCV Image to PIL Image (Fix Resize Issue)
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

    ### Step 5: Resize the Image
    pil_img = pil_img.resize((384, 384))

    ### Step 6: Convert PIL Image to NumPy Array (TensorFlow Format)
    img_array = img_to_array(pil_img)

    ### Step 7: Reshape to Model Input Format
    x = np.expand_dims(img_array, axis=0)  # (1, 384, 384, 3)

    ### Step 8: Preprocess the Image for Model Compatibility
    x = preprocess_input(x)

    ##Run prediction

    y_pred = app.state.model.predict(x)

    classes = ['battery',
            'biological',
        'cardboard',
        'clothes',
        'glass',
        'metal',
        'paper',
        'plastic',
        'shoes',
        'trash']

    predicted_class = classes[(np.argmax(y_pred[0]))]
    if predicted_class == 'glass':
            y_pred_2 = app.state.model_glass.predict(x)
            print(y_pred_2)
            print(type(y_pred_2))

            return {"glass_brown":float(y_pred_2[0][0]),
                    "glass_green":float(y_pred_2[0][1]),
                    "glass_transparent":float(y_pred_2[0][2]),
                    }

    else:

        print(y_pred)
        print(type(y_pred))

        return {"battery":float(y_pred[0][0]),
                "biological":float(y_pred[0][1]),
                "cardboard":float(y_pred[0][2]),
                "clothes":float(y_pred[0][3]),
                "glass":float(y_pred[0][4]),
                "metal":float(y_pred[0][5]),
                "paper":float(y_pred[0][6]),
                "plastic":float(y_pred[0][7]),
                "shoes":float(y_pred[0][8]),
                "trash":float(y_pred[0][9]),
        }

@app.post('/glass')
async def receive_image(img: UploadFile=File(...)):
    ### Step 1: Read the Image File
    contents = await img.read()

    ### Step 2: Convert Bytes to NumPy Array (Fixing `fromstring` issue)
    nparr = np.frombuffer(contents, np.uint8)

    ### Step 3: Decode the Image into OpenCV Format (BGR)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # numpy.ndarray (H, W, C)

    ### Step 4: Convert OpenCV Image to PIL Image (Fix Resize Issue)
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

    ### Step 5: Resize the Image
    pil_img = pil_img.resize((384, 384))

    ### Step 6: Convert PIL Image to NumPy Array (TensorFlow Format)
    img_array = img_to_array(pil_img)

    ### Step 7: Reshape to Model Input Format
    x = np.expand_dims(img_array, axis=0)  # (1, 384, 384, 3)

    ### Step 8: Preprocess the Image for Model Compatibility
    x = preprocess_input(x)

    ##Run prediction

    y_pred = app.state.model_glass.predict(x)
    print(y_pred)
    print(type(y_pred))

    return {"brown":float(y_pred[0][0]),
            "green":float(y_pred[0][1]),
            "transparent":float(y_pred[0][2]),
    }
