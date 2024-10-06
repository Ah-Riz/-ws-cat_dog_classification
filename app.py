from fastapi import FastAPI, File, UploadFile
from architecture import MobileNetV2
import torch
from data_prep import get_data_transform
from PIL import Image
import io
from fastapi.responses import JSONResponse
from utils import convert_params
import yaml
from dotenv import load_dotenv
import os
from datetime import datetime
from google.cloud import firestore
import uuid
import base64

app = FastAPI()

load_dotenv()
db = firestore.AsyncClient(project=os.getenv("FIRESTORE_DB_PROJECT"), database=os.getenv("FIRESTORE_DB"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
        
choosen_arch = config["choose_arch"]
method = [x for x in config["arch"] if x["name"] == choosen_arch][0]
model_config = convert_params(method["params"])

model = MobileNetV2(**model_config)
model.load_state_dict(torch.load("asset/best_model.pt", weights_only=True, map_location=device))
model = model.to(device)
model.eval()

classes=["cat", "dog"]

def prep_image(image: Image.Image):
    transform = get_data_transform()
    image = transform(image)
    image = image.unsqueeze(0)
    return image.to(device)

def img_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    return f"{base64.b64encode(img_byte).decode()}"
    
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())) 
        image_tensor = prep_image(image)
        time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with torch.no_grad():
            output = model(image_tensor)
            test_pred_label = output.argmax(dim=1)
            predicted_class = classes[test_pred_label.item()]
            probs = torch.nn.functional.softmax(output, dim=1)
            conf,_ = torch.max(probs, 1)
            conf = conf.item()
            if conf>float(os.getenv("CONFIDENCE_THRESHOLD")):
                is_uncertainty = 0
            else:
                is_uncertainty = 1
        
        doc_ref = db.collection(os.getenv("FIRESTORE_DB_COLLECTION")).document(time_stamp)
        result = {
            'id': str(uuid.uuid4()),
            "time_stamp":time_stamp,
            "image64":img_to_base64(image),
            "prediction":predicted_class,
            "confidence":conf,
            "is_uncertainty":is_uncertainty
            }
        await doc_ref.set(result)
        
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error":str(e)}, status_code=400)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)