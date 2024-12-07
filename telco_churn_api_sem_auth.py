import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from typing_extensions import Annotated, List, Dict, Union

app = FastAPI()

model = joblib.load('churn.pipeline.pkl')

@app.post('/predict')
def predict_instances(instances: List[Dict[str,Union[str,float,int]]]):
    instance_frame = pd.DataFrame(instances)

    predictions = model.predict_proba(instance_frame)

    results = {}
    for i, row in enumerate(predictions):
        prediction = model.classes_[np.argmax(row)]
        probability = np.amax(row)
        results[i] = {"prediction": prediction, "probability": probability}
    
    return results