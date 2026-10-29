from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from predict import predict_data


app = FastAPI(title="Wine Classifier API")

CLASS_NAMES = ["cultivar_0", "cultivar_1", "cultivar_2"]

class WineData(BaseModel):
    """
    Pydantic BaseModel representing wine chemical measurements.

    Attributes:
        The 13 measurements of the wine, in the same order as the training data.
        Each must be greater than 0 (gt=0), and the example values fill in the /docs page.
    """
    alcohol: float = Field(gt=0, examples=[13.2])
    malic_acid: float = Field(gt=0, examples=[1.78])
    ash: float = Field(gt=0, examples=[2.14])
    alcalinity_of_ash: float = Field(gt=0, examples=[11.2])
    magnesium: float = Field(gt=0, examples=[100])
    total_phenols: float = Field(gt=0, examples=[2.65])
    flavanoids: float = Field(gt=0, examples=[2.76])
    nonflavanoid_phenols: float = Field(gt=0, examples=[0.26])
    proanthocyanins: float = Field(gt=0, examples=[1.28])
    color_intensity: float = Field(gt=0, examples=[4.38])
    hue: float = Field(gt=0, examples=[1.05])
    od280_od315_of_diluted_wines: float = Field(gt=0, examples=[3.4])
    proline: float = Field(gt=0, examples=[1050])

class WineResponse(BaseModel):
    response:int
    class_name:str
    confidence:float



"""Modern web apps use a technique named routing. This helps the user remember the URLs. 
For instance, instead of having /booking.php they see /booking/. Instead of /account.asp?id=1234/ 
they’d see /account/1234/."""

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """Concurrent (multiple tasks can run simultaneously)"""
    return {"status": "healthy"}

@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    """
    Predict the wine cultivar based on provided features.
    This endpoint accepts wine chemical measurements and returns the predicted cultivar class.
    Args:
        wine_features (WineData): A WineData object containing:
            - the 13 measurements (alcohol, malic_acid, ..., proline), all greater than 0
    Returns:
        WineResponse: A response object containing:
            - response (int): The predicted cultivar class (0, 1, or 2)
            - class_name (str): The name of the predicted cultivar
            - confidence (float): The model's probability for that class
    Raises:
        HTTPException: Returns a 500 status code with error details if prediction fails.
    Example:
        POST /predict
        {
            "alcohol": 13.2,
            "malic_acid": 1.78,
            ...
            "proline": 1050
        }
        Response:
        {
            "response": 0,
            "class_name": "cultivar_0",
            "confidence": 0.986
        }
    """
    try:
        features = [list(wine_features.model_dump().values())]

        prediction, confidence = predict_data(features)
        return WineResponse(response=int(prediction[0]), class_name=CLASS_NAMES[int(prediction[0])],
                            confidence=round(float(confidence[0]), 3))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    
