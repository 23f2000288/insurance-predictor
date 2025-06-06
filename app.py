from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pickle
import pandas as pd

# import ml model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

tier_1_cities = ["Mumbai", "Delhi", "Banglore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = ["Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Vishakhapatnam", "Sasaram", "Dehri",
                 "Bhopal", "Nagpur", "Vadodra", "Surat", "Rajkot", "Jodhpur", "Varanasi", "Raipur", "Amritsar", "Agra",
                 "Dehradun", "Mysore", "jabalpur", "Guwahati", "Thiruvanathapuram", "Ludhiana", "Nasik", "Allahabad", "Udaipur",
                 "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijaywada", "Tiruchirappalli", "Bhavnagar","Gawalior","Dhanbad","Barailey",
                 "Aligarh", "Gaya","Kozikhodo", "Warangal", "Kolhapur", "Belaspur", "Jalandhar", "Guntur", "Asansol", "Siliguri"]

# pydantic model to validate incoming data
class UserInput(BaseModel):
    age: Annotated[int, Field(...,gt=0,lt=120, description='age of the user')]
    weight:Annotated[float, Field(...,gt=0,lt=400, description='weight of the user')]
    height:Annotated[float, Field(...,gt=0,lt=2.5, description='height of the user')]
    income_lpa:Annotated[float, Field(...,gt=0,lt=120, description='annual salary of the user')]
    smoker:Annotated[bool, Field(..., description='Is user a smoker o')]
    city:Annotated[str, Field(...,description='city of the user')]
    occupation:Annotated[Literal['retired', 'freelancer', 'student', 'government_job',
       'business_owner', 'unemployed', 'private_job'], Field(..., description='occupation of the user')]
    
    @computed_field
    @property
    def bmi(self) -> float:
        return self.weight/(self.height**2)
    
    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 20:
            return "high"
        elif self.smoker and self.bmi > 27:
            return "medium"
        else:
            return "low"
    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 65:
            return "middle_aged"
        else:
            return "senior"
    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
             return 2
        else:
             return 3

# defination of apis
@app.post('/predict')
def predict_premium(data: UserInput):
    input_df=pd.DataFrame([{
        'bmi' : data.bmi,
        'age_group' : data.age_group,
        'lifestyle_risk' : data.lifestyle_risk,
        'city_tier' : data.city_tier,
        'income_lpa' : data.income_lpa,
        'occupation' : data.occupation

    }])

    prediction=model.predict(input_df)[0]

    return JSONResponse(status_code=200,content={'predicted_category': prediction})
            