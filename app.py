from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import uvicorn


origins = [
    "https://smart-farming-v2.vercel.app",
    "http://localhost",
    "http://localhost:3000",
]



# Define the data model
class Item(BaseModel):
    temperature: float
    humidity: float
    soil_moisture: float

# Load the model
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create the FastAPI application
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/crop')
def recommend(item: Item):
    # Print the data received
    print(item)
    # Make a prediction using the model
    prediction = model.predict([[item.temperature, item.humidity, item.soil_moisture]])
    
    label_mapping = {
    0:'apple',1: 'banana',2: 'blackgram',3: 'chickpea',4: 'coconut',5: 'coffee',6: 'cotton', 7: 'grapes',8: 'jute',9: 'kidneybeans',10:'lentil',11:'maize',
    12:'mango',13:'mothbeans',14:'mungbeans',15:'muskmelon',16:'orange',17:'papaya',18:'pigeonpeas',19:'pomegranate',20:'rice',21:'watermelon'
}
    predicted_class = label_mapping[prediction[0]]
    print("predicted_class:", predicted_class)
    return {'crop': predicted_class}

# Run the application using uvicorn
# This should be in a separate file or under a __name__ == "__main__" condition
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
