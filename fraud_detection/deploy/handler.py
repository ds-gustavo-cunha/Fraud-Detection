# Library imports
import datetime
import uvicorn 
import pandas                        as     pd
from   fastapi                       import FastAPI
from   pydantic                      import BaseModel
from   typing                        import List
from   frauddetection.FraudDetection import FraudDetection


# instanciate a FastAPI object
app = FastAPI()

# create a schema that the request must follow
class Item(BaseModel):
  step:             int
  type:             str
  amount:           float
  nameOrig:         str
  oldbalanceOrg:    float
  newbalanceOrig:   float
  nameDest:         str
  oldbalanceDest:   float
  newbalanceDest:   float
  isFlaggedFraud:   int



@app.post('/fraud_detection') # path for prediction
def index( data: List[Item] ):  
  
  # check if request has at least 50 transaction (threshold = 50)
  if len(data) < 50:
    return f'Request must have at least 50 transactions, you sent {len(data)}.'

  else:
    # Instantiate FraudDetection class
    pipeline = FraudDetection()

    # create a dataframe with json data
    df_description = pipeline.create_dataframe( data )

    # send data to the data description transformations
    df_eng = pipeline.data_description( df_description )

    # send data to the feature engineering transformations
    df_filter = pipeline.feature_engineering( df_eng )

    # send data to the data filtering operations transformations
    df_eda = pipeline.data_filtering( df_filter )

    # send data to the exploratory data analysis transformations
    df_prep = pipeline.eda( df_eda )

    # send data to the data preparation transformations
    df_pred = pipeline.data_prep( df_prep )

    # send data to make prediction
    df_ml_result = pipeline.make_prediction( df_pred )

    # save request and prediction on database for future analyses
    #df_to_db = pipeline.save_request( df_ml_result )

    # select column for API response
    df_api_response = df_ml_result[['type', 'amount', 'name_orig', 
                                    'old_balance_orig', 'new_balance_orig', 
                                    'name_dest', 'old_balance_dest', 
                                    'new_balance_dest', 'prediction']]


    return df_api_response # return prediction



# Run the API created with uvicorn on http://127.0.0.1:8000 [local host]
if __name__ == '__main__':
  uvicorn.run(app, host='127.0.0.1', port=8000)
