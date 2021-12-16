import inflection
import pickle
import json
import datetime
import os
import pandas                      as pd
from   pymongo                     import MongoClient
from   pyspark.sql                 import SparkSession
from   pyspark.sql.functions       import col, when, abs
from   pyspark.sql.types           import IntegerType, DecimalType
from   pyspark.ml.classification   import GBTClassifier, GBTClassificationModel
from   pyspark.ml.feature          import VectorAssembler, RobustScaler, RobustScalerModel
from   pyspark.sql                 import functions                                            as F


# create a pipeline class
class FraudDetection():

    def __init__(self): # class constructor
        # create Spark session
        self.ss = SparkSession.builder.appName("fraud_detection").getOrCreate()
        
        # create path for feature transformers and ML model
        self.features_path = "features/"
        self.model_path = "models/ml_model.spark"


    def create_dataframe(self, data):
        """Create a spark dataframe from json data sent on request"""
        
        ####### NOT THE MOST EFFICIENT, BUT AT LEAST WORKS #######
        ## TO DO: search for a better solution to create this dataframe ...
        # create a pandas dataframe from json request
        df_request = pd.DataFrame(columns = ['step', 'type', 'amount', 
                                             'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
                                             'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud'] ) 
        
        # populate pandas dataframe with transactions
        for row in data: 
            df_request = df_request.append( dict(row) , ignore_index= True)

        # convert pandas dataframe to spark dataframe
        df_request = self.ss.createDataFrame( df_request )


        return df_request



    def data_description( self, df_description ):
        """df_description is the dataframe to be processed accordint to the DATA DESCRIPTION section"""

        # change from camel case to snake case
        snake_case = [ inflection.underscore( column ) for column in df_description.columns ]
            
        # assign snake case to dataframe column names
        df_description = df_description.toDF( *snake_case )

        # rename columns to avoid typos
        df_description = df_description.withColumnRenamed("oldbalance_org" , "old_balance_orig")  \
                                       .withColumnRenamed("newbalance_orig" , "new_balance_orig") \
                                       .withColumnRenamed("oldbalance_dest" , "old_balance_dest") \
                                       .withColumnRenamed("newbalance_dest" , "new_balance_dest")

########################################NAME DEST ->  .withColumnRenamed("namedest" , "name_dest")
        print(df_description.columns)
#########################################

        return df_description # data description done



    def feature_engineering( self, df_eng ):
        """df_eng is the dataframe to be processed according to the FEATURE ENGINEERING section"""

        # delta_balance_orig will be the variation in origin balance,
        # that is, new_balance_orig - old_balance_orig
        df_eng = df_eng.withColumn('delta_balance_orig', col('new_balance_orig') - col('old_balance_orig') )

        # delta_balance_dest will be the variation in destination balance,
        # that is, new_balance_dest - old_balance_dest
        df_eng = df_eng.withColumn('delta_balance_dest', col('new_balance_dest') - col('old_balance_dest') )

        # orig_rate will be the rate between new_balance_orig and old_balance_orig
        # that is, (new_balance_orig + 1) / (old_balance_orig + 1).
        # We added 1 to avoid division by zero
        df_eng = df_eng.withColumn('orig_rate', (col('new_balance_orig') + 1) / (col('old_balance_orig') + 1) )

        # dest_rate will be the rate between new_balance_dest and old_balance_dest
        # that is, (new_balance_dest + 1) / (old_balance_dest + 1).
        # We added 1 to avoid division by zero
        df_eng = df_eng.withColumn('dest_rate', (col('new_balance_dest') + 1) / (col('old_balance_dest') + 1) )


        return df_eng # feature engineering done



    def data_filtering( self, df_filter ):
        """df_filter is the dataframe to be processed according to the DATA FILTERING section"""

        # "truncate" delta_balances to two decimals
        df_filter = df_filter.withColumn("delta_balance_orig", df_filter['delta_balance_orig'].cast( DecimalType(12, 2) ) )
        df_filter = df_filter.withColumn("delta_balance_dest", df_filter['delta_balance_dest'].cast( DecimalType(12, 2) ) )

        # ======= step variable =======
        # Once the reference for the step variable is not clear, 
        # we will remove this variable to avoid misleading interpretations.
        df_filter = df_filter.drop('step')

        # ======= is_flagged_fraud =======
        # Once there are many transactions with amounts more than 200,000 that weren't flagged,
        # the "is_flagged_fraud" feature seems not to be following its description and 
        # this could lead to misleading conclusions.
        # Solution: remove is_flagged_fraud feature and 
        # ask for more information from the business team about this variable
        df_filter = df_filter.drop('is_flagged_fraud')


        return df_filter # data filtering done



    def eda( self, df_eda ):
        """df_eda is the dataframe to be processed according to the EXPLORATORY DATA ANALYSIS section"""
        
        ### no transformation so far

        return df_eda # eda done



    def data_prep( self, df_prep ):
        """df_prep is the dataframe to be processed according to the DATA PREPARATION section"""

        ################################
        # ======= prepare scaling and encoding =======

        ####### RESCALING
        # create a list with columns to rescale
        col_to_rescale = ['new_balance_dest',
                        'old_balance_orig',
                        'orig_rate',
                        'amount']

        # iterate over columns to rescale
        for col in col_to_rescale:            
           
            # instanciate vectorizer
            vectorizer = VectorAssembler( inputCols = [ col ], 
                                        outputCol = f"vec_{ col }" )

            # vectorize the given feature
            df_prep = vectorizer.transform( df_prep )

            # load scaler for transformations
            model_scaler = RobustScalerModel.load( f'{self.features_path}{ col }_rs.spark' )
            
            # rescale feature
            df_prep = model_scaler.transform( df_prep )
            
            # remove unscaled vectorized column
            df_prep = df_prep.drop( f"vec_{ col }" )
            
            
        ####### ENCODING

        # open file with context manager
        with open( f'{self.features_path}type_target_encoder.dict', 'rb') as file:
            # load target encoder
            target_encoder = pickle.load( file )

        # create a function to map type column
        map_func = F.udf(lambda row: target_encoder[ row ][0])

        # map type column
        df_prep = df_prep.withColumn( 'interm_enc', map_func( F.col('type') ) )

        # convert type encoded from string to integer
        df_prep = df_prep.withColumn("interm_enc", df_prep['interm_enc'].cast( DecimalType( precision = 10, scale = 8 ) ) )

        # vectorize type_enc
        vectorizer = VectorAssembler( inputCols = ["interm_enc"], 
                                    outputCol = "type_enc" )

        # vectorize the given feature
        df_prep = vectorizer.transform( df_prep )

        # remove intermediate encode
        df_prep = df_prep.drop('interm_enc')


        return df_prep # dataframe with prepared data



    def make_prediction( self, df_pred ):
        """df_pred is the dataframe to be processed and give a prediction"""

        ################################
        # ======= vectorize columns for prediction =======

        # get selected columns
        selected_cols = ['vec_new_balance_dest_scaled',
                        'type_enc',
                        'vec_old_balance_orig_scaled',
                        'vec_orig_rate_scaled',
                        'vec_amount_scaled']

        # prepare vectorization
        assembler = VectorAssembler( inputCols = selected_cols, outputCol = "features" )

        # vectorize prediction data
        df_pred = assembler.transform( df_pred )


        ################################
        # ======= make prediction =======

        # load ML model
        ml_model = GBTClassificationModel.load( self.model_path )

        # transform prediction dataframe
        df_pred = ml_model.transform( df_pred )
        
        # convert pyspark dataframe to pandas dataframe and return it 
        return df_pred.toPandas() 



    def save_request(self, df_ml_result):
        """df_to_db is the dataframe to be save on mongodb database"""

        # Create a copy of df_ml_result because we will need to convert all columns to str type (to avoid error).
        # See this error few lines below.
        # TO DO: must look for a better solution !!!
        df_to_db = df_ml_result.copy()
        
        # Create a date column on dataframe so that
        # date of prediction will saved on the database
        df_to_db['date'] = datetime.datetime.now()

        # To avoid pymongo error "cannot encode object: Decimal('-9839.64'), of type: <class 'decimal.Decimal'>",
        # we will convert all columns to string.
        # TO DO: must look for a better solution !!!!!!
        df_to_db = df_to_db.astype( str )
    
        # load mongodb endpoint
        FRAUD_DETECTION_MONGODB = os.environ.get( 'DB_FRAUD_DETECTION_CONNECTION' )

        # create a cluster
        cluster = MongoClient( FRAUD_DETECTION_MONGODB )

        # connect to database
        db = cluster['Fraud-Detection-DB']

        # Save request on mongodb database.
        # Note that "api_requests" is the collection inside database
        db.api_requests.insert_many(df_to_db.to_dict('records'))


        return None
