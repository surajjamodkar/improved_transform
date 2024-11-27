import polars as pl
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.feature import VectorAssembler
import numpy as np
from pyspark.sql import SparkSession
import os

spark = SparkSession.builder \
    .appName("BGMM_Local") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()
from pyspark.sql.functions import explode, array
from pyspark.ml.linalg import Vectors
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
import pyarrow as pa
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  # Include time
    datefmt="%Y-%m-%d %H:%M:%S"  # Optional: customize the date-time format
)

## imporved transformer class
class DataTransformer():
    
    """
    Transformer class responsible for processing data to train the CTABGANSynthesizer model
    
    Variables:
    1) train_data -> input dataframe 
    2) categorical_list -> list of categorical columns
    3) mixed_dict -> dictionary of mixed columns
    4) n_clusters -> number of modes to fit bayesian gaussian mixture (bgm) model
    5) eps -> threshold for ignoring less prominent modes in the mixture model 
    6) ordering -> stores original ordering for modes of numeric columns
    7) output_info -> stores dimension and output activations of columns (i.e., tanh for numeric, softmax for categorical)
    8) output_dim -> stores the final column width of the transformed data
    9) components -> stores the valid modes used by numeric columns
    10) filter_arr -> stores valid indices of continuous component in mixed columns
    11) meta -> stores column information corresponding to different data types i.e., categorical/mixed/numerical


    Methods:
    1) __init__() -> initializes transformer object and computes meta information of columns
    2) get_metadata() -> builds an inventory of individual columns and stores their relevant properties
    3) fit() -> fits the required bgm models to process the input data
    4) transform() -> executes the transformation required to train the model
    5) inverse_transform() -> executes the reverse transformation on data generated from the model
    
    """   
    def __init__(self, train_data=pl.DataFrame, categorical_list=[], mixed_dict={}, n_clusters=10, eps=0.005):
        
        self.meta = None
        self.train_data = train_data
        self.categorical_columns= categorical_list
        self.mixed_columns= mixed_dict
        self.n_clusters = n_clusters
        self.eps = eps
        self.ordering = []
        self.output_info = []
        self.output_dim = 0
        self.components = []
        self.filter_arr = []
        self.meta = self.get_metadata()

    def get_metadata(self):
        meta = []
        logging.info('Storing metadata for columns.')
        for index in self.train_data.columns:
            column = self.train_data[index]
            if index in self.categorical_columns:
                mapper = column.value_counts().sort('count', descending= True).select(pl.col(index)).to_numpy().tolist()
                meta.append({
                        "name": index,
                        "type": "categorical",
                        "size": len(mapper),
                        "i2s": mapper
                })
            elif index in self.mixed_columns.keys():
                meta.append({
                    "name": index,
                    "type": "mixed",
                    "min": column.min(),
                    "max": column.max(),
                    "modal": self.mixed_columns[index]
                })
            else:
                meta.append({
                    "name": index,
                    "type": "continuous",
                    "min": column.min(),
                    "max": column.max(),
                })
        logging.info('Finished storing metadata for columns.')       
        return meta
    
    def fit(self):
        # stores the corresponding bgm models for processing numeric data
        model = []
        
        # iterating through column information
        for id_, info in enumerate(self.meta):
            if info['type'] == "continuous":
                # fitting bgm model from SparkML to handle partitioned large datasets
                gm = GaussianMixture(
                    k=self.n_clusters,
                    maxIter=100,
                    seed = 42, 
                    )
                current_column = info['name']
                current_data = self.train_data.select(pl.col(current_column))

                # calculate number of partitions - this affects the speed of model fitting and may need to be retuned
                num_partitions = max(10, int(current_data.shape[0]/50000))

                # saving a polars DF to parquet and reading in spark is fastest way of converting without keeping in memory
                current_data.write_parquet('temp.parquet')
                del current_data
                data_temp = spark.read.parquet('temp.parquet')
                data_temp = data_temp.repartition(num_partitions)
                assembler = VectorAssembler(inputCols=[current_column], outputCol="features")
                transformed_df = assembler.transform(data_temp)

                # fit GMM 
                logging.info(f'fitting model for {current_column}')
                gm_fit = gm.fit(transformed_df)
                model.append(gm_fit)
                
                # means and stds of the modes are obtained from the corresponding fitted bgm model
                m_cov = gm_fit.gaussiansDF.toPandas()
                # Flatten 'mean' column (extract the first element from the list)
                m_cov['mean'] = m_cov['mean'].apply(lambda x: x[0])

                # Extract scalar value from DenseMatrix in the 'cov' column
                m_cov['cov'] = m_cov['cov'].apply(lambda x: x[0, 0] if hasattr(x, 'toArray') else x)
                m_cov['stds'] = np.sqrt(m_cov['cov'])
                m_cov['feat_num'] = m_cov.index

                m_cov = pl.DataFrame(m_cov[['mean','stds', 'feat_num']])

                # Save the means and std.dev of features in meta which can be referred later
                info['m_cov'] = m_cov

                self.model = model
                logging.info(f'model fit completed for {current_column}')
                # keeping only relevant modes that have higher weight than eps and are used to fit the data
                old_comp = np.array(gm_fit.weights) > self.eps

                #mode_freq = tuple(current_data[current_column].value_counts().sort('count', descending= True).select(pl.col(current_column)).to_numpy())
                mode_freq = list(range(self.n_clusters))
                comp = []
                for i in range(self.n_clusters):
                    if (i in (mode_freq)) & old_comp[i]:
                        comp.append(True)
                    else:
                        comp.append(False)
                self.components.append(comp) 
                self.output_info += [(1, 'tanh'), (np.sum(comp), 'softmax')]
                self.output_dim += 1 + np.sum(comp)
        
        self.model = model
        logging.info('Finished fitting all models.')

    def transform(self, data: pl.DataFrame, save_parquet:bool = True):
        # stores the transformed values
        values = []

        # used for accessing filter_arr for transforming mixed columns
        mixed_counter = 0

        # create directory to store intermediate values
        os.makedirs('feature_values/', exist_ok=True)

        # iterating through column information
        for id_, info in enumerate(self.meta):
            if info['type'] == "continuous":
                current_column = info['name']
                logging.info(f'Creating feature values for column : {current_column}')
                current_data = data.select(pl.col(current_column))
                current_data = current_data.with_row_index()
                
                # get saved means and std.devs for features
                m_cov = info['m_cov']
                

                features = current_data.join(m_cov, how = 'cross')
                features = features.with_columns(((pl.col(current_column) - pl.col('mean'))/(4* pl.col('stds'))).alias('feat_value'))

                # number of distict modes
                n_opts = sum(self.components[id_])          
                # storing the mode for each data point by sampling from the probability mass distribution across all modes based on fitted bgm model 
                # create sparkml vectors for predicting class probabilities
                num_partitions = max(10, int(current_data.shape[0]/50000))
                current_data.write_parquet('temp.parquet')
                del current_data
                data_temp = spark.read.parquet('temp.parquet')
                data_temp = data_temp.repartition(num_partitions)
                assembler = VectorAssembler(inputCols=[current_column], outputCol="features")
                transformed_df = assembler.transform(data_temp)

                # create polars df with predicted probaility
                probs = self.model[id_].transform(transformed_df).select('probability')
                probs = probs.select(vector_to_array("probability").alias("probability"))
                probs = probs.select(*[col('probability').getItem(i).alias(f'prob_{i}') for i in range(0, n_opts+1)])

                col_names = probs.columns
                columns_to_select = [col for col, to_select in zip(col_names, self.components[id_]) if to_select]
                probs = probs.select(columns_to_select)
                probs = pl.from_arrow(pa.Table.from_batches(probs._collect_as_arrow()))

                #relevel probability
                probs = probs.with_columns(sum = pl.sum_horizontal(pl.all()))
                probs = probs.with_columns(pl.all()/pl.col('sum'))
                probs = probs.drop('sum')
                
                # select most probable feature 
                feature_options = [i for i in range(n_opts)]
                probs = probs.with_columns(pl.concat_list(pl.all()).alias('prob_list'))
                probs = probs.with_columns(pl.col("prob_list").map_elements(lambda prob_list: np.random.choice(feature_options, p = prob_list) , return_dtype=pl.Int64).alias("sel_feature"))
                probs = probs.with_row_index()

                # obtaining the normalized values based on the appropriately selected mode and clipping to ensure values are within (-1,1)
                #select only appropriate features
                sel_feat_num = list(range(self.n_clusters))
                sel_feat_num = [col for col, to_select in zip(sel_feat_num, self.components[id_]) if to_select]
                

                # select feature value for selected random component 
                features = features[['index', 'feat_value', 'feat_num']].join(probs[['index', 'sel_feature']], how = 'inner', left_on=['index', 'feat_num'], right_on= ['index', 'sel_feature'])
                # clip feat value
                features = features.with_columns(pl.col('feat_value').clip(-0.99, 0.99))

                feature_rank = probs['sel_feature'].value_counts().sort(pl.col('count'),descending=True).with_row_index('rank')
                feature_rank = feature_rank[['rank', 'sel_feature']]
                
                features = features.join(feature_rank, how = 'left', coalesce= True, left_on= ['feat_num'], right_on= ['sel_feature'])

                # values += (current_column, features)
                # Save feature values - feature values are stored as a long DataFrame instead of wide table with feature value and one hot encoded selected feature number
                # the DF has information of feature number, calculated feature value 
                
                features.write_parquet(f'feature_values/{current_column}.parquet')
                logging.info(f'saved intermediate feature value for column : {current_column}')

                # storing the original ordering for invoking inverse transform
                self.ordering.append(feature_rank)
                info['feature_rank'] = feature_rank

            values = pl.DataFrame()

            # iterating through column information

        #following block converts the long format of saved feature values to desired format. 
        # the output is still a parquet or polars DF (will need changes downline)
        for id_, info in enumerate(self.meta):
            if info['type'] == "continuous":
                logging.info(f'Converting intermediate long format to transformed values for column :  {current_column}')
                current_column = info['name']
                feature_rank = info['feature_rank']
                n_features = feature_rank.shape[0]
                feature_val = pl.read_parquet(f'feature_values/{current_column}.parquet')
                feature_val = feature_val.with_columns([
                                (pl.col("rank") == i).cast(pl.Int64).alias(f"{current_column}_{i}")
                                for i in range(n_features) ])
                sel_columns = ['feat_value'] + [f"{current_column}_{i}" for i in range(n_features) ]
                feature_val = feature_val[sel_columns]
                feature_val = feature_val.rename({'feat_value':current_column})
                values = pl.concat([values, feature_val], how = 'horizontal')

        # Save transformed values
        if save_parquet:
            os.makedirs('transformed_values/', exist_ok= True)
            parquet_path = f'transformed_values/transformed_values.parquet'
            values.write_parquet(parquet_path)
            logging.info(f'Transformed Parquets saved at {parquet_path} ')
        else:
            return values

    def inverse_transform(self, data: pl.DataFrame, save_parquet:bool = True):

        inverted_data = pl.DataFrame()
        # iterating through column information

        for id_, info in enumerate(self.meta):
            if info['type'] == "continuous":
                current_column = info['name']
                logging.info(f'Started inverse transform for column :  {current_column}')
                feature_rank = info['feature_rank']
                m_cov = info['m_cov']
                n_features = feature_rank.shape[0]

                # first reverse the one hot encoded columns to get feature rank
                ohe_cols =  [f"{current_column}_{i}" for i in range(n_features) ]
                ohe_df =  data[ohe_cols]
                ohe_df = ohe_df.with_columns(pl.concat_list(pl.all()).alias("rank"))
                ohe_df = ohe_df.with_columns(pl.col('rank').list.arg_max())

                # get feature number from rank and get std and mean

                inreversing_df = ohe_df[['rank']].join(feature_rank, on = ['rank'], how = 'left', coalesce= True)
                inreversing_df = inreversing_df.join(m_cov, how = 'left', left_on = ['sel_feature'], right_on= ['feat_num'], coalesce= True)

                # join feature value and inversing df 
                inverted_df = pl.concat([data[[current_column]], inreversing_df], how = 'horizontal')
                inverted_df = inverted_df.with_columns((pl.col(current_column) * (4* pl.col('stds')) + pl.col('mean')).alias(current_column))
                inverted_df = inverted_df[[current_column]]
                inverted_data = pl.concat([inverted_data, inverted_df], how = 'horizontal')
                logging.info(f'Finished inverse transform for column :  {current_column}')
        if save_parquet:
            os.makedirs('inverted_values/', exist_ok= True)
            parquet_path = f'inverted_values/inverted_values.parquet'
            inverted_data.write_parquet(parquet_path)
            logging.info(f'Inverse transformed Parquets saved at {parquet_path} ')
        else: 
            return inverted_data

