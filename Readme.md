

# Scaling Gaussian Mixture Models (GMM)

This repository demonstrates scaling Gaussian Mixture Model (GMM) fitting for large datasets using a combination of **Polars** and **Local SparkML**. 

## Approach To Scale
1. Go through the class and run the methods for sample dataset. 
2. Deeper look in the class and method using debugging. See how the data is being transformed step by step. 
3. Understanding why things are being done in a certain way and then see if there are alternatives which give same result (could be in a different form). e.g. value_counts based sorting instead of creating one hot encoded df and counting frequencies through sum.
4. Keeping data in most efficient form till it is needed to be presented. 
   There are instances where operations could be performed on a long format data instead of wide format. e.g. feature and its rank can be stored differently than feature value for a column. When we want to return the output, we can combine both in any desired form. Making it wider and appending as we iterate over columns will take large amount of memory. Long (columnar) format data also allows for faster aggregations like sum, mean etc. 
   When needed, we can join the information.
   Long/Columnar format are faster to write and read and time increases with number of columns. 
5. Iterations can be rewritten in columnar or vectorized form. E.g. 
   ``` python 
                   for i in range(len(current)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)
   ```
   Relevels the probability for a row in *current* array. 
   If we use polars, we have vectorized operation and no looping. This can be further imporved by making *probs* long. 
   ``` python
               probs = probs.with_columns(sum = pl.sum_horizontal(pl.all()))
               probs = probs.with_columns(pl.all()/pl.col('sum'))
   ```
6. Certain Sklearn models can use multiple cores. GMM cannot and that why we choose spark-ML as our modelling fit option. 
7. I tested spark for data transformations as well, but found polars to be faster as the number of operations are larger (needing more swaps) and polars was able to process about 0.5 billion rows.

---

## To-Do List For productionization
1. Seperate out operations and save not only intermediate results but the models as well. We will serialize the models. 
A new folder structure by columns could be created. 
--Fit_1/
      Amount/
         model/
            model.model
         feature_values/
            feature_value.parquet
         transformed_values/
            transformed_values.parquet
      V1/
         model/
            model.model
         feature_values/
            feature_value.parquet
         transformed_values/
            transformed_values.parquet
  
2. Recovery/roubust operation. If a fit for a column fails, we want to have a option to retrain on all columns or only the remaining ones. We will also be able to load models fit previously.
3. Depolyment options - Containerize the application and deploy as a serverless operation (Azure app service connected with a storage account for intermediate files), use exposed api to post and get results for use in userfacing operations
4. Depolyment option - Scheduled - Databricks with scheduler - this provides storage, scheduling and spark compatibility
   - Useful for internal computations
5. Create a package.
    
---
## Features
1. **Optimized Data Transformations**:  
   - Most data transformations are handled using **Polars** for performance.
   - Model fitting is performed using **SparkML**, enabling efficient handling of large datasets.
   
2. **Dynamic Partitioning**:  
   - The number of partitions is dynamically calculated based on the size of the dataset to optimize resource usage.

3. **Efficient Data Persistence**:  
   - Intermediate results and outputs are saved as **Parquet files** for faster loading and reduced memory consumption.

4. **Scalable Performance**:  
   - Execution time scales almost linearly with the dataset size.  
   - Model fit takes around 80% time. Transform and inverse transform are optimized and scale linearly.
   - Example benchmarks on a **MacBook Pro M3 Pro (12 Core, 18GB RAM)** with continuous columns:  
      - 0.8M rows, 2 columns: **~1 minutes**  
      - 3.2M rows, 2 columns: **3.2 minutes**  (4 GB Memory used)
      - 16M rows, 2 columns: **~2 minutes**  
      - 51M rows, 2 columns: **15 minutes**  11:49:48
      - 0.2 billion rows, 2 columns: **48 minutes**
      - Extrapolated to 1 billion rows, 2 cols : **240 minutes**
      - Using Databricks m5.12xlarge | 48CPUs | 192 GB Memory : **estimated 48-60 minutes** 

---

## Memory Optimization Techniques
1. **Long Table Operations**:  
   - Converted wide table operations to long table operations, switching to wide format only when necessary.

2. **Feature Ranking Persistence**:  
   - Feature rankings are persisted, enabling faster transformations and inverse transformations.

3. **Optimized Sorting and Selection**:  
   - **Polars** joins and filters are used for efficient sorting and feature selection based on rankings.

4. **Interfacing Polars and Spark**:  
   - Spark DataFrames and Polars DataFrames are interconverted using **Parquet files** for seamless integration.

5. **Improved One-Hot Encoding (OHE)**:  
   - Logic for converting to one-hot encoded columns and inverse transformations has been enhanced to accommodate feature ranking.

---

## To-Do List
1. **Validate Inverse Transformations**:  
   - Ensure inverse transformations are accurate for all data values (currently validated on sample data).  

2. **Support for Mixed and Categorical Columns**:  
   - Extend functionality to handle mixed and categorical columns, replicating the logic used for continuous columns.  

3. **Optimized Spark Configuration**:  
   - Develop guidelines for configuring Spark based on local or cloud-based compute environments. This process currently relies on trial-and-error with some thumb rules.  

3. **Testing**:  
   - Unit tests and corner cases need to be addressed.
    
---

Here’s a polished and professional README for your project:

---

# Project Setup Guide

Follow these steps to set up and run the project:

## Prerequisites

Ensure you have Python installed on your system. You can verify by running:  
```bash
python --version
```

## Installation Steps

1. **Install `uv`**  
   Install the `uv` CLI tool to manage virtual environments:
   ```bash
   pip install uv
   ```

2. **Create a Virtual Environment**  
   Use `uv` to create a virtual environment for the project:
   ```bash
   uv venv
   ```

3. **Activate the Virtual Environment**  
   Activate the virtual environment to isolate project dependencies:
   ```bash
   source .venv/bin/activate
   ```
   > **Note for Windows Users**:  
   Use the following command instead:  
   ```bash
   .venv\Scripts\activate
   ```

4. **Install Project Dependencies**  
   Install all required packages listed in `requirements.txt`:
   ```bash
   uv pip install -r requirements.txt
   ```
5. **Run the notebook**
    Run `src/run_gmm.py` OR
    Run the notebook `imporved.ipynb`

---


This project is a work in progress.