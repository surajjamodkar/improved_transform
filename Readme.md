

# Scaling Gaussian Mixture Models (GMM)

This repository demonstrates scaling Gaussian Mixture Model (GMM) fitting for large datasets using a combination of **Polars** and **Local SparkML**. 

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
   - Example benchmarks on a **MacBook Pro M3 Pro (16GB RAM)** with continuous columns:  
      - 16M rows, 2 columns: **~2 minutes**  
      - 0.2 billion rows, 2 columns: **48 minutes**

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
    Run the notebook `imporved.ipynb`
---


This project is a work in progress.