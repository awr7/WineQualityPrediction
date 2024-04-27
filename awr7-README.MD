# Wine Quality Prediction using Spark MLlib

 The application uses logistic regression and is designed to run both locally and on AWS EMR.

## Project Overview

The project involves setting up an AWS EMR cluster to train a machine learning model using Apache Spark, creating a Docker container for the application, and deploying it on a standalone EC2 instance. The application is capable of being run directly on the EC2 instance or within a Docker container.

### Step 1: Create an EMR Cluster on AWS EC2

- **Cluster Name**: ML Cluster v1
- **EMR Release**: emr-7.1.0
- **Applications**: Spark Interactive (Hadoop 3.3.6, Hive 3.1.3, JupyterEnterpriseGateway 2.6.0, Livy 0.8.0, Spark 3.5)
- **Cluster Configuration**: Uniform instance groups with Primary (m5.xlarge), Core (m5.xlarge), and Task (m5.xlarge) instances
- **Provisioning Configuration**: 
  - Core size: 4 instances
  - Task size: 1 instance
- **Networking**: 
  - VPC: vpc-02e364994
  - Subnet: subnet-06eff9
  - Security groups for Primary and Core nodes
- **Cluster Termination**: Manually terminate cluster
- **Security Configuration**: 
  - Amazon EC2 key pair: assignment2Key
  - IAM roles: EMR_DefaultRole, EMR_EC2_DefaultRole

### Step 2: Application Development

Developed a Java application using Apache Spark MLlib to perform logistic regression. The application is designed to use a cross-validator to enhance model accuracy.

### Step 3: Dockerization of the Application

- Created a Dockerfile to build a Docker image of the Spark application.
- Built the Docker image and pushed it to Docker Hub for deployment.

  Docker Hub Repository: [angelclasico/winequalityprediction](https://hub.docker.com/repository/docker/angelclasico/winequalityprediction/general)

### Step 4: Setup Standalone EC2 Instance

- Created a standalone EC2 instance.
- Installed Java, Docker, and Apache Spark on the instance to prepare for both direct application execution and Docker-based execution.

### Step 5: Deploying the Application

- Uploaded the JAR file of the application to the EC2 instance.
- Command to submit the Spark job directly on the EC2 instance:

  ```bash
  spark-submit --master spark://ec2-3-91-13-32.compute-1.amazonaws.com:7077 \
  --class com.yourdomain.WineQualityModel \
  /home/ubuntu/WineQualityPrediction-1.0-SNAPSHOT.jar \
  s3://aws-logs-058264353971-us-east-1/TrainingDataset.csv \
  s3://aws-logs-058264353971-us-east-1/ValidationDataset.csv

### Step 6: Deploying with Docker

- Pull the docker image from Docker Hub.
- Run the Docer container on the EC2 instance which would do the same operations as if it was direct.