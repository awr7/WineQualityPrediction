FROM openjdk:11-jdk-slim

# Install wget, net-tools, and procps
RUN apt-get update && apt-get install -y wget net-tools procps

# Download and install Spark
RUN wget https://downloads.apache.org/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz -P /tmp/ && \
    tar xf /tmp/spark-3.5.1-bin-hadoop3.tgz -C /opt/ && \
    mv /opt/spark-3.5.1-bin-hadoop3 /opt/spark && \
    rm /tmp/spark-3.5.1-bin-hadoop3.tgz

# Set environment variables
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:/opt/spark/bin:/opt/spark/sbin

# Copy Hadoop/YARN configuration files
COPY conf /opt/hadoop/conf

# Set environment variables for Hadoop/YARN configurations
ENV HADOOP_CONF_DIR=/opt/hadoop/conf
ENV YARN_CONF_DIR=/opt/hadoop/conf

# Copy the Spark application JAR file to the container
COPY target/WineQualityPrediction-1.0-SNAPSHOT.jar /opt/spark/apps/WineQualityPrediction-1.0-SNAPSHOT.jar

# Expose port 4040 for Spark UI
EXPOSE 4040

# Default command: run the Spark application
CMD ["spark-submit", "--class", "com.yourdomain.WineQualityModel", "--master", "yarn", "/opt/spark/apps/WineQualityPrediction-1.0-SNAPSHOT.jar"]
