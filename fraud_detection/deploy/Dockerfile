ARG PYSPARK_SUBMIT_ARGS="--master local[1] pyspark-shell"
ARG JAVA_HOME="usr/bin/java"
ARG uvicorn="/usr/local/bin/uvicorn"
ARG DB_FRAUD_DETECTION_CONNECTION="MONGO-DB-ENDPOINT-HIDDEN-INTENTIONALLY"
FROM ubuntu:20.04
COPY . ./
EXPOSE 8000
WORKDIR ./
RUN apt update
RUN apt-get -y install python3.9
RUN apt-get -y install python3-pip
RUN apt-get -y install openjdk-8-jre
RUN pip install "pyspark==3.2.0"
RUN pip install "fastapi==0.70.0" "uvicorn==0.15.0" "pandas==1.3.4" "inflection==0.5.1" "pymongo==4.0"
CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
