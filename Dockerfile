FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Copy requirements file and install dependencies
COPY ./requirements.txt /build/requirements.txt
RUN pip install -r /build/requirements.txt

# Set the working directory
WORKDIR /workspace
