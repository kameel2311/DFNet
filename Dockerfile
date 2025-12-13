FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Copy requirements file and install dependencies
COPY ./requirements.txt /build/requirements.txt
RUN pip install -r /build/requirements.txt

RUN apt-get update && apt-get install -y \
    libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace
