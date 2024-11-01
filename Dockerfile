# Dockerfile
FROM pytorch/pytorch:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_LAUNCH_BLOCKING=1
ENV TORCH_USE_CUDA_DSA=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter Lab dependencies
RUN apt-get update && apt-get install -y nodejs npm
RUN pip install jupyterlab

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Start Jupyter Lab when the container launches
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]