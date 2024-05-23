# Set the base image (for a simple deploy use the python image with your needed version)
# Use command FROM
FROM python:3.12.13

# Copy all files that you need for your workdir (your name choice)
COPY ./requirements.txt /work_dir/
COPY ./server.py /work_dir/
COPY ./best_model.pkl /work_dir/

# Set your workdir (move "yourself" to your workdir to run commands inside there)
WORKDIR /work_dir/

# Install your app requirements and run necessary commands
RUN pip install -r requirements.txt

# Expose a port to your container listen
EXPOSE 80

# Define the command to execute your container
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
