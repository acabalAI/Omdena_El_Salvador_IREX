# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"

# Run streamlit_app.py when the container launches
CMD ["streamlit", "run", "app/streamlit_app.py"]



