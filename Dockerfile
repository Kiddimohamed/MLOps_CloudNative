# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY  requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/



# Make port 5000 available to the world outside this container
EXPOSE 5000


# Run flask_app.py when the container launches
#CMD ["python", "flask-app.py"]
# Command to run the application using Uvicorn
CMD ["uvicorn", "app:flask-app", "--host", "0.0.0.0", "--port", "5000"]