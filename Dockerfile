# Use the official Python base image with version 3.9
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the necessary files to the container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt
RUN pip install flask

# Expose the port that your Flask application listens on
EXPOSE 8080

# Set environment variables, if necessary
# ENV FLASK_ENV=production

# Specify the command to run your Flask application
CMD ["python", "app.py"]
