    # Use an official Python runtime as a parent image
    FROM python:3.9-slim

    # Set the working directory in the container
    WORKDIR /app

    # Copy the requirements file into the container at /app
    COPY requirements.txt .

    # Install any needed packages specified in requirements.txt
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy the rest of your app's source code from your host to your container at /app
    COPY . .

    # Expose port 7860 to the outside world
    EXPOSE 7860

    # Run app.py when the container launches
    # Use Gunicorn for a production-ready server
    CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:server"]
    