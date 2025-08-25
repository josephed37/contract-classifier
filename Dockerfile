# --- Stage 1: Base Image ---
# Use an official, lightweight Python image as a parent image.
# The "slim" variant is a good compromise between size and having necessary tools.
FROM python:3.11-slim


# --- Metadata ---
LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="Docker container for the Contract Classification API."

# --- Environment Variables ---
# Set the working directory in the container. All subsequent commands will run from here.
WORKDIR /app

# Set environment variables to prevent Python from writing .pyc files to disc
# and to ensure output is sent straight to the terminal without buffering.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- Dependencies Installation ---
# Copy the requirements file into the container first.
# This is a best practice that leverages Docker's layer caching. If the requirements
# file doesn't change, Docker won't re-run the pip install step on subsequent builds.
COPY requirements.txt .

# Install the Python dependencies specified in requirements.txt.
# --no-cache-dir reduces the image size by not storing the pip cache.
RUN pip install --no-cache-dir -r requirements.txt

# --- Application Code ---
# Copy the rest of the application source code and the models into the container.
# This includes our 'src' directory and the 'models' directory.
COPY ./src ./src
COPY ./models ./models

# --- Expose Port ---
# Expose port 8000 to allow communication with the Uvicorn server running inside the container.
EXPOSE 8000

# --- Command to Run the Application ---
# Define the command that will be executed when the container starts.
# This runs the Uvicorn server, pointing it to our FastAPI app instance in src/main.py.
# --host 0.0.0.0 makes the server accessible from outside the container.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
