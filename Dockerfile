# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project's dependency configuration
COPY pyproject.toml .

# Install dependencies
RUN pip install .[dev,docs]

# Copy the rest of the application's source code from the current directory to the working directory
COPY . .

# Install the project in editable mode
RUN pip install -e .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME neurotheque

# Run the command to start the app
CMD ["bash"]
