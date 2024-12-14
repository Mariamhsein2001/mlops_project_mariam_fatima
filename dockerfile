# Use an official Python runtime as the base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install pipx and Poetry in a single layer
RUN pip install --no-cache-dir pipx && \
    pipx install poetry && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry


# Copy only dependency files first to leverage Docker's caching
COPY poetry.lock pyproject.toml /app/

# Install dependencies with Poetry
RUN poetry install --no-root --no-cache

# Copy the rest of the project files
COPY . /app

# Explicitly install the project in editable mode
RUN poetry run pip install -e .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["poetry", "run", "uvicorn", "src.air_pollution.main:app", "--host", "0.0.0.0", "--port", "8000"]