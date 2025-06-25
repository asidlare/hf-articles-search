# syntax=docker/dockerfile:1
FROM python:3.12.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV PYTHONUNBUFFERED=1


# Set the working directory
WORKDIR /code

# Copy necessary configuration files
COPY pyproject.toml pyproject.toml

# virtual env, setting path and installation
RUN uv venv /usr/local/.venv
ENV PATH="/usr/local/.venv/bin:$PATH"
RUN uv pip install -e ".[dev]"

# copy project code
COPY . .

# Expose the port your application will run on
EXPOSE 8000

# Command to run your application using the virtual environment's python/uvicorn
# CMD ["python", "app/main.py"]
# CMD tail -f /dev/null
CMD ["uvicorn", "app.main:my_app", "--host", "0.0.0.0", "--port", "8000"]
