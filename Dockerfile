# syntax=docker/dockerfile:1
FROM python:3.13.5-bookworm
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /code

# Copy necessary configuration files
COPY . .

RUN pip install uv && uv venv /usr/local/.venv
ENV PATH="/usr/local/.venv/bin:$PATH"
RUN uv pip install -e ".[dev]"

# Expose the port your application will run on
EXPOSE 8000

# Command to run your application using the virtual environment's python/uvicorn
CMD ["uvicorn", "app.main:my_app", "--host", "0.0.0.0", "--port", "8000"]
# CMD tail -f /dev/null
