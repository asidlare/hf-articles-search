# syntax=docker/dockerfile:1
FROM python:3.13.5-bookworm
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /code

# Install pip-tools
RUN pip install pip-tools

# Copy only the pyproject.toml and create and install requirements.txt
COPY pyproject.toml /code/
RUN pip-compile --extra=dev pyproject.toml --output-file=requirements.txt
RUN pip install -r requirements.txt

# Expose the port your application will run on
EXPOSE 8000

# Command to run your application using the virtual environment's python/uvicorn
CMD ["uvicorn", "app.main:my_app", "--host", "0.0.0.0", "--port", "8000"]
#CMD tail -f /dev/null


