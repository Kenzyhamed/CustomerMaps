# Base image with Python
FROM python:3.10

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY app.py .
COPY custom_marks_data.csv .  
# Add data file
COPY tutorial-project-437116-0d601aadcdd2.json /app/tutorial-project-437116-0d601aadcdd2.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/tutorial-project-437116-0d601aadcdd2.json"

# Expose the port Flask will run on
EXPOSE 8081 
# Start the Flask app
CMD ["python", "app.py"]
