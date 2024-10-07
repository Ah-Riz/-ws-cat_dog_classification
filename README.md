# Cat and Dog Detection Web Service

This project implements a web service for detecting cats and dogs in images using machine learning. It utilizes a pre-trained PyTorch model to classify images and provides a user-friendly interface built with Streamlit.

## Features

- Image classification for cats and dogs using a pre-trained PyTorch model
- Web interface using Streamlit for easy interaction
- Asynchronous data fetching and processing
- Integration with Firestore database for data storage and retrieval
- Real-time image processing and prediction

## Prerequisites

- Python 3.7+
- Streamlit
- PyTorch
- Firebase Admin SDK
- Pandas
- AsyncIO

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Ah-riz/cat-dog-detection-webservice.git
   cd cat-dog-detection-webservice
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Firebase credentials:
   - Create a `.env` file in the project root
   - Add your Firestore credentials to the `.env` file:
     ```
     FIRESTORE_DB_COLLECTION=your_collection_name
     GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
     ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run st_app.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the web interface to upload an image of a cat or dog.

4. The application will process the image and display the prediction result.

## Project Structure

- `app.py`: Core application logic
  - Handles image processing and prediction using the trained model
  - Contains utility functions for the main application

- `st_app.py`: Streamlit application
  - Implements the web interface using Streamlit
  - Contains the async `fetch_data()` function for retrieving data from Firestore
  - Manages user interactions and result display

- `assets/`: Directory containing model files
  - `best_model.pt`: Trained PyTorch model for cat and dog detection