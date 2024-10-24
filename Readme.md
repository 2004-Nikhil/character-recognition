# Handwritten Character Recognition

This project is a web application for recognizing handwritten characters using a Convolutional Neural Network (CNN) model. The application is built using Flask for the backend and HTML/CSS/JavaScript for the frontend. The model is trained on the EMNIST dataset.

## Table of Contents

- Installation
- Usage
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [API Endpoints](#api-endpoints)
- Contributing
- License

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/character-recognition.git
    cd character-recognition
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Download the EMNIST dataset:
    - Place your `kaggle.json` file in the root directory of the project.
    - Run the training script to download and preprocess the dataset:
        ```sh
        python training.py
        ```

5. Start the Flask application:
    ```sh
    python app.py
    ```

## Usage

1. Open your web browser and navigate to `http://127.0.0.1:5000/`.
2. Use the canvas to draw a character.

![Character Recognition](https://ibb.co/dmwWCGk)
## Project Structure

```
character-recognition/
    ├── .gitignore
    ├── app.py
    ├── model/
    │   └── model.h5
    ├── package.json
    ├── requirements.txt
    ├── static/
    │   └── scripts.js
    ├── templates/
    │   └── index.html
    └── training.py
```


: The main Flask application.
- `app.py`: Start the webapp from here
- `model/`: Directory containing the trained model.
- `requirements.txt`: List of Python packages required for the project.
- `static/`: Directory containing static files (JavaScript).
- `templates/`: Directory containing HTML templates.
- training.py

: Script for training the CNN model.

## Model Training

The model is trained using the EMNIST dataset. The training script (training.py) performs the following steps:

1. Downloads and extracts the EMNIST dataset.
2. Preprocesses the data (reshaping, normalizing, etc.).
3. Builds and trains a CNN model.
4. Saves the trained model to `model/model.h5`.

To train the model, run:
```sh
python training.py
```

## API Endpoints

- `GET /`: Renders the home page.
- `POST /predict`: Accepts a JSON payload with base64-encoded image data and returns the prediction results.

### Example Request

```json
{
    "image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### Example Response

```json
{
    "results": [0.1, 0.2, 0.7, ...]
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.