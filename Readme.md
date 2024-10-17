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
3. Click the "Predict" button to get the prediction results.

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
    │   ├── scripts.js
    │   └── styles.css
    ├── templates/
    │   └── index.html
    └── training.py
```


: The main Flask application.
- `app.py`: Start the webapp from here
- `model/`: Directory containing the trained model.
- `requirements.txt`: List of Python packages required for the project.
- `static/`: Directory containing static files (JavaScript, CSS).
- `templates/`: Directory containing HTML templates.
- training.py

: Script for training the CNN model.

## Model Training

The model is trained using the EMNIST dataset. The training script (training.py) performs the following steps:

1. Downloads and extracts the EMNIST dataset.
2. Preprocesses the data (reshaping, normalizing, etc.).
3. Builds and trains a CNN model.
4. Saves the trained model to [`model/model.h5`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fworkspaces%2Fcodespaces-blank%2Fcharacter-recognition%2Fapp.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A11%2C%22character%22%3A20%7D%7D%5D%2C%22d0478508-a4c0-41fc-9ec4-da87a7b6659b%22%5D "Go to definition").

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