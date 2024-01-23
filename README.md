# Rotational and Scale Equivariant Convolutional Neural Network

This repository contains the implementation of a Convolutional Neural Network (CNN) with rotational and scale equivariance. The CNN is designed to work with image data, particularly in the context of image classification tasks.



## Introduction
The provided code implements a custom CNN architecture with rotational and scale equivariant convolutional layers. These layers aim to capture rotational and scale-invariant features in the input data, making the model suitable for a variety of image recognition tasks.

## Running the Code
1. **Create a Python Virtual Environment:**
    - For Windows:
      ```bash
      cd path/to/your/project
      python3 -m venv venv_name
      .\venv_name\Scripts\activate
      ```
    - For macOS/Linux:
      ```bash
      cd path/to/your/project
      python3 -m venv venv_name
      source venv_name/bin/activate
      ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the `main.py` File:**
    ```bash
    python3 main.py
    ```