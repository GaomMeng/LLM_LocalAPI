# LLaMA 3.1 Model Deployment with FastAPI

## Introduction

This project demonstrates how to deploy a large language model, specifically Meta's LLaMA 3.1 (8B), using FastAPI. Large Language Models (LLMs) like LLaMA are powerful tools trained on massive datasets to understand and generate human-like text. Deploying these models as an API allows developers to integrate advanced language understanding and generation capabilities into applications, providing significant advantages in natural language processing tasks.

## Why Deploy LLaMA 3.1 with FastAPI?

Deploying LLaMA 3.1 as an API using FastAPI offers several benefits:

1. **Scalability**: FastAPI is designed to support asynchronous request handling, making it suitable for high-concurrency environments, ensuring your model can scale effectively.
2. **Ease of Use**: FastAPI automatically generates interactive API documentation, simplifying testing and integration.
3. **Performance**: FastAPI, combined with PyTorch and Transformers, ensures optimal performance in model inference, with support for GPU acceleration.
4. **Flexibility**: The deployment is customizable, allowing fine-tuning of model parameters, including tokenization, generation length, and sampling techniques.

## Model Parameters

### Key Parameters in the Deployment:

- **`max_new_tokens`**: Limits the length of the generated text to avoid overly long or repetitive outputs. Set to `128` in this deployment.
- **`do_sample`**: Enables sampling to add variability in the model's output, preventing it from generating the same response each time.
- **`temperature`**: Controls the randomness of predictions by scaling the logits before applying softmax. A higher temperature (e.g., `0.7`) makes the output more random.

These parameters allow you to balance between coherent and creative outputs depending on your use case.

## Project Structure

This project contains two main files:

1. **`main.py`**: Contains all the logic for deploying the LLaMA 3.1 model as a FastAPI service. It handles requests, processes inputs, and returns generated text.
   
2. **`frontend.html`**: Stores the HTML content for the web interface. The interface allows users to interact with the model by sending messages and receiving generated responses.

### `main.py`
- **Model Deployment**: Configures the LLaMA 3.1 model pipeline using the `transformers` library.
- **API Endpoints**: 
  - `/generate`: POST endpoint that accepts user messages, processes them, and returns the model's generated text.
  - `/`: GET endpoint that serves the HTML frontend.
- **Utility Functions**: 
  - `extract_system_reply`: Extracts the system's reply from the generated text.
  - `processing_message`: Formats the conversation history for input to the model.

### `frontend.html`
- **UI Design**: A simple chat interface built with HTML, Tailwind CSS, and JavaScript. 
- **JavaScript Functions**:
  - `sendMessage()`: Sends user input to the backend API and displays the model's response.
  - `newChat()`: Clears the chat history for a new conversation.

## How to Run the Project

To run this project locally, follow these steps:

### Prerequisites

- Python 3.7+
- PyTorch with GPU support (if using a CUDA-capable GPU)
- `transformers` library
- `fastapi` and `uvicorn`

### Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/GaomMeng/LLM_LocalAPI.git
    cd llama3.1-fastapi
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application**:
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

4. **Access the Application**:
   - Open your web browser and navigate to `http://localhost:8000` to use the chat interface.
   - The API documentation is available at `http://localhost:8000/docs`.

## Customization

You can customize the behavior of the model and the frontend by modifying the parameters in `main.py` and the design elements in `frontend.html`.

- **Model Path**: Update `model_path` in `main.py` if your model is stored in a different location.
- **Frontend Design**: Modify `frontend.html` to change the UI appearance or add new features.

## Conclusion

This project serves as a starting point for deploying LLaMA 3.1 or other large language models in production environments. By combining FastAPI's efficiency with the power of LLaMA, you can build applications that leverage advanced language processing capabilities with ease and flexibility.

Feel free to fork the project and adapt it to your specific needs. Contributions and suggestions are always welcome!
