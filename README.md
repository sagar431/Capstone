# Multi-Modal LLM Chat

This project implements a multi-modal language model chat interface using a fine-tuned Phi-1.5 model and CLIP for image understanding.

## Features

- Text-based chat using a fine-tuned Phi-1.5 model
- Image description using CLIP embeddings
- Audio transcription and response using Whisper

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app.py`

## Usage

- Enter text in the text box for a text-based conversation
- Upload an image for image description
- Upload an audio file for transcription and response

## Model Information

- Text model: Fine-tuned Phi-1.5 on OpenAssistant dataset (sagar007/phi-1_5-finetuned)
- Image understanding: CLIP (ViT-B/32)
- Audio transcription: Whisper (base model)

## Deployment

This app is designed to be deployed on Hugging Face Spaces. Link your GitHub repository to a new Hugging Face Space for automatic deployment.