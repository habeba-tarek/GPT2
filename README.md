# GPT2
A from-scratch implementation of a GPT-2-like transformer model using PyTorch, trained on the TinyStories dataset

This project was built for educational purposes to gain a deep, hands-on understanding of the transformer architecture and its components, without relying on PyTorch's built-in transformer modules.

📖 Project Overview
This repository contains the complete code for building, training, and evaluating a 12-layer decoder-only transformer model inspired by GPT-2. The model features multi-head self-attention, positional encoding, feed-forward networks, and layer normalization—all implemented manually. It was trained on the TinyStories dataset to generate short, coherent English narratives.

Key aspects of the project:

Custom Implementation: All core components (Self-Attention, LayerNorm, Positional Encoding, etc.) are built from the ground up.

Training Pipeline: Includes data preprocessing, tokenization, training loop with AdamW, and gradient clipping.

Evaluation: Model performance is measured using perplexity and qualitative text generation.

Visualization: Features plots for training loss, perplexity, and a dataset word cloud.

🏗️ Model Architecture
The model architecture closely follows the original GPT-2 design:

Layers: 12 decoder-only transformer layers.

Model Dimensions: d_model = 768, d_ff = 3072.

Attention: 12 attention heads (d_k = d_v = 64).

Activation: ReLU in the feed-forward network.

Regularization: Dropout (p=0.1) and Layer Normalization after each sub-layer.

Vocabulary Size: 50,257 (using GPT-2 tokenizer from Hugging Face).

Positional Encoding: Sinusoidal.

📊 Dataset
We used the TinyStories dataset, which contains approximately 2.7 million short stories written in simple English. The dataset is ideal for this project as it is manageable in size while still requiring the model to learn grammar, logic, and narrative flow.

Preprocessing: Text was tokenized using the GPT-2 tokenizer (Byte-Pair Encoding).

Sequence Length: Sequences were truncated/padded to a maximum length of 128 tokens.

Train/Validation Split: 90% for training, 10% for validation.

https://via.placeholder.com/600x400/eee/333?text=Word+Cloud+Visualization
Word cloud showcasing the most frequent tokens in the dataset.

⚙️ Training Details
Optimizer: AdamW (lr=0.0001, betas=(0.9, 0.999))

Loss Function: Cross-Entropy for next-token prediction.

Batch Size: 32

Epochs: 5

Gradient Clipping: Norm of 1.0

Hardware: Training was conducted on an NVIDIA GPU via Google Colab.

https://via.placeholder.com/600x400/eee/333?text=Training+Loss+Plot
The training and validation loss curve over 5 epochs.

📈 Results & Evaluation
After 5 epochs of training:

Training Perplexity: ~43,030

Validation Perplexity: ~42,385

Generated Text Sample (Prompt: "Once upon a time"):

"Once upon a time, there was a little girl who lived in a little house. She was a little girl. She was a little girl. She was a little girl. She was a little girl. She was a little girl."

Analysis:
The high perplexity and repetitive generated text indicate that the model began learning basic word patterns but did not fully converge or capture long-range narrative dependencies. This is expected given the limited number of training epochs and model size compared to the original GPT-2.


