# GAN-ESH: GAN-Based Steganography

## Project Overview
GAN-ESH is a project focused on implementing GAN-based steganography, enabling the secure hiding of GIFs within images using Generative Adversarial Networks (GANs). This repository includes scripts for defining the model, training it, and exploring results through an interactive notebook.

## Features
- **GAN Model Architecture**: Implementation of a GAN for steganographic tasks.
- **Training Pipeline**: Script to train the GAN model.
- **Interactive Notebook**: `gan_steg.ipynb` provides a step-by-step guide to running and understanding the project.

## Files in the Repository
1. **`.env`**: Environment configuration file.
2. **`gan_steg.ipynb`**: Jupyter Notebook for exploring the GAN's functionality.
3. **`model.py`**: Contains the GAN model architecture.
4. **`train.py`**: Script to train the GAN on the dataset.

## Installation

### Prerequisites
Ensure the following are installed:
- Python 3.8+
- Jupyter Notebook

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd GAN-ESH-main
   ```
2. Configure environment variables in `.env`.

## Usage

### Running the Notebook
1. Open `gan_steg.ipynb` in Jupyter Notebook:
   ```bash
   jupyter notebook gan_steg.ipynb
   ```
2. Follow the steps in the notebook to run the model and explore its outputs.

### Training the Model
1. Edit `train.py` to specify your dataset path and hyperparameters.
2. Run the script:
   ```bash
   python train.py
   ```

## Contributions
Feel free to fork this repository and submit pull requests for enhancements or bug fixes.

## License
This project is licensed under [MIT License](LICENSE).
