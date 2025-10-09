ORANGE VARIETIES CLASSIFICATION
1. Project Overview
This project implements a Deep Learning model using PyTorch to classify four different varieties of oranges based on image data:

- Ibadan Sweet
- Tangerine
- Valencia
- Washington

The system includes scripts for training, evaluation, and prediction, forming a complete workflow for fruit image classification using Convolutional Neural Networks (CNNs).
2. Repository Structure
orange_classification/
│
├─ src/                         # Source Code
│  ├─ train_model.py            # Script to train the model
│  ├─ evaluate.py               # Evaluate model performance
│  ├─ predict.py                # Predict class for new images
│  ├─ plot_history.py           # Plot training history (loss & accuracy)
│  └─ preprocess.py             # Preprocess dataset
│
├─ results/                     # Training results and outputs
│  ├─ history.pkl
│  ├─ model.pth                 # Pretrained model (may be large)
│  ├─ loss_curve.png
│  ├─ accuracy_curve.png
│  ├─ training_summary.csv
│  └─ predictions.csv
│
├─ saved_models/                # Optional: save intermediate models
├─ processed_dataset/            # Preprocessed train/validation sets
├─ test_images/                 # Sample test images
└─ README.md                    # Project documentation
3. Setup Instructions
3.1 Clone the Repository
git clone https://github.com/Clivez12/orange_classification.git
cd orange_classification
3.2 Create and Activate Virtual Environment
Windows:
python -m venv .venv
.venv\Scripts\activate

Linux/Mac:
python3 -m venv venv
source venv/bin/activate
3.3 Install Dependencies
pip install -r requirements.txt
3.4 Download Pretrained Model
If the model file (model.pth) is not included in the repository due to size, download it separately and place it inside the results/ directory.
4. Usage
4.1 Train the Model
python src/train_model.py
- Trains the deep learning model
- Saves the best weights and training history to results/
4.2 Evaluate the Model
python src/evaluate.py
- Generates performance metrics (accuracy, loss)
- Plots training/validation curves
- Exports CSV and text reports
4.3 Make Predictions
python src/predict.py
- Predicts orange variety for a given image or folder
- Saves results to results/predictions.csv
5. Dataset
processed_dataset/
├─ train/
│   ├─ ibadan_sweet/
│   ├─ tangerine/
│   ├─ valencia/
│   └─ washington/
└─ val/
    ├─ ibadan_sweet/
    ├─ tangerine/
    ├─ valencia/
    └─ washington/
Notes:
- Only a small sample dataset is included for demonstration.
- Full dataset can be shared privately on request.
- Folder names must match the class labels exactly.
6. License
This project is developed for academic and research purposes. For commercial or extended use, please contact the authors.
7. Authors & Contact
Author 1: Mr. Shangbom F
Author 2: Terna Henry Wua
Email: henryternawua@gmail.com
GitHub: https://github.com/Clivez12