
🟧 ORANGE VARIETIES CLASSIFICATION

(Click the badge above once deployed to open the live demo.)

🧠 1️⃣ Project Overview
This project implements a Deep Learning model using PyTorch to classify four different varieties of oranges based on image data:
•	🍊 Ibadan Sweet
•	🍊 Tangerine
•	🍊 Valencia
•	🍊 Washington
It includes scripts for training, evaluation, and prediction, forming a complete workflow for fruit image classification using Convolutional Neural Networks (CNNs) and a user-friendly Streamlit web interface.

🗂️ 2️⃣ Repository Structure
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
│  ├─ model.pth
│  ├─ loss_curve.png
│  ├─ accuracy_curve.png
│  ├─ training_summary.csv
│  └─ predictions.csv
│
├─ saved_models/                # Saved model checkpoints (best_model.pth)
├─ processed_dataset/           # Preprocessed train/validation sets
├─ test_images/                 # Sample test images
├─ app.py                       # Streamlit web application
├─ requirements.txt             # Dependencies
└─ README.md                    # Documentation

⚙️ 3️⃣ Setup Instructions
🧩 3.1 Clone the Repository
git clone https://github.com/Clivez12/orange_classification.git
cd orange_classification
🧩 3.2 Create and Activate a Virtual Environment
Windows:
python -m venv .venv
.venv\Scripts\activate
Linux/Mac:
python3 -m venv venv
source venv/bin/activate
🧩 3.3 Install Dependencies
pip install -r requirements.txt
🧩 3.4 Download Pretrained Model
If the model file (best_model.pth) is not included due to size, download it separately and place it inside:
saved_models/

🚀 4️⃣ Usage
🧠 Train the Model
python src/train_model.py
•	Trains the CNN model
•	Saves best weights and history to results/
📊 Evaluate the Model
python src/evaluate.py
•	Evaluates model accuracy & loss
•	Plots training/validation curves
•	Exports results to CSV
🔍 Predict via Command Line
python src/predict.py
•	Predicts orange variety for given image(s)
•	Saves results to results/predictions.csv
🌐 Run Streamlit App
streamlit run app.py
•	Launches a web app to upload and classify images interactively

🧾 5️⃣ Dataset
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
•	Only a sample dataset is provided for demonstration.
•	The full dataset can be shared privately upon request.
•	Folder names must match class labels exactly.

📜 6️⃣ License
This project is developed for academic and research purposes.
For commercial or extended use, please contact the author.

👨‍💻 7️⃣ Author & Contact
Author: Terna Henry Wua
Email: henryternawua@gmail.com
GitHub: https://github.com/Clivez12

