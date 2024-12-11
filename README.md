# **CLIP-Based Indian Bird Species Classifier**

This project leverages OpenAI's CLIP (Contrastive Language–Image Pre-training) model, specifically its vision encoder, to classify 25 bird species found in India. It fine-tunes the CLIP Vision Transformer (ViT) and adapts it for image classification tasks.

---

## **Project Overview**

This project involves:
- Adapting CLIP's vision encoder (`ViT-B/32`) for classification.
- Fine-tuning the model on a dataset of 37,000 images of 25 bird species.
- Implementing a two-layer classification head for improved performance.
- Evaluating the model using metrics like accuracy and confusion matrix.

---

## **Dataset**

### **Indian Bird Species Dataset**
- **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/ichhadhari/indian-birds/data)
- **Description**: Contains images of 25 bird species, split into training (30,000 images) and validation (7,500 images) sets.
- **Classes**: Examples include Asian Green Bee-eater, Common Kingfisher, Indian Peacock, and more.

---

## **Model Architecture**

The project uses the **CLIP ViT-B/32** model. The classification head is implemented as follows:
1. The CLIP vision encoder's output (512-dimensional features) is passed through:
   - A fully connected layer that reduces the dimension to 256.
   - A dropout layer for regularization.
   - Another fully connected layer that maps the features to 25 classes.

---

## **Folder Structure**

Here is the folder structure of the project:

```plaintext
CLIP-Bird-Classifier/
├── configs/
│   └── default.yaml          # Configuration file for training
├── datasets/
│   └── __init__.py           # PyTorch Dataset class for loading bird images
├── models/
│   └── __init__.py           # Model definition for CLIP-based classifier
├── scripts/
│   ├── eval.sh               # Script to run evaluation
│   ├── install.sh            # Script to set up the virtual environment
│   └── train.sh              # Script to run training
├── utils/
│   ├── __init__.py           # Utility module
│   └── build_logger.py       # Logger setup
├── cfg.py                    # Base configuration script
├── eval.py                   # Evaluation script
├── train.py                  # Training script
├── README.md                 # Project documentation
├── requirements.txt          # Required Python packages
├── outputs/                  # Directory for saved model weights
│   └── clip_classifier.pth   # Trained model weights (created after training)
└── data/                     # Dataset directory (not included in the repo)
    ├── train/                # Training images (subfolders for each class)
    └── val/                  # Validation images (subfolders for each class)
```

## **Setup and Installation**

### **Prerequisites**
- Python 3.8 or higher
- Conda or virtual environment

### **Installation Steps**

1. Create and activate the environment:
    ```bash
    conda create --name clip_env python=3.9 -y
    conda activate clip_env
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Prepare the dataset:

    Download the dataset from Kaggle(https://www.kaggle.com/datasets/ichhadhari/indian-birds/data).
    Update the dataset path in the config file.

## **Training**

To train the model, run the following command:

```bash
python train.py --config ./configs/default.yaml
```

### **During Training**
- Training and validation progress is displayed after each epoch.
- The model's weights are saved to `outputs/clip_classifier.pth`.

---

## **Evaluation**

To evaluate the model on the validation set, use:

```bash
python eval.py --config ./configs/default.yaml --weights ./outputs/clip_classifier.pth
```


## **Acknowledgments**

- OpenAI for the [CLIP model](https://github.com/openai/CLIP).
- Dataset source: [Kaggle](https://www.kaggle.com/datasets/ichhadhari/indian-birds/data).
- Libraries used: PyTorch, scikit-learn, matplotlib, seaborn, tqdm.

---
