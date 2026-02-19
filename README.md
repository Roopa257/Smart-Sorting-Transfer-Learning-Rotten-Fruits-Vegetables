# Smart-Sorting-fresh-or-rotten-classifier

## Overview  
Smart Sorting is an innovative project focused on enhancing the precision and efficiency of detecting rotten fruits and vegetables using cutting-edge transfer learning techniques. By leveraging the pre-trained **MobileNetV2** deep learning model and adapting it to a specific dataset of fruits and vegetables, this project aims to revolutionize the process of sorting and quality control in the agricultural and food industry.

## Live Demo  
[https://smart-sorting-ai-fresh-or-rotten-classifier-emnpqrbpo2fcshovkb.streamlit.app/]([https://neon-horse-517553.netlify.app](https://smart-sorting-ai-fresh-or-rotten-classifier-emnpqrbpo2fcshovkb.streamlit.app/))

---

## Features
- **99.2% Accuracy:** High-precision classification using MobileNetV2 transfer learning  
- **28 Produce Classes:** Supports detection of 28 different fruits and vegetables  
- **Real-time Processing:** Processes thousands of images per minute  
- **Web Interface:** User-friendly Streamlit web application  
- **Production Ready:** Scalable system designed for industrial deployment  

---

## Folder Structure
- `app.py` : Streamlit application code
- `model/` : Saved model files
- `static/` : Static assets (images, css, etc.)
- `Dataset/` : Training dataset
- `Documents/` : Project documentation


## Installation

### Prerequisites  
- Python 3.7+  
- 8GB+ RAM for model training  
- GPU support (optional, for faster training)  

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd Smart-Sorting-Transfer-Learning
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
## Usage
Run the Streamlit app using:
```
streamlit run app.py
```
Open the given local URL in your browser, upload an image, and view the classification results.

## Dataset
The model is trained on a dataset of fresh and rotten fruits and vegetables images collected from various sources.

## Model
- Model used: Transfer with MobileNetV2
## Supported Produce Classes
The model supports 28 classes of fruits and vegetables:
- Apple (healthy/rotten)
- Banana (healthy/rotten)
- Bell Pepper (healthy/rotten)
- Carrot (healthy/rotten)
- Cucumber (healthy/rotten)
- Grapes (healthy/rotten)
- Lemon (healthy/rotten)
- Mango (healthy/rotten)
- Orange (healthy/rotten)
- Potato (healthy/rotten)
- Strawberry (healthy/rotten)
- Tomato (healthy/rotten)
- Watermelon (healthy/rotten)
- Onion (healthy/rotten)
 ## Docker Deployment

```
FROM python:3.8-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app1.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Make your changes  
4. Add tests if applicable  
5. Submit a pull request  


## License
This project is licensed under the MIT License — see the LICENSE file for details.
## Acknowledgments
- MobileNetV2 architecture by Google  
- Dataset from Kaggle community  
- TensorFlow and Keras frameworks  
- Streamlit for web interface  

## Future Enhancements
- Support for additional produce types  
- Mobile application development  
- IoT device integration  
- Real-time video stream processing  
- Advanced analytics dashboard  
- Multi-language support 
