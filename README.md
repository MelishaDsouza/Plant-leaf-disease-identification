# Plant-leaf-disease-identification
Plant-leaf-disease-identification using CNN ,deep Learning and flask

# plant-leaf-Disease-Identification-using-CNN
* **Problem statement** :The project ”Plant Leaf Disease Identification Using Image Recognition” aims to develop a system that automates the detection and classification of diseases in plant leaves. Traditionally, this task involves labor-intensive and error-prone manual inspections. The primary goal is to create an efficient, accurate tool using deep learning techniques to iden- tify and categorize leaf diseases quickly, reducing the need for manual intervention. The dataset used for this project includes images of healthy and diseased leaves from tomato, potato, and pepper bell plants. This project seeks to improve early disease detection and management in agriculture by leveraging advanced image recognition technologies.

* **Dataset** :Download a dataset from kaggle. Here, I have taken a dataset that contains 15 classes of 3 types of plants. Also, the dataset was split into train and test dataset in the code app.iynb

* **Functional requirements** :
• Image Processing and Analysis:Use OpenCV and NumPy for handling and analyz- ing plant leaf images.
• Model Development and Training: Utilize TensorFlow and Keras for training CNN models to classify leaf diseases.
• Interactive Development Environment:VS Code for coding and debugging.
• Web Interface: Develop a user-friendly web interface with Flask for image uploads
and result displays.

* **Software Requirements** :
• Integrated Development Environment (IDE):
  Visual Studio Code: Used for coding, debugging, and running Python scripts.
• Web Development :
  1. HTML/CSS: For creating the web pages and styling them.
  2. JavaScript: For handling client-side interactions and image preview functionality.
• Database :
  1. File Storage: To store uploaded images and possibly results.
  2. Directory Path: Specify the paths used for uploading and storing images (e.g., /Users/melisha/Desktop/6th sem project/uploads).
• Operating System :
  MacOS or Windows: As your development and deployment environment.

* **Libraries and Frameworks** :
• TensorFlow: For building and training the Convolutional Neural Network (CNN). Version: 2.x
• OpenCV: For image processing tasks. Version: 4.x
• NumPy: For numerical operations and array manipulations.
• Pandas: For data manipulation and analysis.
• Scikit-learn: For data preprocessing and splitting.
• Matplotlib: For plotting and visualizing images and results.
• Flask: For creating the web application to upload and display results. Version: 2.x


* **Data Flow Design** :
• User Uploads Image :
   Input: User selects and uploads an image through the web interface.
• Front-End Processing :
  Image Preview: Process: JavaScript previews the image on the client side.
• Back-End Processing Receive Image:
  Process: Flask server receives the uploaded image file. Output: Saved image file path.
* Preprocess Image :
  Input: Saved image file path. Process: Resize and normalize the image. Output: Preprocessed image. Load Model:
  Input: Path to the pre-trained model. Process: Load model into memory.
• Model Inference :
  Predict Disease: Input: Preprocessed image. Process: Model performs inference to classify the image. Output: Predicted class and confidence score.
• Result Presentation Generate Result Page :
  Input: Prediction results, image URL. Process: Create HTML page with results. Output: Result page with prediction and image. Display Results:
  Input: Result page URL. Process: Display result page to the user.

<img width="1440" alt="index" src="https://github.com/user-attachments/assets/a4d3efa4-034a-493c-b01e-bd64b3803ea1">

<img width="1436" alt="Screenshot 2024-07-30 at 6 44 47 PM" src="https://github.com/user-attachments/assets/c2d8ebeb-5d60-4404-b752-f48e25e5c6cf">

<img width="1405" alt="Screenshot 2024-07-30 at 6 26 21 PM" src="https://github.com/user-attachments/assets/0abd559c-af9b-42a0-b795-cf979e52aff7">
