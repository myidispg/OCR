# Optical Character Recogniton 

When I  started with Machine Learning, one of the most basic examples is the MNIST dataset. Almost every course employed this dataset to teach students about some important concepts successfuly. So, I wanted to use that dataset in a different way to see how much I learnt. And I am proud to say the now I have successfully employed those concepts in a real world usage scenario.

This OCR displays a Tkinter GUI window with a canvas. The user can draw on the canvas and be provided with real time recognition of the drawn characters.

### Demo
![demo gif](https://github.com/myidispg/OCR/blob/master/demo.gif)

### Usage
1. Clone the repository.
2. Execute the **gui_main.py** file.
3. Start writing and the trained model will detect what you have written.

### File details
1. **character_segment.py**- This file is used to segment the word images into constituent characters. 
2. **convert_mnist_format.py**- This is used to convert the preprocessed(binarized and skeletonized) image to MNIST format. The MNIST format is 28x28 pixel images with characters centered in the 20x20 pixel area.
3. **model_train.py**- This loads the data and trains a Convolutional Network on the images. The input is of the format- (1,28,28,1).
4. **old_mnist.py**- This is just a script I used to test all the things before integrating them into the final version.
5. **save_images.py**- I downloaded the EMNIST dataset for this. The train set contains approx 697000 images. The dataset proved to be too much for my laptop's 8GB memory. Any subsequent operation threw a **MemoryError**. But what I could do was atleast load the data. So, I loaded the data and saved all the images to my Hard Disk. While training, I used a batch generator to load the images in batches of 512. This solved my problem of not being able to train due to low memory resource.

##### Attention:
** This OCR works only on single characters or words in which the characters are seperated by spaces. It is not able to detect cursive handwriting yet. I was working on it but I was not able to accurately segment characters with open loops like 'u', 'm', 'w' etc. The segmentation was on point for closed loop characters like 'a', 'd' f', 'o' etc. **
