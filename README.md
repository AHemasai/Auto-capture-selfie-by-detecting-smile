#AUTO-CAPTURE-OF-SELFIE-BY-SMILE-DETECTION
##DESCRIPTION
"Auto Capture Smile Detector" is a machine learning project designed to automate the process of capturing selfies by detecting smiles. Leveraging computer vision algorithms and facial recognition techniques, the system analyzes live camera feeds or static images to identify facial features, particularly the presence of a smile. Once a smile is detected with a certain level of confidence, the system triggers the camera to capture the selfie automatically, eliminating the need for manual input.
This project utilizes machine learning models trained on vast datasets of facial expressions to accurately recognize smiles in real-time. By employing techniques such as convolutional neural networks (CNNs) or deep learning architectures, the system can continuously improve its accuracy and performance over time. The application of this technology extends beyond just capturing selfies, offering potential applications in photography, social media, and user interface interactions where smile detection plays a crucial role. Overall, the "Auto Capture Smile Detector" project showcases the power of machine learning in automating tasks and enhancing user experiences in various domains.
###TEAM MEMBERS DETAILS
Achutha Hemasai-9921004005@klu.ac.in
Ampavatina Manisha-9921004029@klu.ac.in
A.Ashok kumar-9921004830@klu.ac.in
Pasupuleti Kavya sree-9921004543@klu.ac.in
####THE PROBLEM IT SOLVES
"Auto Capture Smile Detector" project addresses several key problems:

Convenience: Traditional selfie-taking methods often require manual triggering of the camera, which can be cumbersome, especially when trying to capture the perfect moment. By automatically detecting smiles and capturing photos, this project eliminates the need for manual intervention, providing users with a more convenient and seamless experience.

Enhanced User Experience: Smiles are often associated with positive emotions and moments worth capturing. By automatically detecting smiles and capturing selfies at the right moment, the project enhances the overall user experience, allowing individuals to effortlessly capture joyful moments without having to worry about operating the camera.

Accessibility: For individuals with mobility impairments or those who struggle with using traditional camera controls, this project provides a more accessible way to capture selfies. By relying on smile detection, it enables a broader range of users to participate in the selfie-taking experience independently.

Efficiency: The automation of the selfie-capturing process saves time and effort for users. Instead of repeatedly attempting to capture the perfect selfie manually, the system efficiently detects smiles and captures photos in real-time, streamlining the overall process.

Improved Photography: Smile detection technology can lead to better-quality photos by ensuring that subjects are smiling and expressing genuine emotions when the picture is taken. This can result in more engaging and visually appealing selfies, enhancing the overall quality of personal and social media photography.

Overall, the "Auto Capture Smile Detector" project solves the problem of manual intervention in the selfie-capturing process, providing users with a more convenient, accessible, and efficient way to capture joyful moments through automated smile detection.
#####USE CASES
Social Media Platforms
Mobile Photography Apps
Event Photography
Healthcare and Wellbeing
######CHALLENGES
Accuracy and Reliability: One of the primary challenges is achieving high accuracy and reliability in smile detection. Variations in lighting conditions, facial expressions, and camera angles can impact the performance of the machine learning model. Ensuring consistent and accurate detection across diverse scenarios requires extensive training data and robust algorithms.
Real-Time Processing: Implementing smile detection in real-time poses a significant technical challenge, especially on resource-constrained devices such as smartphones. The system must process live camera feeds quickly and efficiently to detect smiles in real-time without significant delays or performance issues.
Privacy and Ethical Considerations: Smile detection technology raises privacy concerns, particularly regarding the collection and use of facial data. Developers must implement appropriate privacy measures to protect users' personal information and ensure compliance with data protection regulations.
#######IMAGES OF OUTCOMES OF THE PROJECT
![selfie-2023-12-06-18-08-56](https://github.com/AHemasai/Auto-capture-selfie-by-detecting-smile/assets/126152730/44ada5dc-caa8-4e08-bf1a-836928b71ae6)
![Screenshot 2022-11-12 001337](https://github.com/AHemasai/Auto-capture-selfie-by-detecting-smile/assets/126152730/d4871f98-ed3a-400e-9ff7-07afd2a08017)

#########DEMO VIDEO 


#########EXACT STEPS TO TEST THE PROJECT
*Install any dependencies or libraries required for running the project, such as machine learning frameworks (e.g., OPENCV2,DATE AND TIME) and image processing libraries.
*Verify that the training data covers a wide range of demographics, facial expressions, and lighting conditions to improve the model's robustness.
*Ensure that the camera is positioned correctly and focused on the subject's face to capture clear images.
*Smile naturally in front of the camera to trigger the smile detection algorithm.
*Once your smile is detected, the application should automatically capture a selfie.
*After the selfie is captured, review the resulting image to ensure that it accurately reflects your smile.
*Evaluate the quality of the selfie and the accuracy of the smile detection feature.
*TECHNOLOGIES USED
Machine Learning (ML),Computer Vision,Facial Recognition,Image Processing,Python Programming Language,Camera Integration.
#########SOURCE CODE:
import cv2
import datetime
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:\\Users\\HP\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('C:\\Users\\HP\\Lib\\site-packages\\cv2\\data\\haarcascade_smile.xml')
while (True):
    ret, frame = cap.read()
    original_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        face_roi = frame[y:y + h, x:x + w]
        gray_roi = gray[y:y + h, x:x + w]
        smile = smile_cascade.detectMultiScale(gray_roi, 1.3, 25)
        for x1, y1, w1, h1 in smile:
            cv2.rectangle(face_roi, (x1, y1), (x1+w1, y1+h1),(0, 0, 225), 2)
            time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            file_name = f'selfie-{time_stamp}.png'
            cv2.imwrite(file_name,original_frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
