import cv2
import os
import numpy as np
import json
os.makedirs("faces", exist_ok=True)
if os.path.exists("users.json"):
    with open("users.json","r") as file:
        users=json.load(file)
else:
    users = {}
recognizer = cv2.face.LBPHFaceRecognizer_create()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


training_faces = []
training_labels = []
for filename in os.listdir("faces"):

    if filename.endswith(".jpg"):

        # Get the person's name from the filename.
        name = filename[:-4]

        # Find their ID from users.json.
        for user_id, user_name in users.items():

            if user_name == name:

                # Load their saved face.
                image = cv2.imread(
                    os.path.join("faces", filename),
                    cv2.IMREAD_GRAYSCALE
                )

                # Add the face and its ID to the training data.
                training_faces.append(image)
                training_labels.append(int(user_id))

                break
if len(training_faces)>0:
    recognizer.train(training_faces, np.array(training_labels))
    print("Face recog trained")
else:
    print("No faces found")
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the camera.")
    exit()
print("Camera Running")

print("Press Q to exit.")
# Keep track of whether the recognizer was actually trained.
model_trained = len(training_faces) > 0
while True:
    success, frame = camera.read()
    if not success:
        print("Error: Camera frame could not be read.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=5, minSize=(80,80))
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)

        # Crop the detected face from the camera image.
        face = gray[y:y+h, x:x+w]

        # Ask the trained recognizer who this face belongs to.
       # Only try to recognize faces if we actually trained the model.
        if model_trained:

            # Ask the trained recognizer who this face belongs to.
            label, confidence = recognizer.predict(face)

            if confidence < 80:

                # Convert the ID returned by the recognizer into a name.
                name = users[str(label)]

                text = f"Welcome, {name}!"

            else:

                text = "Unknown face"

        else:

            # The program can still detect faces,
            # but it doesn't have anyone registered yet.
            text = "No users registered"

        # Display the result above the face.
        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )
        cv2.imshow("Facial Login - Face Detection - Made By Abdalla Ali", frame)
        

    key = cv2.waitKey(1)&0xFF
    if key == ord("r"):
        if len(faces)>0:
            name = input("Enter name: ")

# Give this person a new ID.
            new_id = str(len(users) + 1)

            # Save the person's name with their ID.
            users[new_id] = name

            # Save the updated user list.
            with open("users.json", "w") as file:
                json.dump(users, file, indent=4)

            (x,y,w,h) = faces[0]
            face = gray[y:y+h, x:x+w]

            filename = f"faces/{name}.jpg"
            cv2.imwrite(filename, face)

            print(f"Face registered for {name}!")
        else:       
            print("No face detected.")

camera.release()

cv2.destroyAllWindows()

        