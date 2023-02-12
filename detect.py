import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# For static images:
file1="chair\chair.jpg"
file2="chair\chair2.jpg"
file3="chair\chair3.jpg"
# file1="3.jpg"
# file2="malvestida-DMl5gG0yWWY-unsplash.jpg"
# file3="revolt-164_6wVEHfI-unsplash.jpg"

IMAGE_FILES = [file1,file3]
with mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            model_name='Chair') as objectron:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # cv2.imshow("img" ,image )
    # cv2.waitKey(0)

    image=cv2.resize(image,(640,640))
    # Convert the BGR image to RGB and process it with MediaPipe Objectron.
    results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print("results", results) 

    # Draw box landmarks.
    if not results.detected_objects:
      print(f'No box landmarks detected on {file}')
      # continue
    print(f'Box landmarks of {file}:')
    annotated_image = image.copy()
    for detected_object in results.detected_objects:
      mp_drawing.draw_landmarks(
          annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
      mp_drawing.draw_axis(annotated_image, detected_object.rotation,
                            detected_object.translation)
      cv2.imshow("img", annotated_image)
      
      cv2.imwrite('results/' + str(idx) + '.png', annotated_image)
  




# For webcam input:
# cap = cv2.VideoCapture("example1.MOV")
# with mp_objectron.Objectron(static_image_mode=False,
#                             max_num_objects=5,
#                             min_detection_confidence=0.5,
#                             min_tracking_confidence=0.99,
#                             model_name='Shoe') as objectron:

#   while cap.isOpened():
#     success, image = cap.read()
#     image=cv2.resize(image , (640,640))
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       # continue
#       break

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = objectron.process(image)
#     # print("results", results) 

#     # if not results.detected_objects:
#       # print(f'No box landmarks detected')
     

#     # Draw the box landmarks on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.detected_objects:
#         print("Detection Done")
#         for detected_object in results.detected_objects:
#             mp_drawing.draw_landmarks(
#               image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
#             mp_drawing.draw_axis(image, detected_object.rotation,
#                                  detected_object.translation)
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Objectron', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()
