import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron
import time



write_video=True
fps = 5
resolution=(640,640)


#  For webcam input:


cap = cv2.VideoCapture("14.MOV")
video_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, resolution)

with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.99,
                            model_name='Shoe') as objectron:



 while cap.isOpened():
    success, image = cap.read()
    image=cv2.resize(image , (640,640))
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      # continue
      break


    start = time.time()
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = objectron.process(image)
    # print("results", results) 

    if not results.detected_objects:
      print(f'No box landmarks detected')
     



    # Draw the box landmarks on the image.
    image.flags.writeable = True
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detected_objects:
        print("Detection Done")
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(
              image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(image, detected_object.rotation,
                                 detected_object.translation)


    end_time= time.time()  
    print("time: {}s, fps: {}".format(end_time - start, 1 / (end_time - start)))
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Objectron', cv2.flip(image, 0))

    image=cv2.flip(image, 0)
    video_writer.write(image)


   
    # fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    # # if write_video:
    # vout = cv2.VideoWriter()
    # vout.open('output_video_demo.avi',fourcc,fps,resolution,True)
    # vout.write(image)

    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
# vout.release()
