import cv2, glob

glob_img=glob.glob("*.jpg")

detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

for traversal_img in glob_img:
    img = cv2.imread(traversal_img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face=detect.detectMultiScale(gray_image, 1.20, 5)

    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255,0), 2)

    cv2.imshow("Detect Multiple Images", img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows    

