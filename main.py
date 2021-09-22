#import matplotlib.pylab as plt
import cv2
import numpy as np

video = cv2.VideoCapture('/home/user/Downloads/test2.mp4')

if video.isOpened == False:
    print("Error reading video file")

frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)


result = cv2.VideoWriter('filename.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

# finds the region of interest


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# finds the convert each frames of video into cropped image after performing canny edge detection
def process(image):
    # print(image.shape)
    if image.any() != None:
        height = image.shape[0]
        width = image.shape[1]
        region_of_interest_vertices = [
            (0, height),
            (width / 2, height / 2 + 40),
            (width, height)
        ]
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canny_image = cv2.Canny(gray_image, 120, 120)
        cropped_image = region_of_interest(canny_image,
                                           np.array([region_of_interest_vertices], np.int32), )

        hough_transform(cropped_image)

    return cropped_image


# apply hough trabsform and print the lines

def hough_transform(image):
    lines = cv2.HoughLinesP(image,
                            rho=2,
                            theta=np.pi / 180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)

    print(lines)


def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=6)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


# finds the convert each frames of video into cropped image after performing canny edge detection
def process(image):
    # print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2 + 40),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 120, 120)
    cropped_image = region_of_interest(canny_image,
                                       np.array([region_of_interest_vertices], np.int32), )
    lines = cv2.HoughLinesP(cropped_image,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=40,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=120)
    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines


# input the video and send each frame to function process


cap = cv2.VideoCapture('/home/user/Downloads/test2.mp4')


while cap.isOpened():
    ret, frame = cap.read()
    frame = process(frame)
    result.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(45) & 0xFF == ord('q'):
        break

cap.release()
video.release()
result.release()
cv2.destroyAllWindows()

print("The video was successfully saved")
