# import argparse
import cv2 as cv
import numpy as np


# import sys


# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv.CAP_PROP_POS_MSEC)) < upper


def rescaleFrame(frame, scale=0.75):  # images videos and live video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA), width, height

def main():  # 3input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv.VideoCapture('Videos/CV_ass1.mp4') # CV_ass1_lowest
    ball, temp_width, temp_height = rescaleFrame(cv.imread('Videos/ball_ref.png', 0), 0.5)
    replace = cv.imread('Videos/replace.png')
    fireball, fireball_width, fireball_height = rescaleFrame(cv.imread('Videos/fireball.png'), 0.1)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # saving output video as .mp4
    out = cv.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    lava, lava_width, lava_height = rescaleFrame(cv.imread('Videos/lava.png'), 2)
    lava = lava[0:frame_height, 0:frame_width, :]
    n = 0  # kernel size

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            '''Gray for 4 sec'''
            if between(cap, 0, 1000):
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)  # back to BGR for saving purpose
            if between(cap, 1000, 2000):
                pass
            if between(cap, 2000, 3000):
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)  # back to BGR for saving purpose
            if between(cap, 3000, 3300):
                pass
            if between(cap, 3000, 3700):
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)  # back to BGR for saving purpose
            if between(cap, 3700, 3800):
                pass
            if between(cap, 3800, 4000):
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)  # back to BGR for saving purpose

            '''Gaussian Filtering'''
            if between(cap, 4000, 5000):
                n = 5
                frame = cv.GaussianBlur(frame, (n, n), 0)
                cv.putText(frame, 'Gaussian kernel size: ' + str(n), (30, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (123, 121, 200), 2)
            if between(cap, 5200, 6000):
                n = 9
                frame = cv.GaussianBlur(frame, (n, n), 0)
                cv.putText(frame, 'Gaussian kernel size: ' + str(n), (30, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (123, 121, 200), 2)
            if between(cap, 6200, 7000):
                n = 13
                frame = cv.GaussianBlur(frame, (n, n), 0)
                cv.putText(frame, 'Gaussian kernel size: ' + str(n), (30, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (123, 121, 200), 2)
            if between(cap, 7200, 8000):
                n = 17
                frame = cv.GaussianBlur(frame, (n, n), 0)
                cv.putText(frame, 'Gaussian kernel size: ' + str(n), (30, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (123, 121, 200), 2)

            if between(cap, 4000, 8000):
                cv.putText(frame, 'Gaussian filters add a uniform blur to the whole picture', (100, 400),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (51, 153, 255), 2)

            '''Bilateral Filters'''
            if between(cap, 8200, 9000):
                n = 5
                s = 50
                frame = cv.bilateralFilter(frame, n, s, s)
                cv.putText(frame, 'Bilateral kernel size: ' + str(n), (30, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (123, 121, 200), 2)
                cv.putText(frame, 'Sigma: ' + str(s), (30, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (123, 121, 200), 2)
            if between(cap, 9200, 10000):
                n = 5
                s = 100
                frame = cv.bilateralFilter(frame, n, s, s)
                cv.putText(frame, 'Bilateral kernel size: ' + str(n), (30, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (123, 121, 200), 2)
                cv.putText(frame, 'Sigma: ' + str(s), (30, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (123, 121, 200), 2)
            if between(cap, 10200, 11000):
                n = 15
                s = 50
                frame = cv.bilateralFilter(frame, n, s, s)
                cv.putText(frame, 'Bilateral kernel size: ' + str(n), (30, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (123, 121, 200), 2)
                cv.putText(frame, 'Sigma: ' + str(s), (30, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (123, 121, 200), 2)
            if between(cap, 11200, 12000):
                n = 15
                s = 100
                frame = cv.bilateralFilter(frame, n, s, s)
                cv.putText(frame, 'Bilateral kernel size: ' + str(n), (30, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (123, 121, 200), 2)
                cv.putText(frame, 'Sigma: ' + str(s), (30, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (123, 121, 200), 2)
            if between(cap, 8000, 12000):
                cv.putText(frame, 'Whereas, Bilateral filters only apply a gaussian filter', (100, 400),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (51, 153, 255), 2)
                cv.putText(frame, ' if the pixel is similar to its neighbors', (100, 425), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (51, 153, 255), 2)


            '''Color Spaces'''
            if between(cap, 12000, 13000):
                frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                cv.putText(frame, 'HSV color space', (100, 400), cv.FONT_HERSHEY_SIMPLEX, 0.7, (51, 153, 255), 2)

            if between(cap, 13000, 14000):
                frame = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
                cv.putText(frame, 'LAB color space', (100, 400), cv.FONT_HERSHEY_SIMPLEX, 0.7, (51, 153, 255), 2)

            '''Grabbing the object in HSV'''
            if between(cap, 14000, 20000):
                frame = cv.GaussianBlur(frame, (29, 29), 10)  # a lot of blurring cause the flor has a lot of variations
                frame = cv.bilateralFilter(frame, 20, 120, 120)  # both types of blur just to save the computation power that the bilateral needs
                frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                h, s, v = cv.split(frame)
                threshold, h = cv.threshold(h, 179, 255, cv.THRESH_BINARY)  # 229
                threshold, s = cv.threshold(s, 177, 255, cv.THRESH_BINARY)  # 158
                threshold, v = cv.threshold(v, 198, 255, cv.THRESH_BINARY)  # 176
                frame = cv.merge([h, s, v])
                frame = cv.cvtColor(frame, cv.COLOR_HSV2BGR)  # making it a binary image
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                threshold, frame = cv.threshold(frame, 100, 255, cv.THRESH_BINARY)
                # frame=h
                if between(cap, 16000, 18000):
                    morph_open = cv.morphologyEx(frame, cv.MORPH_OPEN, np.ones((60, 60), np.uint8))
                    morph_close = cv.morphologyEx(morph_open, cv.MORPH_CLOSE, np.ones((120, 120), np.uint8))
                    morph = cv.cvtColor(morph_close, cv.COLOR_GRAY2BGR)
                    morph[np.where((morph == [255, 255, 255]).all(axis=2))] = [0, 0, 255]  # changes in other color
                    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
                    frame = cv.bitwise_or(frame, morph)
                    cv.putText(frame, 'Adding Opening and Closing (not much of a change)', (100, 425),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (51, 153, 255), 2)
                if between(cap, 18000, 20000):
                    dilate = cv.dilate(frame, np.ones((5, 5), np.uint8), iterations=1)
                    dilate = cv.cvtColor(dilate, cv.COLOR_GRAY2BGR)
                    dilate[np.where((dilate == [255, 255, 255]).all(axis=2))] = [0, 0, 255]  # changes in other color
                    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
                    frame = cv.bitwise_or(frame, dilate)
                    cv.putText(frame, 'Adding Dilation', (100, 425), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                               (51, 153, 255), 2)
                cv.putText(frame, 'Chose the HSV space', (100, 400), cv.FONT_HERSHEY_SIMPLEX, 0.7, (51, 153, 255), 2)

            '''Sobel edge detector'''
            if between(cap, 20000, 25000):
                frame = cv.bilateralFilter(frame, 15, 100, 100)
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                if between(cap, 20000, 21500):
                    ksize = 1
                if between(cap, 21500, 23000):
                    ksize = 3
                if between(cap, 23000, 24000):
                    ksize = 5
                if between(cap, 24000, 25000):
                    ksize = 7
                scale = 1  # default
                delta = 0  # default
                sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
                sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

                # combined_sobel = cv.bitwise_or(sobelx, sobely)

                blank = np.zeros(sobelx.shape, dtype='float64')
                frame = cv.merge([sobelx, blank, sobely]).astype(np.uint8)
                # gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
                # frame = cv.bitwise_or(frame, gray)
                cv.putText(frame, 'Sobel x ', (30, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (255, 0, 0), 2)
                cv.putText(frame, 'Sobel y ', (30, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 0, 255), 2)
                cv.putText(frame, 'Kernel size:  ' + str(ksize), (30, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (120, 100, 230), 2)

            if between(cap, 25000, 35000):

                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                gray = cv.bilateralFilter(gray, 15, 80, 80)
                rows = gray.shape[0]
                minRadius = 17  # best values
                maxRadius = 30
                param1 = 200
                param2 = 30
                ''' changing Radius '''
                if between(cap, 25000, 26000):
                    minRadius = 0
                    maxRadius = 0
                    param1 = 200
                    param2 = 30
                if between(cap, 26000, 27000):
                    minRadius = 0
                    maxRadius = 10
                    param1 = 200
                    param2 = 30
                if between(cap, 27000, 28000):
                    minRadius = 8
                    maxRadius = 20
                    param1 = 200
                    param2 = 30
                if between(cap, 28000, 29000):
                    minRadius = 15
                    maxRadius = 30
                    param1 = 200
                    param2 = 30
                if between(cap, 29000, 30000):
                    minRadius = 40
                    maxRadius = 100
                    param1 = 200
                    param2 = 30
                '''Changing Param 1 & 2'''
                if between(cap, 30000, 31000):
                    minRadius = 17
                    maxRadius = 30
                    param1 = 50
                    param2 = 30
                if between(cap, 31000, 32000):
                    minRadius = 17
                    maxRadius = 30
                    param1 = 100
                    param2 = 30
                if between(cap, 32000, 33000):
                    minRadius = 17
                    maxRadius = 30
                    param1 = 200
                    param2 = 30
                if between(cap, 33000, 34000):
                    minRadius = 17
                    maxRadius = 30
                    param1 = 200
                    param2 = 10
                if between(cap, 34000, 35000):
                    minRadius = 17
                    maxRadius = 30
                    param1 = 200
                    param2 = 50
                circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                          param1=param1, param2=param2,
                                          minRadius=minRadius, maxRadius=maxRadius)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        center = (i[0], i[1])
                        # circle center
                        cv.circle(frame, center, 1, (0, 100, 100), 3)
                        # circle outline
                        radius = i[2]
                        cv.circle(frame, center, radius, (255, 0, 255), 3)
                cv.putText(frame, 'Min Radius: ' + str(minRadius), (30, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (255, 0, 255), 2)
                cv.putText(frame, 'Max Radius: ' + str(maxRadius), (30, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (255, 0, 255), 2)
                cv.putText(frame, 'Upper thresh of Canny: ' + str(param1), (30, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (190, 200, 200), 2)
                cv.putText(frame, 'Thresh for center: ' + str(param2), (30, 125), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (190, 200, 200), 2)
                cv.putText(frame, 'Detecting circles with Hough Transform', (100, 400), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                           (51, 153, 255), 2)

            if between(cap, 35000, 37000):
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                gray = cv.bilateralFilter(gray, 15, 80, 80)
                rows = gray.shape[0]
                minRadius = 17  # best values
                maxRadius = 30
                param1 = 200
                param2 = 30
                circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                          param1=param1, param2=param2,
                                          minRadius=minRadius, maxRadius=maxRadius)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        # draw rectangle around the found object with hough transform
                        radius = i[2]
                        cv.rectangle(frame, (i[0] - radius, i[1] - radius),
                                            (i[0] + radius, i[1] + radius), (0, 0, 255), thickness=2)
            if between(cap, 37000, 40000):
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                blank = np.zeros([frame_height, frame_width])
                # All the 6 methods: ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                #                     'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

                # Apply template Matching
                res = cv.matchTemplate(frame, ball, cv.TM_SQDIFF_NORMED) # tried all 6 methods this seemed to have the best results
                res = float(255) * (res - np.min(res)) / (np.max(res) - np.min(res))
                res = (255-res)
                blank[temp_height // 2 :blank.shape[0] - temp_height // 2,
                      temp_width // 2: blank.shape[1] - temp_width // 2] = res
                blank[frame_height - temp_height:frame_height, frame_width - temp_width:frame_width] = ball
                frame = cv.cvtColor(blank.astype(np.uint8), cv.COLOR_GRAY2BGR)
                cv.putText(frame, 'Using template matching with a template image:', (330, 500), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                                          (51, 153, 255), 2)

            if between(cap, 40000, 43500):
                # kernel = np.array([[-1, -1, -1],
                #                    [-1,  9, -1],
                #                    [-1, -1, -1]])
                kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
                frame = cv.filter2D(frame, ddepth=-1, kernel=kernel)
                cv.putText(frame, 'Sharpening the image using a Filter', (100, 400), cv.FONT_HERSHEY_SIMPLEX,
                           0.7, (51, 153, 255), 2)
            if between(cap, 43500, 47360):
                sub = cv.GaussianBlur(frame, (29, 29), 0)
                frame = cv.addWeighted(frame, 2, sub, -1, 0)
                cv.putText(frame, 'Sharpening the image by subtracting a smoothed version', (100, 400), cv.FONT_HERSHEY_SIMPLEX,
                           0.7, (51, 153, 255), 2)

            if between(cap, 47360, 52860):
                frame2 = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  ##BGR to HSV
                lb = np.array([0, 148, 100])
                ub = np.array([75, 255, 255])
                mask = cv.inRange(frame2, lb, ub)
                # cv.imshow('Mask', mask)
                kernal = np.ones((5, 5), np.uint8)
                opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernal)  ##Morphology
                dilate = cv.dilate(opening, kernal, iterations=2)
                midResult1 = cv.bitwise_and(replace, replace, mask=dilate)
                invert = cv.bitwise_not(dilate, dilate, mask=None)  ##invert the mask
                midResult2 = cv.bitwise_and(frame, frame, mask=invert)
                frame = cv.bitwise_or(midResult2, midResult1)
                cv.putText(frame, 'Holding a Magical invisible Ball!!!', (150, 100), cv.FONT_HERSHEY_SIMPLEX,
                           0.7, (51, 153, 255), 2)
            if between(cap, 52860, 55160):
                # frame2 = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  ##BGR to HSV
                frame = cv.GaussianBlur(frame, (15, 15), 1)
                frame = cv.bilateralFilter(frame, 15, 80, 80)
                lb = np.array([0, 140, 120])
                ub = np.array([165, 255, 245])
                mask = cv.inRange(frame, lb, ub)
                kernal_dil = np.ones((3, 3), np.uint8)
                kernal_open = np.ones((11, 11), np.uint8)
                opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernal_open)  ##Morphology
                dilate = cv.dilate(opening, kernal_dil, iterations=2)
                # cv.imshow('Mask', mask)
                # cv.imshow('dilate', dilate)
                rows = mask.shape[0]
                minRadius = 10  # best values
                maxRadius = 45
                param1 = 210
                param2 = 9
                circles = cv.HoughCircles(dilate, cv.HOUGH_GRADIENT, 1, rows / 8,
                                               param1=param1, param2=param2,
                                               minRadius=minRadius, maxRadius=maxRadius)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        center = (i[0], i[1])
                        cv.circle(frame, center, 1, (0, 100, 100), 3)
                        # ystart = i[0]-(fireball_height //2)
                        # yend = (i[0]+fireball_height//2)
                        # xstart = i[1] - (fireball_width // 2)
                        # xend = (i[1] + fireball_width // 2)
                        '''Try to replace with other object had problems with edges'''
                        # if xstart <= 0 + fireball_width//2 or xend >= frame_width - fireball_width//2 or ystart<=0 + fireball_height//2 or yend >= frame_height - fireball_height//2:
                        #     break
                        # else:
                        # frame[ystart -1: yend, xstart-1: xend] = fireball
                        # cv.imshow('asdfas', fireball)
                        # frame = cvzone.overlayPNG(frame , fireball, [i[0],i[1]])

                        cv.circle(frame, center, 1, (0, 100, 100), 3)
                        # circle outline
                        radius = i[2]
                        cv.circle(frame, center, radius, (255, 0, 255), 3)
                cv.putText(frame, 'Tracking fast moving ball, with both houghCircles and color', (100, 100), cv.FONT_HERSHEY_SIMPLEX,
                                        0.7, (51, 153, 255), 2)

            if between(cap, 55160, 59970):
                lb = np.array([0, 110, 90])
                ub = np.array([90, 240, 240])
                mask = cv.inRange(frame, lb, ub)
                kernal_dil = np.ones((5, 5), np.uint8)
                kernal_open = np.ones((5, 5), np.uint8)
                opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernal_open)  ##Morphology
                dilate = cv.dilate(opening, kernal_dil, iterations=2)
                cv.imshow('test', dilate)
                rows = mask.shape[0]
                minRadius = 5  # best values
                maxRadius = 20
                param1 = 100
                param2 = 7
                circles = cv.HoughCircles(dilate, cv.HOUGH_GRADIENT, 1, rows / 8,
                                          param1=param1, param2=param2,
                                          minRadius=minRadius, maxRadius=maxRadius)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        center = (i[0], i[1])
                        # circle outline
                        radius = i[2]
                        blank = np.zeros(frame.shape[:2], dtype='uint8')
                        mask_actual = cv.circle(blank, center, radius, 255, -1)
                        inv_mask = (255 - mask_actual)
                        masked_lava = cv.bitwise_and(lava, lava, mask=mask_actual)
                        masked = cv.bitwise_and(frame, frame, mask=inv_mask)
                        frame = cv.bitwise_or(masked_lava, masked)
                cv.putText(frame, 'WOW that dude is juggling LAVA!!!', (100, 450), cv.FONT_HERSHEY_SIMPLEX, 0.7, (51, 153, 255), 2)

            # write frame that you processed to output
            out.write(frame)

            # (optional) display the resulting frame
            cv.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv.waitKey(20) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()