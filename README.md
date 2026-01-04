# LicensePlate_Yolov8_FastPlate-ocr
Car license plate recognition test using fast_plate_ocr
Installation:

Download and extract the project, then extract the .zip files containing the test images Test and Test1.

fast_plate_ocr works with Python 3.10 or higher, so you must install this version.

If using Anaconda, the graphical interface for conda, it's advisable to create a new environment and select a Python version 3.10 or higher from the program dropdown.

Additionally, install:

python pip-script.py install fast-plate-ocr[onnx] (assumes CPU usage)

python pip-script.py install ultralytics

python pip-script.py install imutils

python pip-script.py install scikit-image

Perform the evaluation Running:

python GetNumberInternationalLicensePlate_Yolov8_FastPlate-ocr.py

The results of the processing are displayed on the screen for each image, with a final tally of 10 correct matches out of 12 images. The images are named with the license plate number of the car they contain, allowing verification if the image name matches the detected license plate number.

Observations

The best.pt model, obtained from the project

https://github.com/ablanco1950/DetectSpeedLicensePlate_Yolov8_Filters_PaddleOCR, is used for license plate detection.

Both this project and the current one meet the two conditions: they can be run from a personal computer and they use OCR tools from open sources that do not require an API key.

By changing line 7 in the previously executed evaluation program, replacing "Test1" with "Test" and running it, the result was 12 correct detections out of 25 images. However, running the evaluation performed in the project https://github.com/ablanco1950/DetectSpeedLicensePlate_Yolov8_Filters_PaddleOCR

this would indicate superior detection with paddleocr over fast-plate-ocr.

The errors detected in the detection with fast-plate-ocr consisted of:

Confusion of I with 1
Confusion of O with 0

Errors detecting license plates with a large number of characters.

In contras tusing fast-plate-ocr is faster  the filters used in the project that uses paddleocr are avoided (they do not provide any improvements).

References

https://github.com/ankandrew/fast-plate-ocr

https://github.com/ablanco1950/DetectSpeedLicensePlate_Yolov8_Filters_PaddleOCR
