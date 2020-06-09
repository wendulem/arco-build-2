from flask import Flask, request
from flask_cors import CORS, cross_origin
from imageai.Detection.Custom import CustomObjectDetection
from urllib.request import urlopen
import os
from google.cloud import vision
import cv2
import numpy as np

app = Flask(__name__)
cors = CORS(app)

app.debug = True

# Loading model (how does it scale to others?)
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
# Assuming model is in base folder
detector.setModelPath("detection_model-ex-023--loss-0006.246.h5")
# Assuming config is in base folder
detector.setJsonPath("detection_config.json")
detector.loadModel()

# Main ML Server Endpoint
@app.route('/', endpoint='image_processing', methods=['POST'])
@cross_origin()
def ml_test():
    # Parsing request body for image URL
    url = request.json['image_link']

    # Reads image from URL (scale this and inference to multiple images)
    resp = urlopen(url)
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Performs inference on input image
    detections = detector.detectObjectsFromImage(
        input_image=img, input_type="array", output_type="array")[1]  # use output_image to save inference

    # Gets bounding box dimensions (how to do all these w one model?)
    bounding_boxes = dict()
    for detection in detections:
        # detection outputs array of 4 values of where the box is[topleft x, topleft y, bottomright x, bottomrigh# t y]
        print(detection["name"], " : ", detection["percentage_probability"],
              " : ", detection["box_points"])
        bounding_boxes[detection["name"]] = detection["box_points"]

    # Gets bounding box keys
    box_keys = list(bounding_boxes)
    # Gets cropped images for different labels
    cropped_boxes = dict()
    for key in box_keys:
        bounding_box = bounding_boxes[key]
        print(bounding_box)

        x = bounding_box[0]
        y = bounding_box[1]
        w = bounding_box[2] - bounding_box[0]
        h = bounding_box[3] - bounding_box[1]

        # Gets subimage from bounding box
        subimage = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cropped = img[y:y+h, x:x+w]  # this needs editing for multiple images

        cropped_boxes[key] = cropped
        # Writes and displays images for testing
        # cv2.imwrite("thumbnail.png", cropped)
        # cv2.imshow("cropped", cropped)
        # cv2.waitKey(0)

    # Authenticates and loads vision client
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'arco-test-build-66c309b745cc .json'
    client = vision.ImageAnnotatorClient()

    # Return text log
    log = []

    # Performs OCR on bounding box subimages
    cropped_keys = list(cropped_boxes)
    for key in cropped_keys:
        # Converts image in memory to PNG (Better way of doing this?)
        image = vision.types.Image(content=cv2.imencode(
            '.png', cropped_boxes[key])[1].tobytes())

        # Google Vision API OCR
        response = client.document_text_detection(image=image)

        # Temporary logging while still building; adding words to log
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                print('\nBlock confidence: {}\n'.format(block.confidence))

                for paragraph in block.paragraphs:
                    print('Paragraph confidence: {}'.format(
                        paragraph.confidence))

                    for word in paragraph.words:
                        word_text = ''.join([
                            symbol.text for symbol in word.symbols
                        ])
                        print('Word text: {} (confidence: {})'.format(
                            word_text, word.confidence))

                        log.append(word_text)

                        for symbol in word.symbols:
                            print('\tSymbol: {} (confidence: {})'.format(
                                symbol.text, symbol.confidence))

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))

        # Crops and saves images for testing
        # region = image.crop(143,1495,676,1778)
        # region.save("hello_world.png")

    # Returns string of all words detected
    return (' '.join(map(str, log)))


if __name__ == '__main__':
    app.run()
