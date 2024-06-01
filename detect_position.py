import cv2

protoFile = "pose_model/pose_deploy_linevec.prototxt"
weightsFile = "pose_model/pose_iter_440000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


def detect_position(img_path):
    image = cv2.imread(img_path)
    frameHeight, frameWidth = image.shape[:2]

    # Prepare the image for the network
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)

    # Run the forward pass
    output = net.forward()

    # Find the points
    points = []
    nPoints = 18
    for i in range(nPoints):
        # Confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > 0.1:  # If the probability is greater than threshold
            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)

    if points[8] and points[11]:
        if points[8][0] < points[11][0]:
            return 1
        else:
            return 0

    return -1
