
x1, y1, x2, y2 =  20, 50, 620, 400 # Region of Interest

image = cv2.imread("frame_bottle.png")
image = image[y1:y2, x1:x2]
area = calculate_red_area(image)

print(f"Red area: {area} pixels")

# visualize_red_detection("frame_bottle.png")
visualize_red_detection(image)