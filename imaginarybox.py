import cv2
import numpy as np

def draw_bounding_box_and_inscribed_circle(image, bbox, box_color=(255, 0, 0), circle_color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box and the largest possible inscribed circle within it.

    Args:
        image: The image (NumPy array).
        bbox: A tuple or list (x1, y1, x2, y2) representing the bounding box.
        box_color: The color of the bounding box (BGR format).
        circle_color: The color of the circle (BGR format).
        thickness: The thickness of the lines.
    """
    x1, y1, x2, y2 = bbox

    # 1. Draw the Bounding Box
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)

    # 2. Calculate Width and Height
    width = x2 - x1
    height = y2 - y1

    # 3. Calculate the Radius
    radius = int(min(width / 2, height / 2))  # Convert to integer

    # 4. Find the Circle's Center
    center_x = int((x1 + x2) / 2)  # Convert to integer
    center_y = int((y1 + y2) / 2)  # Convert to integer
    center = (center_x, center_y)

    # 5. Draw the Circle
    cv2.circle(image, center, radius, circle_color, thickness)

    return image


# Example Usage:
if __name__ == '__main__':
    # Create a sample image (replace with your actual image loading)
    img = np.zeros((300, 400, 3), dtype=np.uint8)  # Black image

    # Example Bounding Boxes (replace with your detection data)
    bounding_boxes = [
        (50, 50, 150, 150),
        (200, 75, 280, 150),
        (50, 200, 120, 240)
    ]

    for bbox in bounding_boxes:
        img = draw_bounding_box_and_inscribed_circle(img, bbox)

    # Display the image (or save it)
    cv2.imshow("Image with Bounding Boxes and Circles", img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()