import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Define supermarket items + prices
PRICE_LIST = {
    "apple": 30,
    "banana": 10,
    "orange": 25,
    "bottle": 50,
    "cup": 40,
    "sandwich": 50,
    "pizza": 80,
    "donut": 35,
    "cake": 100,
}

# Helper: check if two boxes are close
def is_same_object(box1, box2, threshold=50):
    x1, y1, x2, y2 = box1
    cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2  # center
    x1b, y1b, x2b, y2b = box2
    cx2, cy2 = (x1b + x2b) / 2, (y1b + y2b) / 2
    return abs(cx1 - cx2) < threshold and abs(cy1 - cy2) < threshold

def run_cashier(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    counts = defaultdict(int)
    seen_objects = []  # store (class, box)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                if cls_name in PRICE_LIST:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    new_box = (x1, y1, x2, y2)

                    # Check if this object is already seen
                    already_seen = False
                    for prev_cls, prev_box in seen_objects:
                        if prev_cls == cls_name and is_same_object(prev_box, new_box):
                            already_seen = True
                            break

                    if not already_seen:
                        counts[cls_name] += 1
                        seen_objects.append((cls_name, new_box))

                    # Draw box + label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, cls_name, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show running counts
        y = 30
        for k, v in counts.items():
            cv2.putText(frame, f"{k}: {v}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y += 30

        cv2.imshow("Supermarket Cashier", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print Final Bill
    subtotal = 0
    print("\n===== RECEIPT =====")
    for item, count in counts.items():
        price = PRICE_LIST[item]
        line_total = price * count
        subtotal += line_total
        print(f"{item} x{count} @ {price:.2f} = {line_total:.2f}")
    print(f"TOTAL: {subtotal:.2f}")
    print("===================\n")

# Run
run_cashier()
