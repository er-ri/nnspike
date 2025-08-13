# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CLAHE+OTSU: Merge contours with close x-coordinates (Obstacle grouping & visualization) ---
img_path = r'C:/Users/MSAD/github/nnspike/storage/20250811/frames/20250811154714/frame_1211.png'

        max_x = min_x + w

# main関数化
def main():
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    img_path = r'C:/Users/MSAD/github/nnspike/storage/20250811/frames/20250811154714/frame_1211.png'
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {img_path}')

    x1, y1, x2, y2 = 100, 150, 540, 330
    roi_w, roi_h = x2-x1, y2-y1
    img_with_roi = img.copy()
    x_merge_threshold = 150

    img_gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe_full = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe_full = clahe_full.apply(img_gray_full)
    _, mask_full = cv2.threshold(img_clahe_full, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = mask_full[y1:y2, x1:x2]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 10
    filtered_contours = []
    rect_centers_x = []
    rect_centers_y = []
    rects = []
    for cnt in contours:

        def main():
            import cv2
            import numpy as np
            import matplotlib.pyplot as plt

            img_path = r'C:/Users/MSAD/github/nnspike/storage/20250811/frames/20250811154714/frame_1211.png'
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f'Image not found: {img_path}')

            x1, y1, x2, y2 = 100, 150, 540, 330
            roi_w, roi_h = x2-x1, y2-y1
            img_with_roi = img.copy()
            x_merge_threshold = 150

            img_gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe_full = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_clahe_full = clahe_full.apply(img_gray_full)
            _, mask_full = cv2.threshold(img_clahe_full, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            mask = mask_full[y1:y2, x1:x2]

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 10
            filtered_contours = []
            rect_centers_x = []
            rect_centers_y = []
            rects = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if area < min_area:
                    continue
                cx = x + w // 2
                cy = y + h // 2
                filtered_contours.append(cnt)
                rect_centers_x.append(cx)
                rect_centers_y.append(cy)
                rects.append((x, y, w, h))

            # --- Merge rects by x coordinate ---
            if rects:
                merged_rects = []
                rects_sorted = sorted(rects, key=lambda r: r[0])
                current = list(rects_sorted[0])
                for r in rects_sorted[1:]:
                    if abs(r[0] - current[0]) < x_merge_threshold:
                        min_x = min(current[0], r[0])
                        min_y = min(current[1], r[1])
                        max_x = max(current[0]+current[2], r[0]+r[2])
                        max_y = max(current[1]+current[3], r[1]+r[3])
                        current = [min_x, min_y, max_x-min_x, max_y-min_y]
                    else:
                        merged_rects.append(tuple(current))
                        current = list(r)
                merged_rects.append(tuple(current))
            else:
                merged_rects = []

            # 物体がなかった場合はx=320のみprintして終了
            if not merged_rects:
                print('No obstacle detected. x=320')
                return

            # ここから下はmerged_rectsが空でない場合のみ実行される
            min_x = merged_rects[0][0]
            min_y = merged_rects[0][1]
            w = merged_rects[0][2]
            h = merged_rects[0][3]
            max_x = min_x + w
            max_y = min_y + h
            left_edge_x = x1 + min_x
            right_edge_x = x1 + max_x
            group_center_x = x1 + min_x + w // 2
            group_center_y = y1 + min_y + h // 2
            if group_center_x < 320:
                edge_x = right_edge_x
                edge_label = 'Right edge'
                edge_dist = abs(right_edge_x - 320)
                target_x = edge_x + 150
                cv2.line(img_with_roi, (edge_x, group_center_y), (320, group_center_y), (0,255,255), 3)
            else:
                edge_x = left_edge_x
                edge_label = 'Left edge'
                edge_dist = abs(left_edge_x - 320)
                target_x = edge_x - 150
                cv2.line(img_with_roi, (edge_x, group_center_y), (320, group_center_y), (0,255,255), 3)
            cv2.circle(img_with_roi, (target_x, 330), 10, (0,255,0), -1)
            cv2.putText(img_with_roi, f"{target_x}", (target_x-25, 330-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.line(img_with_roi, (edge_x, y1), (edge_x, y2), (0,255,255), 2)
            cv2.putText(img_with_roi, f"{edge_dist}", ((edge_x+320)//2, group_center_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)
            cv2.circle(img_with_roi, (group_center_x, group_center_y), 10, (255,0,0), -1)
            cv2.putText(img_with_roi, f"G1({group_center_x},{group_center_y})", (group_center_x-40, group_center_y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            print(f'Group 1 center: x={group_center_x}, y={group_center_y}')
            print(f'--- Distance from group 1 {edge_label} to x=320 center line ---')
            print(f'{edge_label} x={edge_x}, distance={edge_dist}')
            cv2.rectangle(img_with_roi, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.line(img_with_roi, (320, y1), (320, y2), (0,255,255), 2)

            # Draw center of each obstacle in ROI (red, image coordinates, bounding box center)
            for rect in rects:
                x, y, w, h = rect
                abs_cx = x1 + x + w // 2
                abs_cy = y1 + y + h // 2
                cv2.circle(img_with_roi, (abs_cx, abs_cy), 7, (0,0,255), -1)
                cv2.putText(img_with_roi, f"{abs_cx}", (abs_cx-15, abs_cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # ...existing code...

        if __name__ == '__main__':
            main()
else:
    print('No merge pairs')

 # Print distance from group 1 edge (left or right) to x=320 center line
print(f'--- Distance from group 1 {edge_label} to x=320 center line ---')
print(f'{edge_label} x={edge_x}, distance={edge_dist}')