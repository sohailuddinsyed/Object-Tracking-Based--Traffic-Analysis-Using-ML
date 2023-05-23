from algorithm.object_detector import YOLOv7
from utils.detections import draw
from tqdm import tqdm
import numpy as np
import cv2

yolov7 = YOLOv7()
yolov7.load('coco.weights', classes='coco.yaml', device='cpu') 

video = cv2.VideoCapture('video.mp4')
width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

if video.isOpened() == False:
    print('[!] error opening the video')

print('[+] tracking video...\n')
pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)
lines = {}
arrow_lines = []
arrow_line_length = 50

vehicle_count = 0
counted_vehicles = set()

try:
    while video.isOpened():
        ret, frame = video.read()
        if ret == True:
            detections = yolov7.detect(frame, track=True)
            detected_frame = frame
            
            for detection in detections:
                color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                
                if 'id' in detection:
                    detection_id = detection['id']

                    if detection_id not in lines:
                        detection['color'] = color
                        lines[detection_id] = {'points':[], 'arrows':[], 'color':color}
                    else:
                        detection['color'] = lines[detection_id]['color']
                    
                    lines[detection_id]['points'].append(np.array([detection['x'] + detection['width']/2, detection['y'] + detection['height']/2], np.int32))
                    points = lines[detection_id]['points']

                    if len(points) >= 2:
                        arrow_lines = lines[detection_id]['arrows']
                        if len(arrow_lines) > 0:
                            distance = np.linalg.norm(points[-1] - arrow_lines[-1]['end'])
                            if distance >= arrow_line_length:
                                start = np.rint(arrow_lines[-1]['end'] - ((arrow_lines[-1]['end'] - points[-1])/distance)*10).astype(int)
                                arrow_lines.append({'start':start, 'end':points[-1]})
                        else:
                            distance = 0
                            arrow_lines.append({'start':points[-2], 'end':points[-1]})
            
            for line in lines.values():
                arrow_lines = line['arrows']
                for arrow_line in arrow_lines:
                    detected_frame = cv2.arrowedLine(detected_frame, arrow_line['start'], arrow_line['end'], line['color'], 2, line_type=cv2.LINE_AA)

            detected_frame = draw(frame, detections)
            
            for detection in detections:
                if 'id' in detection:
                    detection_id = detection['id']

                    if detection_id not in counted_vehicles:
                        counted_vehicles.add(detection_id)
                        vehicle_count += 1
            
            detected_frame = cv2.putText(detected_frame, f"Vehicle Count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            output.write(detected_frame)
            pbar.update(1)
        else:
            break
except KeyboardInterrupt:
    pass

pbar.close()
video.release()
output.release()
yolov7.unload()
