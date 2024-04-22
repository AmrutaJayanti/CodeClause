import cv2
import os


def capture_gestures(output_path, gesture_label, num_samples):
    
    cap = cv2.VideoCapture(0)

    
    output_dir = os.path.join(output_path, gesture_label)
    os.makedirs(output_dir, exist_ok=True)

    
    sample_count = 0

    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        cv2.imshow('Capture Gestures', frame)

        
        
        if cv2.waitKey(1) & 0xFF == ord(' '):
            
            filename = os.path.join(output_dir, f'{gesture_label}_{sample_count}.jpg')
            cv2.imwrite(filename, frame)
            print(f'Saved: {filename}')

            
            sample_count += 1

            
            if sample_count >= num_samples:
                break

    
    cap.release()
    cv2.destroyAllWindows()

def main():
    
    output_path = 'gestures_dataset'

    
    gestures = ['thumbs_up', 'peace_sign', 'fist', 'open_hand']

    
    num_samples = 15

    
    for gesture in gestures:
        print(f'Capturing {num_samples} samples for {gesture} gesture...')
        capture_gestures(output_path, gesture, num_samples)

    print('Gesture capture complete.')

if __name__ == "__main__":
    main()
