# Camera Sabotage detection

- The big task is divided into smaller components, including:
  - Occlusion detection (camera is covered, or an object is placed close to the len)
  - Displacement detection (a static camera's view is unintentionally changed)
  - Defocusing (Blur, unclear vision due to rain, snow, dust, etc.)

## Occlusion detection:
Approach of the authors in the [paper](./papers/(2012)Sabotage.pdf)

- The author compare ratio of current-frame-entropy and background-entropy with a threshold to determine tampering.
- Additional BHATTACHARYYA distance of current-frame-histogram and background-histogram is computed and compared with a threshold to ensure the tampering.
- Testing code:
  ```
  cd code/
  python occlusion.py --input ../clips/Occlusion.mp4 --algo MOG2 --scale 0.5 --threshold 0.4
  ```


## Displacement detection:
Approach of authors in the [paper](./papers/(2012)Sabotage.pdf) is not robust with moving scence (crowd scence)


## Defocusing detection:
Approach of authors in the [paper](./papers/(2007)Automatic_Control_of_Video_Surveillance_Camera_Sab.pdf)

- The author defines a background model based on Edge Detection.
- The Camera's Focus is tampered when the ratio of number of current-frame-edge to back-ground-edge is decreased significantly.
- **Manual threshold** must be chosen to trigger the alarm.
- Testing code:
  ```
  cd code/
  python defocus edges.py --input ../clips/Defocusing.mp4 --scale 0.5 -t 0.5
  ```

|Strength | Weakness|
|---------|---------|
|Fast, Reasonable approach (using number of edges) | Not adaptive threshold |
