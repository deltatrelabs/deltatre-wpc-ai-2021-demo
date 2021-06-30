# Applied Artificial Intelligence on multimedia content (WPC Days 2021 AI)

In this repository you can find the slides and demo for **Applied Artificial Intelligence on multimedia content** session, presented (in Italian) at [WPC Days 2021 AI](https://www.wpc-days.it/eventi/artificial-intelligence-day/) on June 30th, 2021.

Abstract:

Usually we can see examples of artificial intelligence applied to tabular data, text or images, for classification, recommendation or forecasting purposes.

In this session, instead, we share concepts, findings, and demo from our explorations and projects in Deltatre Innovation Lab, related to Machine Learning and Deep Learning applied to multimedia content, covering different use cases: from automatic video content analysis to improving video quality, passing by content generation and tools that can improve our working and/or entertainment experiences.

Speakers:

- [Clemente Giorio](https://www.linkedin.com/in/clemente-giorio-03a61811/) (Deltatre, Microsoft MVP)
- [Gianni Rosa Gallina](https://www.linkedin.com/in/gianni-rosa-gallina-b206a821/) (Deltatre, Microsoft MVP)

---

## Setup local environment

Demo have been developed and tested in a **Python 3.9.4** virtual environment, in **Windows 10 21H1**.  
You can setup a working virtual environment on your local machine by cloning this repository and using the following steps (in Powershell):

```powershell
python -m venv .venv
.venv/scripts/activate
python -m pip install -U pip
pip install wheel
pip install -r requirements.txt
```

Once cloned and set up, you can use [Visual Studio Code](https://code.visualstudio.com/) to explore, run and/or debug the code.

## How to run demo on your content

The demo you can find in this repo are based on [MediaPipe](https://google.github.io/mediapipe/) project.
To run the demo you can use these commands (in Powershell):

```powershell
.venv/scripts/activate
$env:PYTHONPATH = "."
python ./src/mediapipe_demo.py --input_file 'PATH_TO_VIDEO_FILE' --model_type 'mediapipe_holistic' --processor_count 4
```

- `processor_count` should be set to a *proper value* depending on the available CPU cores and RAM in your machine. For each processor, a Python sub-process will be instantiated, allocating an instance of MediaPipe model (about 1GB RAM) to process a sub-range of frames of the source video. If set to `-1`, the script will use all available CPU cores to process the video. *Without any special setting or configuration, GPU acceleration is not available*.

- `model_type` can be one of:
    - 'mediapipe_holistic'
    - 'mediapipe_pose'
    - 'mediapipe_objectron'
    - 'mediapipe_face_detection'
    - 'mediapipe_face_mesh'

> Additional models can be easily added to the demo by implementing the corresponding detection model from MediaPipe, following the same approach you can see in the code.

- As additional options:
    - you can tweak model detection and tracking thresholds, using these two parameters (default is 0.5 for both):
        - `min_detection_confidence`
        - `min_tracking_confidence`
    - you can specify where the script saves annotated output videos by setting the `output_folder` parameter. Default is `./outputs` folder.

## License
---

Copyright (C) 2019-2021 Deltatre.  
Licensed under [CC BY-NC-SA 4.0](./LICENSE).
