
# Football Analysis

A comprehensive tool for processing and analyzing video footage, producing detailed insights into gameplay and player performance enhancing game understanding and performance evaluation.

This repository is forked from [mradovic38/football_analysis](https://github.com/mradovic38/football_analysis)

## Changes
1. Self-initialize club color.
2. More tolerent color-based club assignment
* When extracting jersey color, choose more corners to prevent background color's affect.
* Add a moving average filter as a low pass filter.
3. Goalkeeper club assignment
* Color-based -> Position-based
4. Keypoint detection
* Reduce threshold of confidence to prevent missing some occluded key points.

## ⚽ Features
1. Comprehensive Object Detection and Tracking
2. Field Keypoint Detection
3. Player Club Assignment
4. Real-World Position Mapping
5. Dynamic Voronoi Diagram
6. Ball Possession Calculation
7. Speed Estimation
8. Live Video Preview
9. Tracking Data Storage

## ❓ [How to Run](https://github.com/mradovic38/football_analysis/blob/master/README.md)

## License

This project is licensed under the [MIT License](LICENSE). However, it uses the YOLO11 models, which are licensed under the [AGPL-3.0 and Enterprise Licenses](https://www.ultralytics.com/license).
