# Decisions

All of these were fixed before any tracking result was looked at.

## One class: vehicle

The task is a traffic count from a drone, so car, van, truck and bus are merged into one class. VisDrone's official tracking benchmark scores five classes separately (pedestrian, car, van, truck, bus). These numbers are therefore not comparable to its leaderboard, and the README says so. Merging also removes a failure that matters little for counting: a van detected as a car in one frame and a van in the next.

## Writing SORT, ByteTrack-style association and camera-motion compensation myself

Libraries exist for all three. Writing them makes every choice visible and testable:
- the Kalman state and its noise
- the matching cost and when a match is rejected
- when a track is born and when it dies
- how the camera's motion moves a track

The tests check each behaviour on a synthetic scene with a known answer, and the code is about 250 lines.

## Detections cached once, trackers compared on the same boxes

Detection is the slow, GPU part; tracking is milliseconds per frame. Caching every detector's boxes (down to confidence 0.05) means each tracker is scored on identical input. Any difference between two trackers is then about association, not about the detector having had a different run.

## Tracker settings fixed, not tuned on test-dev

The thresholds in `scripts/run_tracking.py` are the published defaults of SORT and ByteTrack:
- confidence 0.5 for a detection to start or extend a track; ByteSORT's second pass goes down to 0.1
- IoU 0.3 for a match, and 0.5 in ByteSORT's second pass
- 10 frames before a lost track is dropped
- 3 hits before a track is reported

The same values are used for every detector and both splits. Validation is used to choose between designs. Test-dev is scored once at the end with the same settings.

## Ignored regions are applied to the tracker output, as VisDrone does

Annotators skipped very dense or very distant areas and marked them as ignored. A correct detection there has no label to match, so without this rule it would count as a false positive. Tracker boxes lying mostly (more than half their area) inside an ignored region are dropped before scoring.

## Training at 960 px on a 4 GB GPU

Most VisDrone vehicles are under 30 px wide in a roughly 1,400 px frame. At YOLO's default 640 px many shrink to a few pixels, below what the network's coarsest feature maps can resolve. At 960 px, batch 8 fits in the RTX 3050's 4 GB. Detections are run at the same size.

## The count metric

A traffic survey reports how many vehicles passed. The count error compares the number of distinct track IDs with the number of distinct ground-truth vehicles in each sequence. Every broken track counts a vehicle twice, and every merge hides one. It is the metric closest to what the output would be used for, and the one MOTA hides: a tracker can score well on MOTA while still splitting each vehicle into several tracks.

## A separate Python environment

Fine-tuning needs the CUDA build of PyTorch. The other projects on this machine pin the CPU build, and their published numbers come from it. This project has its own `.venv`, so neither changes the other.
