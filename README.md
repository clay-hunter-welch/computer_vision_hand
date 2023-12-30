# computer vision hand tracking and finger counting

interpreters required:

opencv-python

mediapipe
  
Leveraging OpenCV and mediapipe, there are a few projects here; the most interesting is FingerCountingProject.py which
tracks hand landmarks and then reproduces your finger postures, open or closed.  I experimented with algorithms to 
determine the curled state of each finger, most of them having to do with calculating joint angles.  Ultimately I 
realized that I could just check the distance from a given fingertop to the 0 landmark at the base of the palm, and as
soon as that distance is less than the distance from the finger base to the 0 landmark, it can be called curled.  
The thumb is a little special, but then I guess it's a pretty special digit--thanks, evolution.  
I also got to play with image compositing using opencv; the overlays I am doing are per finger, plus the palm, 
lower arm, and backdrop are all separate elements.  The base images are sourced from Midjourney, but there was some 
Photoshop work required to split out parts and refind the composition.  I wound up using 24 bit pngs and a very 
helpful compositing method I found on stack overflow.  
Finally the code runs the current time through math.sin() with some frequency tweaking to add a little spooky bobbing in 
the robot's disconnected arm, which was fun to figure out how to do.  

Resolved issues:

- distance from camera is not normalized to a unit scale, so "further away" equals "closer together" for trackers, which
  causes issues.  Unit length can be defined as the distance from landmark 0 to landmark 5, and could be used to compensate for this.
- thumb curl determination is tricky and finicky, can improve this.
- there is a Texas A&M Gig'Em/Terminator 2 gesture for thumbs up; this should be differentiated from a normal thumbs out gesture.

Future work:

- currently only one hand tracks, but OpenCV will track multiple hands.  Further, hand orientation could also be mirrored correctly:
  thumb on left vs thumb on right.
